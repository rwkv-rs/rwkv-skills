import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager

from API_servers.router.common import (
    check_password,
    client_closed_response,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    run_sync_with_disconnect_watch,
)
from API_servers.router.schemas import (
    ChatRequest,
    ChoiceLogitsRequest,
    TranslateRequest,
    TranslateResponse,
)


router = APIRouter()


def create_translation_prompt(source_lang, target_lang, text):
    lang_names = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    return f"{source_name}: {text}\n\n{target_name}:"


@router.get("/v1/models")
async def list_models(request: Request):
    engine = request.app.state.engine
    model_name = engine.args.MODEL_NAME.split("/")[-1]
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "owned_by": "rwkv_lightning"}],
    }


@router.post("/v1/choice_logits")
async def choice_logits(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = ChoiceLogitsRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    prompts = [req.prompt] if req.prompt else list(req.contents)
    if not prompts or any(not prompt for prompt in prompts):
        return JSONResponse(status_code=400, content={"error": "Empty prompt"})

    try:
        async with reserve_prefill_capacity(request, len(prompts)):
            if len(prompts) == 1:
                prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None
                result = await run_sync_with_disconnect_watch(
                    request,
                    engine.score_choice_logits,
                    prompt=prompts[0],
                    choices=req.choices,
                    temperature=req.temperature,
                    prefix_cache_manager=prefix_cache_manager,
                )
                return {
                    "id": f"choice_logits-{time.time_ns()}",
                    "object": "choice.logits",
                    "created": int(time.time()),
                    "model": req.model,
                    **result,
                }

            results = await run_sync_with_disconnect_watch(
                request,
                engine.score_choice_logits_batch,
                prompts=prompts,
                choices=req.choices,
                temperature=req.temperature,
            )
            return {
                "id": f"choice_logits-{time.time_ns()}",
                "object": "choice.logits.batch",
                "created": int(time.time()),
                "model": req.model,
                "results": results,
            }
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.post("/v2/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = ChatRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    if req.stream:
        cancel_token = CancellationToken()
        stream = engine.batch_infer_stream(
            prompts=req.contents,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
            ban_tokens=req.ban_tokens,
            chunk_size=req.chunk_size,
            cancel_token=cancel_token,
        )
        return prefill_sse_response(request, stream, cancel_token, len(req.contents))

    try:
        async with reserve_prefill_capacity(request, len(req.contents)):
            results = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate,
                prompts=req.contents,
                max_length=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                stop_tokens=req.stop_tokens,
                ban_tokens=req.ban_tokens,
            )
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)

    choices = []
    for i, text in enumerate(results):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        )

    return {
        "id": "rwkv7-batch",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
    }


@router.post("/translate/v1/batch-translate")
async def batch_translate(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = TranslateRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    prompts = [
        create_translation_prompt(req.source_lang, req.target_lang, text)
        for text in req.text_list
    ]

    try:
        async with reserve_prefill_capacity(request, len(prompts)):
            translated_texts = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate,
                prompts=prompts,
                max_length=2048,
                temperature=1.0,
                top_k=1,
                top_p=0,
                alpha_presence=0,
                alpha_frequency=0,
                alpha_decay=0.996,
                stop_tokens=[],
            )
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except Exception:
        return JSONResponse(
            status_code=500,
            content=TranslateResponse(translations=[]).model_dump(),
        )

    translations_result = []
    for translation in translated_texts:
        translations_result.append(
            {
                "detected_source_lang": req.source_lang if req.source_lang != "auto" else "en",
                "text": translation.strip(),
            }
        )

    return TranslateResponse(translations=translations_result).model_dump()


@router.post("/FIM/v1/batch-FIM")
async def fim_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = ChatRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    prompts = []
    for prefix, suffix in zip(req.prefix, req.suffix):
        prompts.append(f"✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}")

    if req.stream:
        cancel_token = CancellationToken()
        stream = engine.batch_infer_stream(
            prompts=prompts,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=[],
            ban_tokens=req.ban_tokens,
            chunk_size=req.chunk_size,
            cancel_token=cancel_token,
        )
        return prefill_sse_response(request, stream, cancel_token, len(prompts))

    try:
        async with reserve_prefill_capacity(request, len(prompts)):
            results = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate,
                prompts=prompts,
                max_length=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                stop_tokens=[],
                ban_tokens=req.ban_tokens,
            )
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)

    choices = []
    for i, text in enumerate(results):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        )

    return {
        "id": "rwkv7-batch",
        "object": "FIM.completion",
        "model": req.model,
        "choices": choices,
    }


@router.post("/big_batch/completions")
async def big_batch_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = ChatRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    cancel_token = CancellationToken()
    stream = engine.big_batch_stream(
        prompts=req.contents,
        max_length=req.max_tokens,
        temperature=req.temperature,
        stop_tokens=req.stop_tokens,
        chunk_size=req.chunk_size,
        cancel_token=cancel_token,
    )
    return prefill_sse_response(request, stream, cancel_token, len(req.contents))
