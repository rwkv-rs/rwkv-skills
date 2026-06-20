import asyncio
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from infer import inference_deps
from infer.batch_inference import BatchInferenceMixin
from infer.big_batch import BigBatchMixin
from infer.cancellation import InferenceCancelled, PrefillBszLimitExceeded
from infer.inference_utils import InferenceUtilsMixin
from infer.metrics import MetricsCollector


class InferenceEngine(
    InferenceUtilsMixin,
    BatchInferenceMixin,
    BigBatchMixin,
):
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="model_inference"
        )
        self._prefill_queue = deque()
        self._prefill_reserved_bsz = 0
        self._prefill_next_ticket = 0
        self._prefill_condition = None
        # Live observability; gpu_sampler is wired in by create_app once started.
        self.metrics = MetricsCollector()

    def _encode_single_token_choices(self, choices):
        if not isinstance(choices, dict) or not choices:
            raise ValueError("choices must be a non-empty object")

        choice_token_ids = {}
        for label, text in choices.items():
            if not isinstance(label, str) or not label:
                raise ValueError("choice labels must be non-empty strings")
            if not isinstance(text, str) or not text:
                raise ValueError(f"choice {label} must be a non-empty string")
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) != 1:
                raise ValueError(
                    f"choice {label}={text!r} must encode to exactly one token, got {token_ids}"
                )
            choice_token_ids[label] = int(token_ids[0])
        return choice_token_ids

    @staticmethod
    def _score_encoded_choices(logits, choice_token_ids, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if logits.dim() != 1:
            raise ValueError(
                f"choice scoring expects 1D logits, got {tuple(logits.shape)}"
            )
        vocab_size = logits.numel()
        for label, token_id in choice_token_ids.items():
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"choice {label} token id {token_id} is outside logits vocab size {vocab_size}"
                )

        scaled_logits = logits.float()
        if temperature != 1.0:
            scaled_logits = scaled_logits / float(temperature)

        choice_logits = {
            label: float(scaled_logits[token_id].item())
            for label, token_id in choice_token_ids.items()
        }
        max_logit = max(choice_logits.values())
        exp_values = {
            label: math.exp(value - max_logit)
            for label, value in choice_logits.items()
        }
        total = sum(exp_values.values())
        if total <= 0:
            raise ValueError("choice logits produced a non-positive softmax denominator")
        choice_probabilities = {
            label: float(value / total)
            for label, value in exp_values.items()
        }
        return choice_logits, choice_probabilities

    def score_choice_logits(
        self,
        prompt,
        choices,
        temperature=1.0,
        prefix_cache_manager=None,
        cancel_token=None,
    ):
        choice_token_ids = self._encode_single_token_choices(choices)
        encoded_prompt, state, out, matched_tokens, cache_source = (
            self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
        )
        try:
            logits = out.float()
            if logits.dim() == 2:
                if logits.size(0) != 1:
                    raise ValueError("choice logits scoring expects a single prompt")
                logits = logits[0]
            if logits.dim() != 1:
                raise ValueError(
                    f"unexpected logits shape for choice scoring: {tuple(logits.shape)}"
                )

            choice_logits, choice_probabilities = self._score_encoded_choices(
                logits,
                choice_token_ids,
                temperature=temperature,
            )
            best_choice = max(choice_logits, key=choice_logits.__getitem__)

            return {
                "prompt_tokens": len(encoded_prompt),
                "matched_prefix_tokens": matched_tokens,
                "cache_source": cache_source,
                "choice_token_ids": choice_token_ids,
                "choice_logits": choice_logits,
                "choice_probabilities": choice_probabilities,
                "best_choice": best_choice,
                "temperature": float(temperature),
            }
        finally:
            del state
            self._cleanup_cuda_memory()

    def score_choice_logits_batch(self, prompts, choices, temperature=1.0, cancel_token=None):
        if not prompts:
            raise ValueError("prompts must be a non-empty list")
        choice_token_ids = self._encode_single_token_choices(choices)
        encoded_prompts = [self.tokenizer.encode(prompt) for prompt in prompts]
        if any(not encoded_prompt for encoded_prompt in encoded_prompts):
            raise ValueError("choice logits prompts must be non-empty")

        state = self.model.generate_zero_state(len(encoded_prompts))
        try:
            out = self._forward_batch_prompts_chunked(
                encoded_prompts,
                state,
                cancel_token=cancel_token,
            ).float()
            if out.dim() == 1:
                if len(encoded_prompts) != 1:
                    raise ValueError(
                        "batched choice scoring returned 1D logits for multiple prompts"
                    )
                out = out.unsqueeze(0)
            if out.dim() != 2 or out.size(0) != len(encoded_prompts):
                raise ValueError(
                    f"unexpected batched logits shape for choice scoring: {tuple(out.shape)}"
                )

            results = []
            for index, logits in enumerate(out):
                choice_logits, choice_probabilities = self._score_encoded_choices(
                    logits,
                    choice_token_ids,
                    temperature=temperature,
                )
                results.append(
                    {
                        "index": index,
                        "prompt_tokens": len(encoded_prompts[index]),
                        "matched_prefix_tokens": 0,
                        "cache_source": None,
                        "choice_token_ids": choice_token_ids,
                        "choice_logits": choice_logits,
                        "choice_probabilities": choice_probabilities,
                        "best_choice": max(choice_logits, key=choice_logits.__getitem__),
                        "temperature": float(temperature),
                    }
                )
            return results
        finally:
            del state
            self._cleanup_cuda_memory()

    def _get_prefill_condition(self):
        if self._prefill_condition is None:
            self._prefill_condition = asyncio.Condition()
        return self._prefill_condition

    async def acquire_prefill_permit(
        self, request_bsz: int, request_label: str = "", cancel_token=None
    ):
        request_bsz = max(1, int(request_bsz))
        max_prefill_bsz_limit = int(
            getattr(
                self.model,
                "max_prefill_bsz_limit",
                getattr(self.model, "max_prefill_bsz", request_bsz),
            )
        )
        if request_bsz > max_prefill_bsz_limit:
            print(
                f"[PrefillQueue] rejected path={request_label} "
                f"request_bsz={request_bsz} max_prefill_bsz_limit={max_prefill_bsz_limit}"
            )
            raise PrefillBszLimitExceeded(request_bsz, max_prefill_bsz_limit)

        condition = self._get_prefill_condition()

        async with condition:
            ticket = self._prefill_next_ticket
            self._prefill_next_ticket += 1
            self._prefill_queue.append(ticket)
            queued_logged = False

            try:
                while True:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise InferenceCancelled("request disconnected while queued")

                    is_turn = self._prefill_queue and self._prefill_queue[0] == ticket
                    if is_turn and hasattr(self.model, "refresh_max_prefill_bsz"):
                        current_limit = self.model.refresh_max_prefill_bsz()
                    else:
                        current_limit = getattr(self.model, "max_prefill_bsz", request_bsz)
                    current_limit = min(int(current_limit), max_prefill_bsz_limit)
                    available_bsz = max(0, int(current_limit) - self._prefill_reserved_bsz)

                    if is_turn and request_bsz <= available_bsz:
                        self._prefill_reserved_bsz += request_bsz
                        self._prefill_queue.popleft()
                        condition.notify_all()
                        print(
                            f"[PrefillQueue] admitted ticket={ticket} path={request_label} "
                            f"request_bsz={request_bsz} reserved_bsz={self._prefill_reserved_bsz} "
                            f"max_prefill_bsz={current_limit}"
                        )
                        return {
                            "ticket": ticket,
                            "request_bsz": request_bsz,
                            "max_prefill_bsz": int(current_limit),
                        }

                    if not queued_logged:
                        ahead = sum(1 for queued_ticket in self._prefill_queue if queued_ticket < ticket)
                        print(
                            f"[PrefillQueue] queued ticket={ticket} path={request_label} "
                            f"request_bsz={request_bsz} requests_ahead={ahead} "
                            f"reserved_bsz={self._prefill_reserved_bsz} max_prefill_bsz={current_limit}"
                        )
                        queued_logged = True

                    try:
                        await asyncio.wait_for(condition.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
            except BaseException:
                if ticket in self._prefill_queue:
                    self._prefill_queue.remove(ticket)
                    condition.notify_all()
                    print(
                        f"[PrefillQueue] removed ticket={ticket} path={request_label} "
                        f"request_bsz={request_bsz}"
                    )
                raise

    async def release_prefill_permit(
        self, request_bsz: int, request_label: str = "", ticket: int | None = None
    ):
        request_bsz = max(1, int(request_bsz))
        condition = self._get_prefill_condition()

        async with condition:
            self._prefill_reserved_bsz = max(0, self._prefill_reserved_bsz - request_bsz)
            current_limit = (
                self.model.refresh_max_prefill_bsz()
                if hasattr(self.model, "refresh_max_prefill_bsz")
                else request_bsz
            )
            max_prefill_bsz_limit = int(
                getattr(
                    self.model,
                    "max_prefill_bsz_limit",
                    getattr(self.model, "max_prefill_bsz", request_bsz),
                )
            )
            current_limit = min(int(current_limit), max_prefill_bsz_limit)
            print(
                f"[PrefillQueue] released ticket={ticket} path={request_label} "
                f"request_bsz={request_bsz} reserved_bsz={self._prefill_reserved_bsz} "
                f"max_prefill_bsz={current_limit}"
            )
            condition.notify_all()
        # A released permit marks a finished request (decode failures are tracked
        # separately via failed_batches).
        self.metrics.record_request(success=True)

    def shutdown(self):
        self.executor.shutdown(wait=False)
