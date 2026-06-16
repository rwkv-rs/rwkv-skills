import atexit
import os
import socket
from dataclasses import fields
from os import PathLike
from time import perf_counter
from typing import Mapping
from uuid import uuid4

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.state_cache import StatePrefixIndex, StateSlotManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.tokenizers import get_rwkv_tokenizer


def _allocate_tcp_init_method() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return f"tcp://127.0.0.1:{port}"


def _make_shared_memory_name() -> str:
    return f"nanovllm-{os.getpid()}-{uuid4().hex}"


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.tensor_parallel_size > 1:
            config.distributed_init_method = _allocate_tcp_init_method()
            config.shared_memory_name = _make_shared_memory_name()
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = get_rwkv_tokenizer()
        config.eos = self.tokenizer.eos_token_id
        config.stop_token_seqs = self.tokenizer.get_default_stop_token_seqs()
        self.model_runner.eos = config.eos
        self.model_runner.stop_token_seqs = config.stop_token_seqs
        self.scheduler = Scheduler(config)
        if config.rwkv_state_cache_enable:
            self.scheduler.prefix_index.cache_key_token_rewriter = self.tokenizer.canonicalize_state_cache_token_ids
            self.model_runner.attach_state_cache(self.scheduler.slot_manager, self.scheduler.prefix_index)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.allow_sparse_penalty_state = True
        self.scheduler.add(seq)
        return seq

    def abort(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def update_weights(self, weights: str | PathLike | Mapping[str, torch.Tensor]):
        if not self.scheduler.is_finished():
            raise RuntimeError("nano-vllm-rwkv hot weight update requires an idle scheduler.")
        if getattr(self.model_runner, "world_size", 1) > 1 and not isinstance(weights, (str, PathLike)):
            raise ValueError("Tensor-parallel hot weight update requires a shared .pth path.")
        self.model_runner.call("update_weights", weights)
        self._reset_scheduler_after_weight_update()

    def _reset_scheduler_after_weight_update(self):
        scheduler_config = getattr(self.scheduler, "config", None)
        rwkv_state_cache_enable = getattr(
            self.scheduler,
            "rwkv_state_cache_enable",
            getattr(scheduler_config, "rwkv_state_cache_enable", False),
        )
        if scheduler_config is None:
            return
        if rwkv_state_cache_enable:
            rewriter = self.tokenizer.canonicalize_state_cache_token_ids
            self.scheduler.slot_manager = StateSlotManager(scheduler_config.num_state_blocks)
            self.scheduler.prefix_index = StatePrefixIndex(cache_key_token_rewriter=rewriter)
            self.model_runner.attach_state_cache(self.scheduler.slot_manager, self.scheduler.prefix_index)
            return
        self.scheduler.block_manager = BlockManager(scheduler_config.num_state_blocks)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        prefill_tokens = (
            sum(seq.prefill_step_tokens(self.scheduler.config.rwkv_prefill_chunk_size) for seq in seqs)
            if is_prefill
            else 0
        )
        capture_logprobs = any(getattr(seq, "capture_logprobs", False) for seq in seqs)
        if capture_logprobs:
            token_ids, token_log_probs = self.model_runner.call("run_with_logprobs", seqs, is_prefill)
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            token_log_probs = None
        self.scheduler.postprocess(seqs, token_ids)
        if token_log_probs is not None:
            for seq, token_id, log_prob in zip(seqs, token_ids, token_log_probs, strict=True):
                if getattr(seq, "capture_logprobs", False) and token_id is not None and log_prob is not None:
                    seq.completion_log_probs.append(float(log_prob))
        outputs = [
            (
                seq.seq_id,
                seq.raw_completion_token_ids,
                list(seq.completion_log_probs),
            )
            if getattr(seq, "capture_logprobs", False)
            else (seq.seq_id, seq.raw_completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        num_tokens = prefill_tokens if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        return_logprobs: bool = False,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)
            seq.capture_logprobs = return_logprobs
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for item in output:
                if len(item) == 3:
                    seq_id, token_ids, log_probs = item
                    outputs[seq_id] = {"token_ids": token_ids, "log_probs": log_probs}
                else:
                    seq_id, token_ids = item
                    outputs[seq_id] = {"token_ids": token_ids}
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(output["token_ids"]),
                **output,
            }
            for output in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
