# SPDX-License-Identifier: Apache-2.0
"""The PersonaPlex language model as an SGLang model.

The temporal transformer is SGLang's Llama with paged KV; what is Moshi's
alone lives around it: the 17 summed input embeddings, the depformer that
spells out each frame's agent codes, and the buffers through which the model
runner hands a decode step its fused input row and takes the hidden state
back.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import QuantParamSchema
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    DEPFORMER,
    NUM_AUDIO_STREAMS,
    NUM_STREAMS,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.components.depformer import Depformer

BACKBONE_LAYER_PREFIX = "transformer.layers."


def backbone_weight(
    name: str, tensor: torch.Tensor
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the Llama names for one temporal-transformer tensor."""
    index, _, tail = name[len(BACKBONE_LAYER_PREFIX) :].partition(".")
    base = f"model.layers.{index}."
    if tail == "self_attn.in_proj_weight":
        for proj, shard in zip(
            ("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0), strict=True
        ):
            yield base + f"self_attn.{proj}.weight", shard
    elif tail == "self_attn.out_proj.weight":
        yield base + "self_attn.o_proj.weight", tensor
    elif tail == "norm1.alpha":
        yield base + "input_layernorm.weight", tensor.reshape(-1)
    elif tail == "norm2.alpha":
        yield base + "post_attention_layernorm.weight", tensor.reshape(-1)
    elif tail == "gating.linear_in.weight":
        # Note (wilsonzheng0327): Gate first: the reference activates the first half.
        gate, up = tensor.chunk(2, dim=0)
        yield base + "mlp.gate_proj.weight", gate
        yield base + "mlp.up_proj.weight", up
    elif tail == "gating.linear_out.weight":
        yield base + "mlp.down_proj.weight", tensor
    else:
        raise KeyError(f"unexpected temporal transformer tensor {name!r}")


class PersonaPlexForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.temporal = TEMPORAL_TRANSFORMER
        self.llm = LlamaForCausalLM(
            config=config, quant_config=quant_config, prefix=add_prefix("llm", prefix)
        )
        for layer in self.llm.model.layers:
            layer.self_attn.attn.sliding_window_size = (
                self.get_attention_sliding_window_size()
            )

        dim = self.temporal.dim
        self.audio_emb = nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, dim) for _ in range(NUM_AUDIO_STREAMS)
        )
        self.text_emb = nn.Embedding(TEXT_CARD + 1, dim)
        self.depformer = Depformer(DEPFORMER)

        max_batch = get_schedule().max_running_requests
        dtype = torch.get_default_dtype()
        device = self.text_emb.weight.device
        self.fusion_buffer = torch.zeros(max_batch, dim, dtype=dtype, device=device)
        self.hidden_out = torch.zeros(max_batch, dim, dtype=dtype, device=device)

    def get_attention_sliding_window_size(self) -> int:
        # note (LinzeShi): The full ring masks its oldest slot; the left window
        # excludes the current key, leaving context - 1 visible keys in total.
        return self.temporal.context - 2

    @torch.no_grad()
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """Load validated FP8 KV scales before attention graphs are captured."""
        backbone = self.llm.model
        tp_rank, tp_size = (
            backbone._kv_cache_parallel_layout
        )  # Note (edwardzh): Use Llama's stored KV shard layout; noqa: leading-underscore
        num_layers = backbone.config.num_hidden_layers
        schema = QuantParamSchema.model_validate_json(
            Path(quantization_param_path).read_text(),
            strict=True,
            context={
                "model_type": backbone.config.__class__.model_type,
                "num_hidden_layers": num_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            },
        )
        rank_scales = schema.kv_cache.scaling_factor
        if set(rank_scales) != set(range(tp_size)):
            raise ValueError(f"KV scale ranks must be exactly 0 through {tp_size - 1}")
        else:
            pass
        rounded_scales: dict[int, dict[int, float]] = {}
        for rank, layer_scales in rank_scales.items():
            if set(layer_scales) != set(range(num_layers)):
                raise ValueError(
                    f"KV scale layers for rank {rank} must be exactly 0 through {num_layers - 1}"
                )
            else:
                pass
            rounded_scales[rank] = {}
            for layer_index, scale in layer_scales.items():
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        f"KV scale for rank {rank}, layer {layer_index} must be finite and positive"
                    )
                else:
                    pass
                rounded_scale = torch.tensor(
                    scale, dtype=torch.float32, device="cpu"
                ).item()
                if not math.isfinite(rounded_scale) or rounded_scale <= 0:
                    raise ValueError(
                        f"KV scale for rank {rank}, layer {layer_index} must remain finite and positive in float32"
                    )
                else:
                    pass
                rounded_scales[rank][layer_index] = rounded_scale

        # Note (edwardzh): Validate every layer before changing scales, so a rejected file cannot leave a partially calibrated model.
        updates: list[
            tuple[nn.Module, str, torch.Tensor | None, torch.Tensor, float]
        ] = []
        for layer_index, layer in enumerate(backbone.layers):
            attention = layer.self_attn.attn
            device = layer.self_attn.qkv_proj.weight.device
            rounded_scale = rounded_scales[tp_rank][layer_index]
            for scale_name in ("k_scale", "v_scale"):
                current_scale = getattr(attention, scale_name)
                if current_scale is not None and (
                    not isinstance(current_scale, torch.Tensor)
                    or current_scale.ndim != 0
                    or current_scale.dtype != torch.float32
                    or current_scale.device != device
                ):
                    raise ValueError(
                        f"Layer {layer_index} {scale_name} must be a scalar float32 tensor on {device} or None"
                    )
                else:
                    pass
                prepared_scale = torch.tensor(
                    rounded_scale, dtype=torch.float32, device=device
                )
                updates.append(
                    (
                        attention,
                        scale_name,
                        current_scale,
                        prepared_scale,
                        rounded_scale,
                    )
                )

        for (
            attention,
            scale_name,
            current_scale,
            prepared_scale,
            rounded_scale,
        ) in updates:
            if current_scale is None:
                setattr(attention, scale_name, prepared_scale)
            else:
                current_scale.copy_(prepared_scale)
            # Note (edwardzh): Attention backends consume both tensor and host scalar scales; they must agree after float32 rounding.
            setattr(attention, f"{scale_name}_float", rounded_scale)

    def embed_rows(self, rows_NK: torch.Tensor) -> torch.Tensor:
        """Sum the 17 stream embeddings of each row, in the reference's order
        (agent codebooks, user codebooks, then text) so bf16 rounding matches."""
        assert rows_NK.shape[-1] == NUM_STREAMS, rows_NK.shape
        summed = self.audio_emb[0](rows_NK[:, 1])
        for k in range(1, NUM_AUDIO_STREAMS):
            summed = summed + self.audio_emb[k](rows_NK[:, 1 + k])
        return summed + self.text_emb(rows_NK[:, 0])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **_,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            input_embeds = self.fusion_buffer[: input_ids.shape[0]]
        else:
            pass
        hidden = self.llm.model(input_ids, positions, forward_batch, input_embeds)
        if forward_batch.forward_mode.is_decode():
            self.hidden_out[: hidden.shape[0]] = hidden
        else:
            last_rows = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            self.hidden_out[: last_rows.shape[0]] = hidden[last_rows]
        return self.llm.logits_processor(
            input_ids, hidden, self.llm.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        backbone: list[tuple[str, torch.Tensor]] = []
        depformer: dict[str, torch.Tensor] = {}
        embeddings: dict[str, torch.Tensor] = {}
        required_weights = {"out_norm.alpha", "text_linear.weight", "text_emb.weight"}
        required_weights.update(
            f"{BACKBONE_LAYER_PREFIX}{layer}.{suffix}"
            for layer in range(self.temporal.num_layers)
            for suffix in (
                "self_attn.in_proj_weight",
                "self_attn.out_proj.weight",
                "norm1.alpha",
                "norm2.alpha",
                "gating.linear_in.weight",
                "gating.linear_out.weight",
            )
        )
        for name, tensor in weights:
            required_weights.discard(name)
            if name.startswith(BACKBONE_LAYER_PREFIX):
                backbone.extend(backbone_weight(name, tensor))
            elif name == "out_norm.alpha":
                backbone.append(("model.norm.weight", tensor.reshape(-1)))
            elif name == "text_linear.weight":
                backbone.append(("lm_head.weight", tensor))
            elif name == "text_emb.weight":
                embeddings["text_emb.weight"] = tensor
                # Note (wilsonzheng0327): Never read (rows arrive pre-embedded); filled
                # so no parameter is left uninitialised.
                backbone.append(("model.embed_tokens.weight", tensor[:TEXT_CARD]))
            elif name.startswith("emb."):
                embeddings["audio_" + name] = tensor
            elif name.startswith(("depformer", "linears.")):
                depformer[name] = tensor
            else:
                raise KeyError(f"unexpected PersonaPlex tensor {name!r}")
        if required_weights:
            raise ValueError(
                f"missing PersonaPlex backbone weights: {sorted(required_weights)}"
            )
        else:
            pass
        self.llm.load_weights(backbone)
        missing, unexpected = self.audio_emb.load_state_dict(
            {
                k[len("audio_emb.") :]: v
                for k, v in embeddings.items()
                if k != "text_emb.weight"
            },
            strict=False,
        )
        assert not missing and not unexpected, (missing, unexpected)
        self.text_emb.load_state_dict({"weight": embeddings["text_emb.weight"]})
        self.depformer.load_reference_weights(depformer)


EntryClass = PersonaPlexForCausalLM
