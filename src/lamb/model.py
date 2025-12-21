# type: ignore
from typing import Any, cast

import torch
from peft import LoraConfig, TaskType
from torch import nn
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from lamb.bridge import VerticalLatentMemoryBridge
from lamb.config import LaMBConfig
from lamb.utils import pick_attn_implementation, pick_dtype


class LaMBModel(nn.Module):
    def __init__(self, config: LaMBConfig):
        super().__init__()
        self.config = config
        self.base_model: Any  # PEFT-enhanced model with adapter methods
        print(f"[Model] Loading {config.model_name}...")

        dtype = (
            config.dtype
            if isinstance(getattr(config, "dtype", None), torch.dtype)
            else pick_dtype(config.device)
        )
        attn_impl = getattr(config, "attn_implementation", None) or pick_attn_implementation(
            config.device
        )
        device = torch.device(config.device)

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=dtype,
                **model_kwargs,
            )
        except TypeError:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                **model_kwargs,
            )

        cast(torch.nn.Module, self.base_model).to(device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

        for param in self.base_model.parameters():
            param.requires_grad = False
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        self.has_compressor = config.compressor_lora_rank > 0
        if self.has_compressor:
            compressor_cfg = LoraConfig(  # type: ignore
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.compressor_lora_rank,
                lora_alpha=config.compressor_lora_alpha,
                target_modules=target_modules,
            )
            self.base_model.add_adapter(compressor_cfg, adapter_name="compressor")

        self.bridge = VerticalLatentMemoryBridge(self.base_model.config).to(
            config.device, dtype=self.base_model.dtype
        )

        embed_dim = self.base_model.config.hidden_size
        self.memory_input_embeds = nn.Parameter(
            torch.randn(1, config.num_memory_tokens, embed_dim, dtype=self.base_model.dtype).to(
                config.device
            )
            * 0.02
        )

        self.rotary_emb = None
        print("[Model] Initializing Bridge with Pre-trained K/V Projections...")

        layers = None
        if hasattr(self.base_model, "model"):
            layers = self.base_model.model.layers
        elif hasattr(self.base_model, "layers"):
            layers = self.base_model.layers

        if layers is not None:
            for i, layer in enumerate(layers):
                if hasattr(layer, "self_attn"):
                    attn = layer.self_attn
                    k_weight = attn.k_proj.weight.data
                    v_weight = attn.v_proj.weight.data
                    init_weight = torch.cat([k_weight, v_weight], dim=0)

                    if self.bridge.projectors[i].weight.shape == init_weight.shape:
                        self.bridge.projectors[i].weight.data.copy_(init_weight)  # type: ignore[operator]
                    else:
                        print(
                            f"!! Warning: Shape mismatch at layer {i}. "
                            f"Bridge: {self.bridge.projectors[i].weight.shape}, Init: {init_weight.shape}"
                        )

        for name, module in self.base_model.named_modules():
            if "rotary_emb" in name or "RotaryEmbedding" in module.__class__.__name__:
                print(f"[Model] Found rotary embedding module: {name}")
                self.rotary_emb = module
                break

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self.named_parameters() if p.requires_grad]

    def compress(
        self, context_ids: torch.Tensor, context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self.has_compressor:
            self.base_model.set_adapter("compressor")
        ctx_embeds = self.base_model.get_input_embeddings()(context_ids)
        mem_embeds = self.memory_input_embeds.expand(context_ids.shape[0], -1, -1)
        inputs_embeds = torch.cat([ctx_embeds, mem_embeds], dim=1)
        mem_mask = torch.ones(context_ids.shape[0], self.config.num_memory_tokens).to(
            self.config.device
        )
        attention_mask = torch.cat([context_mask, mem_mask], dim=1)

        outputs = self.base_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True
        )
        layers_to_use = outputs.hidden_states[1:]

        extracted_states = []
        for layer_out in layers_to_use:
            extracted_states.append(layer_out[:, -self.config.num_memory_tokens :, :])

        return tuple(extracted_states)

    def forward(  # noqa: PLR0915
        self, full_input_ids: torch.Tensor, split_indices: list[int], return_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        split_idx = split_indices[0]
        context_ids = full_input_ids[:, :split_idx]
        target_ids = full_input_ids[:, split_idx:]
        context_mask = (context_ids != self.tokenizer.pad_token_id).long()

        alpha = float(getattr(self.config, "ce_alpha", 0.0))
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        # If alpha==1.0, KL weight is 0 so we can skip the teacher forward pass entirely.
        t_logits: torch.Tensor | None = None
        if alpha < 1.0:
            if self.has_compressor:
                self.base_model.disable_adapters()
            with torch.no_grad():
                t_out = self.base_model(input_ids=full_input_ids)
                t_logits = t_out.logits[:, split_idx - 1 : -1, :].clone()
                del t_out

        layer_latents = self.compress(context_ids, context_mask)
        past_key_values = self.bridge(layer_latents, rotary_module=self.rotary_emb)

        if self.has_compressor:
            self.base_model.disable_adapters()
        s_out = self.base_model(
            input_ids=target_ids, past_key_values=past_key_values, use_cache=False
        )
        s_logits = s_out.logits[:, :-1, :]

        if alpha < 1.0:
            if t_logits is None:
                raise RuntimeError("Expected teacher logits when ce_alpha < 1.0")
            temp = float(self.config.kl_temperature)
            s_log_probs = functional.log_softmax((s_logits.float() / temp), dim=-1)
            t_probs = functional.softmax((t_logits[:, 1:, :].float() / temp), dim=-1)
            kl_per_vocab = functional.kl_div(
                s_log_probs, t_probs, reduction="none", log_target=False
            )
            kl_per_token = kl_per_vocab.sum(dim=-1)
            kl_loss = kl_per_token.mean() * (temp**2)
        else:
            kl_loss = s_logits.detach().mean() * 0.0

        gold_next = target_ids[:, 1:]
        if gold_next.numel() == 0:
            ce_loss = kl_loss.detach() * 0.0
        else:
            ignore_idx = (
                int(self.tokenizer.pad_token_id)
                if self.tokenizer.pad_token_id is not None
                else -100
            )
            ce_loss = functional.cross_entropy(
                s_logits.reshape(-1, s_logits.size(-1)).float(),
                gold_next.reshape(-1),
                ignore_index=ignore_idx,
            )
        loss = (1.0 - alpha) * kl_loss + alpha * ce_loss

        if not return_metrics:
            return loss

        with torch.no_grad():
            student_pred = s_logits.argmax(dim=-1)
            ignore_idx = (
                int(self.tokenizer.pad_token_id)
                if self.tokenizer.pad_token_id is not None
                else -100
            )

            if gold_next.numel() == 0:
                token_correct = 0
                token_total = 0
            else:
                valid = gold_next != ignore_idx
                token_correct = int(((student_pred == gold_next) & valid).sum().item())
                token_total = int(valid.sum().item())
            token_acc = (float(token_correct) / float(token_total)) if token_total else 0.0

            teacher_student_correct: int | None = None
            teacher_student_total: int | None = None
            teacher_student_acc: float | None = None
            if alpha < 1.0 and t_logits is not None:
                teacher_pred = t_logits[:, 1:, :].argmax(dim=-1)
                valid = (
                    gold_next != ignore_idx
                    if gold_next.numel()
                    else torch.ones_like(student_pred, dtype=torch.bool)
                )
                teacher_student_correct = int(((teacher_pred == student_pred) & valid).sum().item())
                teacher_student_total = int(valid.sum().item())
                teacher_student_acc = (
                    float(teacher_student_correct) / float(teacher_student_total)
                    if teacher_student_total
                    else 0.0
                )

        metrics: dict[str, float | int] = {
            "token_acc": float(token_acc),
            "token_correct": int(token_correct),
            "token_total": int(token_total),
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "ce_alpha": float(alpha),
        }

        if alpha < 1.0:
            metrics["kl_loss"] = float(kl_loss.detach().cpu())
            if (
                teacher_student_acc is not None
                and teacher_student_correct is not None
                and teacher_student_total is not None
            ):
                metrics["teacher_student_acc"] = float(teacher_student_acc)
                metrics["teacher_student_correct"] = int(teacher_student_correct)
                metrics["teacher_student_total"] = int(teacher_student_total)
        return loss, metrics
