"""CLaRa Model Implementation.

Implements Apple's CLaRa architecture with memory token compression.
"""

import json
import os
import warnings
from typing import Any, cast

import torch
from peft import LoraConfig, TaskType
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lamb.bridge import DocumentCompressor
from lamb.config import ClaraConfig
from lamb.utils import pick_attn_implementation, pick_dtype


class ClaraModel(nn.Module):
    """CLaRa: Compressing Language for Retrieval Augmentation.

    Architecture:
    - Single base LLM used for both encoding and decoding
    - LoRA adapters for encoder and decoder modes
    - Memory tokens added to vocabulary
    - Document compressor projects encoder outputs to memory embeddings
    """

    def __init__(self, config: ClaraConfig):
        super().__init__()
        self.config = config

        print(f"[CLaRa] Loading base model: {config.model_name}...")

        # Load base model
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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if getattr(config, "chat_template", ""):
            template_name = str(config.chat_template)
            temp_tokenizer = AutoTokenizer.from_pretrained(
                template_name,
                trust_remote_code=True,
            )
            if not getattr(temp_tokenizer, "chat_template", None):
                raise ValueError(f"chat_template={template_name!r} did not provide a chat_template")
            self.tokenizer.chat_template = temp_tokenizer.chat_template
            print(f"[CLaRa] tokenizer.chat_template copied from {template_name}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add memory tokens to vocabulary
        self._add_memory_tokens()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add LoRA adapters
        self._add_lora_adapters()

        # Create document compressor
        hidden_size = self.base_model.config.hidden_size

        # Use original compression (decoder with mem tokens in input) or custom (cross-attention)
        if config.use_clara_original:
            self.compressor = None  # Original method doesn't use separate compressor
            print("[CLaRa] Using original compression: memory tokens in input sequence")
        else:
            self.compressor = DocumentCompressor(
                hidden_size=hidden_size,
                num_memory_tokens=config.compress_rate,
                use_mlp=config.use_compressor_mlp,
                mlp_hidden_dim=config.compressor_mlp_hidden_dim,
                encoder_pool_method=config.encoder_pool_method,
            ).to(device=device, dtype=dtype)
            print("[CLaRa] Using custom compression: cross-attention pooling")

        print(f"[CLaRa] Model initialized. Stage: {config.stage}")

    def _add_memory_tokens(self) -> None:
        """Add memory tokens (<mem_0>, <mem_1>, ...) to tokenizer and model."""
        num_mem_tokens = self.config.compress_rate

        # Create memory token strings
        mem_tokens = [f"<mem_{i}>" for i in range(num_mem_tokens)]

        # Add separator token if using original CLaRa format
        special_tokens = mem_tokens.copy()
        if self.config.use_clara_original:
            special_tokens.append("<SEP>")

        # Add to tokenizer
        num_added = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print(f"[CLaRa] Added {num_added} memory tokens to vocabulary")

        # Store separator token if added
        if self.config.use_clara_original:
            self.sep_token = "<SEP>"
            self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<SEP>")
            print(f"[CLaRa] Added <SEP> separator token (id={self.sep_token_id})")
        else:
            self.sep_token = None
            self.sep_token_id = None

        # Resize model embeddings
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Initialize new token embeddings randomly
        vocab_size_original = len(self.tokenizer) - num_added
        embed_layer = cast(nn.Embedding, self.base_model.get_input_embeddings())

        with torch.no_grad():
            # Initialize memory token embeddings with small random values
            embed_layer.weight[vocab_size_original:] = (
                torch.randn(
                    num_added,
                    embed_layer.embedding_dim,
                    dtype=embed_layer.weight.dtype,
                    device=embed_layer.weight.device,
                )
                * 0.02
            )

        # Make memory token embeddings trainable
        embed_layer.weight.requires_grad = True

        # Store memory token IDs for easy access
        self.mem_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(f"<mem_{i}>") for i in range(num_mem_tokens)],
            dtype=torch.long,
        )

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters for encoder and decoder."""
        existing_cfg = getattr(self.base_model, "peft_config", None)
        existing_adapters: set[str] = set()
        if isinstance(existing_cfg, dict):
            existing_adapters = set(existing_cfg.keys())

        want = {"encoder_adapter", "decoder_adapter"}
        if want.issubset(existing_adapters):
            print("[CLaRa] Existing LoRA adapters detected; skipping adapter registration")
            return

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        # Encoder adapter (for document compression)
        encoder_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.encoder_lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )
        if "encoder_adapter" not in existing_adapters:
            self.base_model.add_adapter(encoder_config, adapter_name="encoder_adapter")

        # Decoder adapter (for generation)
        decoder_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.decoder_lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )
        if "decoder_adapter" not in existing_adapters:
            # PEFT warns when the model already has a `peft_config` attribute.
            # In our case, multiple adapters are intentional.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Already found a `peft_config` attribute in the model\.",
                    category=UserWarning,
                )
                self.base_model.add_adapter(decoder_config, adapter_name="decoder_adapter")

        print(
            f"[CLaRa] Added LoRA adapters (encoder r={self.config.encoder_lora_rank}, decoder r={self.config.decoder_lora_rank})"
        )

    def compress_documents(
        self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress documents into memory token embeddings.

        Args:
            doc_input_ids: [batch * num_docs, doc_seq_len]
            doc_attention_mask: [batch * num_docs, doc_seq_len]

        Returns:
            memory_embeddings: [batch * num_docs, num_memory_tokens, hidden_size]
            encoder_hidden_states: [batch * num_docs, seq_len, hidden_size]
        """
        if self.config.use_clara_original:
            return self._compress_original(doc_input_ids, doc_attention_mask)
        else:
            return self._compress_custom(doc_input_ids, doc_attention_mask)

    def _compress_custom(
        self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom compression: cross-attention pooling (our method)."""
        # Set encoder adapter
        self.base_model.set_adapter("encoder_adapter")

        # Encode documents
        with torch.set_grad_enabled(self.training):
            encoder_outputs = self.base_model(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last hidden state
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        # Compress to memory embeddings using cross-attention
        memory_embeddings = self.compressor(
            encoder_hidden_states,
            attention_mask=doc_attention_mask,
        )

        return memory_embeddings, encoder_hidden_states

    def _compress_original(
        self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original CLaRa compression: append memory tokens to input, extract their embeddings."""
        # Append memory token IDs to input sequence
        num_mem_tokens = self.config.compress_rate
        batch_size = doc_input_ids.size(0)

        # Create memory token IDs [batch, num_mem_tokens]
        mem_token_ids = (
            self.mem_token_ids.unsqueeze(0).expand(batch_size, -1).to(doc_input_ids.device)
        )

        # Concatenate: [batch, doc_seq_len + num_mem_tokens]
        input_ids_with_mem = torch.cat([doc_input_ids, mem_token_ids], dim=1)
        attention_mask_with_mem = torch.cat(
            [
                doc_attention_mask,
                torch.ones(batch_size, num_mem_tokens, device=doc_attention_mask.device),
            ],
            dim=1,
        )

        # Set encoder adapter
        self.base_model.set_adapter("encoder_adapter")

        # Forward through decoder with memory tokens in sequence
        with torch.set_grad_enabled(self.training):
            outputs = self.base_model(
                input_ids=input_ids_with_mem,
                attention_mask=attention_mask_with_mem,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last hidden state: [batch, doc_seq_len + num_mem_tokens, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Extract ONLY memory token positions (last num_mem_tokens)
        memory_embeddings = hidden_states[
            :, -num_mem_tokens:, :
        ]  # [batch, num_mem_tokens, hidden_size]

        # Return full hidden states for MSE loss computation
        return memory_embeddings, hidden_states

    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass with memory token replacement.

        Args:
            input_ids: [batch, seq_len] - contains memory token IDs as placeholders
            attention_mask: [batch, seq_len]
            memory_embeddings: [batch, num_docs * num_mem_tokens, hidden_size]
            labels: [batch, seq_len] - optional, for computing loss

        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        # Set decoder adapter
        self.base_model.set_adapter("decoder_adapter")

        # Replace memory token IDs with actual embeddings
        inputs_embeds = self._replace_memory_tokens(input_ids, memory_embeddings)

        # Forward through decoder
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
        }

    def _replace_memory_tokens(
        self,
        input_ids: torch.Tensor,
        memory_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Replace memory token IDs with actual compressed embeddings.

        Args:
            input_ids: [batch, seq_len]
            memory_embeddings: [batch, total_mem_tokens, hidden_size]
                where total_mem_tokens = num_docs * compress_rate

        Returns:
            inputs_embeds: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        embed_layer = self.base_model.get_input_embeddings()

        # Get base embeddings for all tokens

        inputs_embeds = embed_layer(input_ids)  # [batch, seq_len, hidden_size]
        # Find positions of memory tokens and replace
        mem_token_ids_set = set(self.mem_token_ids.tolist())
        replaced = 0
        for b in range(batch_size):
            mem_idx = 0
            for s in range(seq_len):
                token_id = input_ids[b, s].item()
                if token_id in mem_token_ids_set and mem_idx < memory_embeddings.size(1):
                    replaced += 1
                    inputs_embeds[b, s] = memory_embeddings[b, mem_idx]
                    mem_idx += 1
        # print_decode(self.tokenizer, input_ids)
        assert replaced == memory_embeddings.size(1) * batch_size, (
            f"Replaced {replaced} memory tokens, expected {memory_embeddings.size(1) * batch_size}"
        )

        return inputs_embeds

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get all trainable parameters."""
        trainable = []

        # LoRA parameters (automatically trainable via PEFT)
        for _name, param in self.base_model.named_parameters():
            if param.requires_grad:
                trainable.append(param)

        # Compressor parameters
        for param in self.compressor.parameters():
            if param.requires_grad:
                trainable.append(param)

        return trainable

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""

        os.makedirs(path, exist_ok=True)

        # Save LoRA adapters
        self.base_model.save_pretrained(path)

        # Save compressor
        torch.save(self.compressor.state_dict(), os.path.join(path, "compressor.pt"))

        # Save tokenizer (with memory tokens)
        self.tokenizer.save_pretrained(path)

        # Save config

        with open(os.path.join(path, "clara_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        print(f"[CLaRa] Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, config: ClaraConfig | None = None) -> "ClaraModel":
        """Load model from checkpoint."""

        # Load config if not provided
        if config is None:
            with open(os.path.join(path, "clara_config.json")) as f:
                config_dict = json.load(f)
            config = ClaraConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load LoRA adapters
        # (PEFT handles this automatically when we loaded from pretrained)

        # Load compressor
        compressor_path = os.path.join(path, "compressor.pt")
        if os.path.exists(compressor_path):
            model.compressor.load_state_dict(
                torch.load(compressor_path, map_location=config.device)
            )

        print(f"[CLaRa] Checkpoint loaded from {path}")
        return model
