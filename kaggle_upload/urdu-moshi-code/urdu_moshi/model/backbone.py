import abc
from typing import Dict, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from configs.model_config import Qwen25BackboneConfig, AbstractBackboneConfig


class AbstractTemporalBackbone(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def max_seq_len(self) -> int: ...

    @abc.abstractmethod
    def __call__(
        self,
        embedded_tokens: jnp.ndarray,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            embedded_tokens: (batch, seq_len, embed_dim) - sum of K codebook embeddings
            train: whether in training mode

        Returns:
            context_vectors: (batch, seq_len, model_dim) - used by Depth Transformer
            text_logits: (batch, seq_len, vocab_size) - text token prediction logits
        """
        ...


class Qwen25RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        x_f32 = x.astype(jnp.float32)
        norm = x_f32 * jax.lax.rsqrt(
            jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps
        )
        return (norm * weight).astype(self.dtype)


class Qwen25RotaryEmbedding(nn.Module):
    dim: int
    base: float = 1_000_000.0
    dtype: jnp.dtype = jnp.bfloat16

    def __call__(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        half_dim = self.dim // 2
        freqs = 1.0 / (
            self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
        )
        t = jnp.arange(seq_len, dtype=jnp.float32)
        angles = jnp.outer(t, freqs)
        cos = jnp.cos(angles).astype(self.dtype)
        sin = jnp.sin(angles).astype(self.dtype)
        return cos, sin


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    return x * cos + rotate_half(x) * sin


class Qwen25GQAttention(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    rope_theta: float = 1_000_000.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.k_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.v_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype)
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        self.rope = Qwen25RotaryEmbedding(dim=self.head_dim, base=self.rope_theta, dtype=self.dtype)

    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        cos, sin = self.rope(T)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = jnp.repeat(k, self.kv_groups, axis=2)
        v = jnp.repeat(v, self.kv_groups, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(causal_mask[None, None], attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(self.dtype)

        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class Qwen25MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen25DecoderLayer(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.self_attn = Qwen25GQAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )
        self.mlp = Qwen25MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
        )
        self.input_layernorm = Qwen25RMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)
        self.post_attention_layernorm = Qwen25RMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)

    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        x = x + self.self_attn(self.input_layernorm(x), train)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen25FlaxBackbone(AbstractTemporalBackbone):
    config: Qwen25BackboneConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        cfg = self.config
        self.embed_tokens = nn.Embed(cfg.vocab_size, cfg.hidden_size, dtype=self.dtype)
        self.layers = [
            Qwen25DecoderLayer(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                intermediate_size=cfg.intermediate_size,
                rms_norm_eps=cfg.rms_norm_eps,
                rope_theta=cfg.rope_theta,
                dtype=self.dtype,
            )
            for _ in range(cfg.num_hidden_layers)
        ]
        self.norm = Qwen25RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, self.dtype)
        self.lm_head = nn.Dense(cfg.vocab_size, use_bias=False, dtype=self.dtype)

    @property
    def model_dim(self) -> int:
        return self.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.config.max_position_embeddings

    def embed(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        return self.embed_tokens(token_ids)

    def __call__(
        self,
        embedded_tokens: jnp.ndarray,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = embedded_tokens.astype(self.dtype)

        if self.config.gradient_checkpointing and train:
            for layer in self.layers:
                x = jax.checkpoint(layer)(x, train)
        else:
            for layer in self.layers:
                x = layer(x, train)

        context_vectors = self.norm(x)
        text_logits = self.lm_head(context_vectors)
        return context_vectors, text_logits


def load_qwen25_pretrained_weights(
    config: Qwen25BackboneConfig,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM
        import torch
    except ImportError:
        raise ImportError("transformers and torch required. pip install transformers torch")

    print(f"Downloading {config.hf_model_id} weights...")
    pt_model = AutoModelForCausalLM.from_pretrained(
        config.hf_model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pt_state = pt_model.state_dict()
    flax_params = _convert_qwen25_weights(pt_state, config)
    del pt_model
    print("Weight conversion complete.")
    return flax_params


def _convert_qwen25_weights(
    pt_state: Dict,
    config: Qwen25BackboneConfig,
) -> Dict:
    params = {"params": {}}

    def arr(key):
        return jnp.array(pt_state[key].float().numpy(), dtype=jnp.bfloat16)

    params["params"]["embed_tokens"] = {
        "embedding": arr("model.embed_tokens.weight")
    }

    params["params"]["norm"] = {"weight": arr("model.norm.weight")}

    if "lm_head.weight" in pt_state:
        params["params"]["lm_head"] = {"kernel": arr("lm_head.weight").T}
    else:
        params["params"]["lm_head"] = {
            "kernel": arr("model.embed_tokens.weight").T
        }

    params["params"]["layers"] = {}
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        layer_params = {
            "input_layernorm": {"weight": arr(f"{prefix}.input_layernorm.weight")},
            "post_attention_layernorm": {"weight": arr(f"{prefix}.post_attention_layernorm.weight")},
            "self_attn": {
                "q_proj": {
                    "kernel": arr(f"{prefix}.self_attn.q_proj.weight").T,
                    "bias": arr(f"{prefix}.self_attn.q_proj.bias"),
                },
                "k_proj": {
                    "kernel": arr(f"{prefix}.self_attn.k_proj.weight").T,
                    "bias": arr(f"{prefix}.self_attn.k_proj.bias"),
                },
                "v_proj": {
                    "kernel": arr(f"{prefix}.self_attn.v_proj.weight").T,
                    "bias": arr(f"{prefix}.self_attn.v_proj.bias"),
                },
                "o_proj": {"kernel": arr(f"{prefix}.self_attn.o_proj.weight").T},
            },
            "mlp": {
                "gate_proj": {"kernel": arr(f"{prefix}.mlp.gate_proj.weight").T},
                "up_proj": {"kernel": arr(f"{prefix}.mlp.up_proj.weight").T},
                "down_proj": {"kernel": arr(f"{prefix}.mlp.down_proj.weight").T},
            },
        }
        params["params"]["layers"][str(i)] = layer_params

    return params
