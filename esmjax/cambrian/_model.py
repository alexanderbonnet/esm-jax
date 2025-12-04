import math
from pathlib import Path

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import loguru
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, jaxtyped

MODEL_HYPERPARAMS: dict[str, dict[str, int]] = {
    "esmc-300m-2024-12": dict(dim=960, num_layers=30, num_heads=15),
    "esmc-600m-2024-12": dict(dim=1152, num_layers=36, num_heads=18),
}


def gelu(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Matches the PyTorch GELU implementation."""
    return jax.nn.gelu(x, approximate=False)


def swiglu(x: Float[Array, "... dim*2"]) -> Float[Array, "... dim"]:
    a, b = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(a) * b


@jaxtyped(typechecker=beartype)
def fixed_pos_embedding(n: int, dim: int) -> tuple[Float[Array, " n dim"], Float[Array, " n dim"]]:
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    frequencies = jnp.einsum("i,j->ij", jnp.arange(n), inv_freq)
    emb = jnp.concatenate([frequencies, frequencies], axis=-1)
    return jnp.sin(emb), jnp.cos(emb)


@jaxtyped(typechecker=beartype)
def rotate_half(x: Float[Array, "seq head dim"]) -> Float[Array, "seq head dim"]:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concat((-x2, x1), axis=-1)


@jaxtyped(typechecker=beartype)
def apply_rotary_pos_emb(
    x: Float[Array, "seq head dim"], sin: Float[Array, "seq dim"], cos: Float[Array, "seq dim"]
) -> Float[Array, "seq head dim"]:
    return (x * cos[:, None, :]) + (rotate_half(x) * sin[:, None, :])


@jaxtyped(typechecker=beartype)
class MultiHeadAttention(eqx.Module):
    head_dim: int

    ln: nn.LayerNorm
    ln_q: nn.LayerNorm
    ln_k: nn.LayerNorm
    to_qkv: nn.Linear
    output: nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        key1, key2 = jr.split(key, 2)

        self.head_dim = dim // num_heads

        self.ln = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(dim, use_bias=False)
        self.ln_k = nn.LayerNorm(dim, use_bias=False)

        self.to_qkv = nn.Linear(dim, dim * 3, use_bias=False, key=key1)
        self.output = nn.Linear(dim, dim, use_bias=False, key=key2)

    def __call__(
        self, seq: Float[Array, "seq dim"], mask: Bool[Array, "seq dim"]
    ) -> Float[Array, "seq dim"]:
        seq = jax.vmap(self.ln)(seq)

        query, key, value = jnp.split(jax.vmap(self.to_qkv)(seq), 3, axis=-1)
        query, key = jax.vmap(self.ln_q)(query), jax.vmap(self.ln_k)(key)
        query, key, value = map(
            lambda x: einops.rearrange(x, "seq (head dim) -> seq head dim", dim=self.head_dim),
            (query, key, value),
        )

        sin, cos = fixed_pos_embedding(n=seq.shape[0], dim=self.head_dim)
        query, key = map(lambda x: apply_rotary_pos_emb(x, sin, cos), (query, key))

        out = jax.nn.dot_product_attention(query=query, key=key, value=value, mask=mask)
        out = einops.rearrange(out, "seq head dim -> seq (head dim)")

        return jax.vmap(self.output)(out)


@jaxtyped(typechecker=beartype)
class FeedForward(eqx.Module):
    layer_norm: nn.LayerNorm
    linear_in: nn.Linear
    linear_out: nn.Linear

    def __init__(self, dim: int, factor: float, *, key: PRNGKeyArray) -> None:
        key1, key2 = jr.split(key, 2)
        factor_int = int((dim * factor + 255) // 256 * 256)

        self.layer_norm = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, factor_int * 2, use_bias=False, key=key1)
        self.linear_out = nn.Linear(factor_int, dim, use_bias=False, key=key2)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        x = jax.vmap(self.layer_norm)(x)
        x = jax.vmap(self.linear_in)(x)
        x = swiglu(x)
        return jax.vmap(self.linear_out)(x)


@jaxtyped(typechecker=beartype)
class TransformerLayer(eqx.Module):
    scale: float

    attention: MultiHeadAttention
    ff: FeedForward

    def __init__(
        self, dim: int, num_heads: int, ff_factor: float, scale: float, *, key: PRNGKeyArray
    ) -> None:
        key1, key2 = jr.split(key, 2)

        self.attention = MultiHeadAttention(dim, num_heads, key=key1)
        self.ff = FeedForward(dim, ff_factor, key=key2)
        self.scale = scale

    def __call__(
        self, x: Float[Array, "seq dim"], mask: Bool[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        x = x + self.attention(x, mask) / self.scale
        x = x + self.ff(x) / self.scale
        return x


class SequenceHead(eqx.Module):
    linear_in: nn.Linear
    layer_norm: nn.LayerNorm
    linear_out: nn.Linear

    def __init__(self, dim: int, *, key: PRNGKeyArray) -> None:
        key1, key2 = jr.split(key, 2)
        self.linear_in = nn.Linear(dim, dim, key=key1)
        self.layer_norm = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, 64, key=key2)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq 64"]:
        x = jax.vmap(self.linear_in)(x)
        x = gelu(x)
        x = jax.vmap(self.layer_norm)(x)
        x = jax.vmap(self.linear_out)(x)
        return x


@jaxtyped(typechecker=beartype)
class ESMC(eqx.Module):
    embedding: nn.Embedding
    sequence_head: SequenceHead
    layer_norm: nn.LayerNorm
    layers: list[TransformerLayer]

    def __init__(self, dim: int, num_layers: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        key1, key2, key3 = jr.split(key, 3)

        self.embedding = nn.Embedding(64, dim, key=key1)
        self.layers = [
            TransformerLayer(
                dim=dim,
                num_heads=num_heads,
                ff_factor=8 / 3,
                scale=math.sqrt(num_layers / 36),
                key=key,
            )
            for key in jr.split(key2, num_layers)
        ]
        self.sequence_head = SequenceHead(dim, key=key3)
        self.layer_norm = nn.LayerNorm(dim, use_bias=False)

    def __call__(
        self, tokens: Int[Array, " seq"], mask: Bool[Array, " seq"]
    ) -> tuple[Float[Array, "seq 64"], Float[Array, "seq dim"]]:
        emb = jax.vmap(self.embedding)(tokens)

        for layer in self.layers:
            emb = layer(emb, mask)

        emb = jax.vmap(self.layer_norm)(emb)
        return self.sequence_head(emb), emb

    @classmethod
    def from_pretrained(cls, name: str) -> "ESMC":
        weights_path = get_weights_path(name)
        if not weights_path.is_file():
            loguru.logger.info(
                "Weights not yet converted to Equinox, downloading them from the "
                "hugging face hub and converting them from torch."
            )
            convert_weights_from_torch(name)

        # The seed is not important here as there is no stochasticity at runtime.
        # It is only needed to initialize the model before the weights get loaded.
        key = jr.PRNGKey(seed=43)
        config = MODEL_HYPERPARAMS[name]
        model = cls(**config, key=key)

        return eqx.tree_deserialise_leaves(path_or_file=weights_path, like=model)


def build_conversion_map(num_layers: int) -> dict[str, str]:
    conversion_map = {
        "embedding.weight": "embed.weight",
        "sequence_head.linear_in.weight": "sequence_head.0.weight",
        "sequence_head.linear_in.bias": "sequence_head.0.bias",
        "sequence_head.layer_norm.weight": "sequence_head.2.weight",
        "sequence_head.layer_norm.bias": "sequence_head.2.bias",
        "sequence_head.linear_out.weight": "sequence_head.3.weight",
        "sequence_head.linear_out.bias": "sequence_head.3.bias",
        "layer_norm.weight": "transformer.norm.weight",
    }
    for k in range(num_layers):
        conversion_map.update(
            {
                f"layers.[{k}].attention.ln.weight": f"transformer.blocks.{k}.attn.layernorm_qkv.0.weight",
                f"layers.[{k}].attention.ln.bias": f"transformer.blocks.{k}.attn.layernorm_qkv.0.bias",
                f"layers.[{k}].attention.to_qkv.weight": f"transformer.blocks.{k}.attn.layernorm_qkv.1.weight",
                f"layers.[{k}].attention.output.weight": f"transformer.blocks.{k}.attn.out_proj.weight",
                f"layers.[{k}].attention.ln_q.weight": f"transformer.blocks.{k}.attn.q_ln.weight",
                f"layers.[{k}].attention.ln_k.weight": f"transformer.blocks.{k}.attn.k_ln.weight",
                f"layers.[{k}].ff.layer_norm.weight": f"transformer.blocks.{k}.ffn.0.weight",
                f"layers.[{k}].ff.layer_norm.bias": f"transformer.blocks.{k}.ffn.0.bias",
                f"layers.[{k}].ff.linear_in.weight": f"transformer.blocks.{k}.ffn.1.weight",
                f"layers.[{k}].ff.linear_out.weight": f"transformer.blocks.{k}.ffn.3.weight",
            }
        )
    return conversion_map


WEIGHTS_DIR = Path(__file__).parent.parent.parent / "data" / "weights"


def get_weights_path(name: str) -> Path:
    return WEIGHTS_DIR / f"{name}.eqx"


def update_eqx_with_state_dict(
    module: eqx.Module, state_dict: dict, conversion_map: dict[str, str]
) -> eqx.Module:
    path_vals, treedef = jax.tree.flatten_with_path(module)
    updated_path_vals, count = [], 0
    array: jnp.ndarray
    for names, array in path_vals:
        key = ".".join(str(x).strip(".") for x in names)
        try:
            weights = state_dict[conversion_map[key]]
            assert array.shape == weights.shape, f"{array.shape} != {weights.shape} for {key=}"
            updated_path_vals.append((names, jnp.asarray(weights)))
            count += 1
        except KeyError:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)

    if not count == len(conversion_map):
        raise ValueError(
            f"Did not find all keys in conversion map: {count=}, {len(conversion_map)=}"
        )
    return updated_module


def convert_weights_from_torch(name: str) -> None:
    import huggingface_hub as hf_hub
    import torch

    if name.startswith("facebook"):
        raise ValueError("Remove the leading 'facebook/' from the model's name.")

    eqx_path = get_weights_path(name)
    eqx_path.parent.mkdir(parents=True, exist_ok=True)

    download_path = hf_hub.hf_hub_download(
        repo_id=f"EvolutionaryScale/{name}",
        filename=f"data/weights/{name}_v0.pth".replace("-", "_"),
    )
    state_dict = torch.load(download_path)
    # The key used to initialize the model is not important
    model = ESMC(**MODEL_HYPERPARAMS[name], key=jr.PRNGKey(52))
    conversion_map = build_conversion_map(MODEL_HYPERPARAMS[name]["num_layers"])
    updated_model = update_eqx_with_state_dict(model, state_dict, conversion_map)
    eqx.tree_serialise_leaves(eqx_path, updated_model)
