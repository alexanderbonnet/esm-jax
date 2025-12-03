from pathlib import Path

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import loguru
import torch
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from transformers import AutoModelForMaskedLM

MODEL_HYPERPARAMS: dict[str, dict[str, int]] = {
    "esm2_t48_15B_UR50D": dict(vocab_size=33, dim=2560, num_layers=48, num_heads=40),
    "esm2_t36_3B_UR50D": dict(vocab_size=33, dim=2560, num_layers=36, num_heads=36),
    "esm2_t33_650M_UR50D": dict(vocab_size=33, dim=1280, num_layers=33, num_heads=20),
    "esm2_t30_150M_UR50D": dict(vocab_size=33, dim=640, num_layers=30, num_heads=20),
    "esm2_t12_35M_UR50D": dict(vocab_size=33, dim=480, num_layers=12, num_heads=20),
    "esm2_t6_8M_UR50D": dict(vocab_size=33, dim=320, num_layers=6, num_heads=20),
}


@jaxtyped(typechecker=beartype)
def gelu(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Matches the implementation in the original ESM repo."""
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


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

    layer_norm: nn.LayerNorm
    query: nn.Linear
    key: nn.Linear
    value: nn.Linear
    output: nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.head_dim = dim // num_heads

        self.layer_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, key=key1)
        self.key = nn.Linear(dim, dim, key=key2)
        self.value = nn.Linear(dim, dim, key=key3)
        self.output = nn.Linear(dim, dim, key=key4)

    def __call__(
        self, seq: Float[Array, "seq dim"], mask: Float[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        seq = jax.vmap(self.layer_norm)(seq)

        query, key, value = map(lambda x: jax.vmap(x)(seq), (self.query, self.key, self.value))
        query, key, value = map(
            lambda x: einops.rearrange(x, "seq (head dim) -> seq head dim", dim=self.head_dim),
            (query, key, value),
        )

        sin, cos = fixed_pos_embedding(dim=self.head_dim, n=seq.shape[0])
        query, key = map(lambda x: apply_rotary_pos_emb(x, sin, cos), (query, key))

        out = jax.nn.dot_product_attention(query=query, key=key, value=value, mask=mask)
        out = einops.rearrange(out, "seq head dim -> seq (head dim)")
        return jax.vmap(self.output)(out)


@jaxtyped(typechecker=beartype)
class FeedForward(eqx.Module):
    layer_norm: nn.LayerNorm
    linear_in: nn.Linear
    linear_out: nn.Linear

    def __init__(self, dim: int, factor: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.layer_norm = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, dim * factor, key=key1)
        self.linear_out = nn.Linear(dim * factor, dim, key=key2)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        x = jax.vmap(self.layer_norm)(x)
        x = jax.vmap(self.linear_in)(x)
        x = gelu(x)
        return jax.vmap(self.linear_out)(x)


@jaxtyped(typechecker=beartype)
class TransformerLayer(eqx.Module):
    attention: MultiHeadAttention
    feed_forward: FeedForward

    def __init__(self, dim: int, num_heads: int, ff_factor: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.attention = MultiHeadAttention(dim, num_heads, key=key1)
        self.feed_forward = FeedForward(dim, ff_factor, key=key2)

    def __call__(
        self, x: Float[Array, "seq dim"], mask: Float[Array, " seq"]
    ) -> Float[Array, "seq dim"]:
        x = x + self.attention(x, mask)
        x = x + self.feed_forward(x)
        return x


@jaxtyped(typechecker=beartype)
class RobertaLMHead(eqx.Module):
    dense: nn.Linear
    layer_norm: nn.LayerNorm
    weight: Float[Array, "out_dim dim"]
    bias: Float[Array, " out_dim"]

    def __init__(
        self, dim: int, out_dim: int, weight: Float[Array, "out_dim dim"], *, key: PRNGKeyArray
    ) -> None:
        self.dense = nn.Linear(dim, dim, key=key)
        self.layer_norm = nn.LayerNorm(dim)
        self.weight = weight
        self.bias = jax.numpy.zeros(out_dim)

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq out_dim"]:
        x = jax.vmap(self.dense)(x)
        x = gelu(x)
        x = jax.vmap(self.layer_norm)(x)
        return x @ self.weight.T + self.bias


@jaxtyped(typechecker=beartype)
class ESM2(eqx.Module):
    embedding: nn.Embedding
    layers: list[TransformerLayer]
    lm_head: RobertaLMHead
    layer_norm: nn.LayerNorm

    def __init__(
        self, vocab_size: int, dim: int, num_layers: int, num_heads: int, *, key: PRNGKeyArray
    ) -> None:
        key1, key2, key3 = jax.random.split(key, 3)

        self.embedding = nn.Embedding(vocab_size, dim, key=key1)

        self.layers = [
            TransformerLayer(dim, num_heads, 4, key=key) for key in jr.split(key2, num_layers)
        ]

        # ESM-2 uses tied weights for the language modeling head
        self.lm_head = RobertaLMHead(
            dim=dim, out_dim=vocab_size, weight=self.embedding.weight, key=key3
        )

        self.layer_norm = nn.LayerNorm(dim)

    def __call__(
        self, tokens: Int[Array, " seq"], mask: Float[Array, " seq"]
    ) -> tuple[Float[Array, "seq vocab"], Float[Array, "seq dim"]]:
        emb = jax.vmap(self.embedding)(tokens)

        # Initially used jax.lax.scan for the loop but compilation is still
        # fast enough and scan may have undesired overhead for GPU jobs.
        for layer in self.layers:
            emb = layer(emb, mask)

        emb = jax.vmap(self.layer_norm)(emb)
        return self.lm_head(emb), emb

    @classmethod
    def from_pretrained(cls, name: str) -> "ESM2":
        weights_path = get_weights_path(name)
        if not weights_path.is_file():
            loguru.logger.info(
                "Weights not yet converted to Equinox, downloading them from "
                "hugging face and converting them from torch."
            )
            convert_weights_from_torch(name)

        # The seed is not important here as there is no stochasticity at runtime.
        # It is only needed to initialize the model before the weights get loaded.
        key = jr.PRNGKey(seed=43)
        config = MODEL_HYPERPARAMS[name]
        model = cls(**config, key=key)

        return eqx.tree_deserialise_leaves(path_or_file=weights_path, like=model)


WEIGHTS_DIR = Path(__file__).parent.parent.parent / "data" / "weights"


def build_conversion_map(num_layers: int) -> tuple[dict[str, str], dict[str, str]]:
    conversion_map = {
        "embedding.weight": "esm.embeddings.word_embeddings.weight",
        "lm_head.bias": "lm_head.bias",
        "lm_head.dense.weight": "lm_head.dense.weight",
        "lm_head.dense.bias": "lm_head.dense.bias",
        "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        "lm_head.weight": "lm_head.decoder.weight",
        "layer_norm.weight": "esm.encoder.emb_layer_norm_after.weight",
        "layer_norm.bias": "esm.encoder.emb_layer_norm_after.bias",
    }
    for k in range(num_layers):
        conversion_map.update(
            {
                f"layers.[{k}].attention.query.weight": f"esm.encoder.layer.{k}.attention.self.query.weight",
                f"layers.[{k}].attention.query.bias": f"esm.encoder.layer.{k}.attention.self.query.bias",
                f"layers.[{k}].attention.key.weight": f"esm.encoder.layer.{k}.attention.self.key.weight",
                f"layers.[{k}].attention.key.bias": f"esm.encoder.layer.{k}.attention.self.key.bias",
                f"layers.[{k}].attention.value.weight": f"esm.encoder.layer.{k}.attention.self.value.weight",
                f"layers.[{k}].attention.value.bias": f"esm.encoder.layer.{k}.attention.self.value.bias",
                f"layers.[{k}].attention.output.weight": f"esm.encoder.layer.{k}.attention.output.dense.weight",
                f"layers.[{k}].attention.output.bias": f"esm.encoder.layer.{k}.attention.output.dense.bias",
                f"layers.[{k}].attention.layer_norm.weight": f"esm.encoder.layer.{k}.attention.LayerNorm.weight",
                f"layers.[{k}].attention.layer_norm.bias": f"esm.encoder.layer.{k}.attention.LayerNorm.bias",
                f"layers.[{k}].feed_forward.linear_in.weight": f"esm.encoder.layer.{k}.intermediate.dense.weight",
                f"layers.[{k}].feed_forward.linear_in.bias": f"esm.encoder.layer.{k}.intermediate.dense.bias",
                f"layers.[{k}].feed_forward.linear_out.weight": f"esm.encoder.layer.{k}.output.dense.weight",
                f"layers.[{k}].feed_forward.linear_out.bias": f"esm.encoder.layer.{k}.output.dense.bias",
                f"layers.[{k}].feed_forward.layer_norm.weight": f"esm.encoder.layer.{k}.LayerNorm.weight",
                f"layers.[{k}].feed_forward.layer_norm.bias": f"esm.encoder.layer.{k}.LayerNorm.bias",
            }
        )
    return conversion_map


def update_eqx_with_state_dict(
    module: eqx.Module,
    state_dict: dict[str, torch.Tensor],
    conversion_map: dict[str, str],
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


def get_weights_path(name: str) -> Path:
    return WEIGHTS_DIR / f"{name}.eqx"


def convert_weights_from_torch(name: str) -> None:
    if name.startswith("facebook"):
        raise ValueError("Remove the leading 'facebook/' from the model's name.")

    eqx_path = get_weights_path(name)
    eqx_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = AutoModelForMaskedLM.from_pretrained("facebook/" + name).state_dict()
    # The key used to initialize the model is not important
    model = ESM2(**MODEL_HYPERPARAMS[name], key=jr.PRNGKey(47))
    conversion_map = build_conversion_map(MODEL_HYPERPARAMS[name]["num_layers"])
    updated_model = update_eqx_with_state_dict(model, state_dict, conversion_map)
    eqx.tree_serialise_leaves(eqx_path, updated_model)
