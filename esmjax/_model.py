import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def gelu(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Matches the implementation in the original ESM repo."""
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def fixed_pos_embedding_jax(
    dim: int, n: int
) -> tuple[Float[Array, " seq dim"], Float[Array, " seq dim"]]:
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    frequencies = jnp.einsum("i,j->ij", jnp.arange(n), inv_freq)
    emb = jnp.concatenate([frequencies, frequencies], axis=-1)
    return jnp.sin(emb), jnp.cos(emb)


def rotate_half_jax(x: Float[Array, "head dim"]) -> Float[Array, "head dim"]:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb_jax(
    x: Float[Array, "head seq dim"], sin: Float[Array, "seq dim"], cos: Float[Array, "seq dim"]
) -> Float[Array, "head seq dim"]:
    return (x * cos[None, :, :]) + (rotate_half_jax(x) * sin[None, :, :])


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

    def __call__(self, seq: Float[Array, "seq dim"], mask: Float[Array, " seq"]) -> None:
        n, _ = seq.shape

        seq = jax.vmap(self.layer_norm)(seq)

        query = jax.vmap(self.query)(seq)
        key = jax.vmap(self.key)(seq)
        value = jax.vmap(self.value)(seq)

        pattern = "seq (head dim) -> head seq dim"
        query = einops.rearrange(query, pattern, dim=self.head_dim)
        key = einops.rearrange(key, pattern, dim=self.head_dim)
        value = einops.rearrange(value, pattern, dim=self.head_dim)

        sin, cos = fixed_pos_embedding_jax(dim=self.head_dim, n=n)
        query = apply_rotary_pos_emb_jax(query, sin, cos)
        key = apply_rotary_pos_emb_jax(key, sin, cos)

        mask = -jnp.inf * ~jnp.einsum("i,j->ij", mask, mask)

        # NOTE: replace with jax.nn.dot_product_attention
        attention = jnp.einsum("hik,hjk->hij", query, key) / jnp.sqrt(self.head_dim)
        attention = jax.nn.softmax(attention + mask[None, :, :], axis=-1)

        out = jnp.einsum("hij,hjk->hik", attention, value)
        out = einops.rearrange(out, "head seq dim -> seq (head dim)")
        return jax.vmap(self.output)(out)


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
            TransformerLayer(dim=dim, num_heads=num_heads, ff_factor=4, key=key)
            for key in jr.split(key2, num_layers)
        ]

        # ESM-2 uses tied weights for the language modeling head
        self.lm_head = RobertaLMHead(
            dim=dim, out_dim=vocab_size, weight=self.embedding.weight, key=key3
        )

        self.layer_norm = nn.LayerNorm(dim)

    def __call__(
        self, seq: Float[Array, " seq"], mask: Float[Array, " seq"]
    ) -> tuple[Float[Array, "seq vocab"], Float[Array, "seq dim"]]:
        emb = jax.vmap(self.embedding)(seq)
        for layer in self.layers:
            emb = layer(emb, mask)
        emb = jax.vmap(self.layer_norm)(emb)
        return self.lm_head(emb), emb
