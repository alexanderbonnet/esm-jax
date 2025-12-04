import pathlib
from typing import Final

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Int, jaxtyped


def _load_vocab() -> dict[str, int]:
    vocab_path = pathlib.Path(__file__).parent / "_vocab.txt"
    vocab = vocab_path.read_text().splitlines()
    return {token: idx for idx, token in enumerate(vocab)}


VOCAB: Final[dict[str, int]] = _load_vocab()


@jaxtyped(typechecker=beartype)
def tokenize(sequence: str) -> Int[Array, " n"]:
    tokens = (
        [VOCAB["<cls>"]] + [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence] + [VOCAB["<eos>"]]
    )
    return jnp.array(tokens, dtype=jnp.int32)


@jaxtyped(typechecker=beartype)
def pad_and_mask(
    tokens: Int[Array, " n"], pad_length: int = 0
) -> tuple[Int[Array, " m"], Bool[Array, " m"]]:
    if pad_length is None:
        pad_length = 0

    mask = jnp.array([True] * tokens.shape[0] + [False] * pad_length, dtype=jnp.bool_)
    tokens = jnp.concat((tokens, jnp.zeros(pad_length, dtype=tokens.dtype)), axis=0)

    return tokens, mask
