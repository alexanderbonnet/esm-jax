from typing import Final

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Int, jaxtyped

# fmt: off
VOCAB: Final[list[str]] = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R",
    "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X",
    "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"
    ]
# fmt: on
TOKENS: Final[dict[str, int]] = dict(zip(VOCAB, range(len(VOCAB))))


@jaxtyped(typechecker=beartype)
def tokenize(sequence: str) -> Int[Array, " n"]:
    tokens = (
        [TOKENS["<cls>"]]
        + [TOKENS.get(char, TOKENS["<unk>"]) for char in sequence]
        + [TOKENS["<eos>"]]
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
