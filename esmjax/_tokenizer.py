import pathlib
from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Int


def _load_vocab() -> dict[str, int]:
    vocab_path = pathlib.Path(__file__).parent / "_vocab.txt"
    vocab = vocab_path.read_text().splitlines()
    return {token: idx for idx, token in enumerate(vocab)}


VOCAB: Final[dict[str, int]] = _load_vocab()


def tokenize(sequence: str) -> Int[Array, " n"]:
    tokens = (
        [VOCAB["<cls>"]] + [VOCAB.get(char, VOCAB["<unk>"]) for char in sequence] + [VOCAB["<eos>"]]
    )
    return jnp.array(tokens, dtype=jnp.int32)
