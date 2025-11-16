import dataclasses


@dataclasses.dataclass
class ESMConfig:
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int


MODEL_HYPERPARAMS: dict[str, ESMConfig] = {
    "esm2_t48_15B_UR50D": ESMConfig(vocab_size=33, dim=2560, num_layers=48, num_heads=40),
    "esm2_t36_3B_UR50D": ESMConfig(vocab_size=33, dim=2560, num_layers=36, num_heads=36),
    "esm2_t33_650M_UR50D": ESMConfig(vocab_size=33, dim=1280, num_layers=33, num_heads=20),
    "esm2_t30_150M_UR50D": ESMConfig(vocab_size=33, dim=640, num_layers=30, num_heads=20),
    "esm2_t12_35M_UR50D": ESMConfig(vocab_size=33, dim=480, num_layers=12, num_heads=20),
    "esm2_t6_8M_UR50D": ESMConfig(vocab_size=33, dim=320, num_layers=6, num_heads=20),
}
