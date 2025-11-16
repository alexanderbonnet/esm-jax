# esm-jax

Jax implementation of the esm2 protein language model, inference only. The outputs
were verified against the available models on hugging face.

Dependencies are managed with `uv`.

## Example

### Converting weights

First, **weights need to be converted from torch to jax**, the converted weights will appear in the `data/weights` directory under `<model name>.eqx`.

Run `uv sync --group convert` to make sure to install the `transformers` package that we'll use to get the torch weights.

```bash
uv run python -m scripts.convert --name esm2_t30_150M_UR50D
```

The model name could be any of the following:

| name                |
| ------------------- |
| esm2_t48_15B_UR50D  |
| esm2_t36_3B_UR50D   |
| esm2_t33_650M_UR50D |
| esm2_t30_150M_UR50D |
| esm2_t12_35M_UR50D  |
| esm2_t6_8M_UR50D    |

### Inference

```python
import jax.random as jr

import esmjax

key = jr.PRNGKey(43)
model = esmjax.ESM2.from_pretrained("esm2_t30_150M_UR50D", key=key)

sequence = "MGSSHHHHHHSSGLVPAGSHMEEKQILCVGLVVLDIINVVDKYPEEDTDRRCLSQRWQRGGNASNSCTVLSLLGARCAFMGSLAPGHVADFVLDDLRQHSVDLRYVVLQTEGSIPTSTVIINEASGSRTILHAYRNLPDVSAKDFEKVDLTRFKWIHIEGRNASEQVKMLQRIEEHNAKQPLPQKVRVSVEIEKPREELFQLFSYGEVVFVSKDVAKHLGFQSAVEALRGLYSRVKKGATLVCAWAEEGADALGPDGQLLHSDAFPPPRVVDTLGAGDTFNASVIFSLSKGNSMQEALRFGCQVAGKKCGLQGFDGIV"

tokens = esmjax.tokenize(sequence)
tokens, mask = esmjax.pad_and_mask(tokens, pad_length=16) # in case you need to pad or mask

logits, embedding = model(tokens=tokens, mask=mask)

...
```
