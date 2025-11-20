# esm-jax

Jax implementation of the esm2 protein language model, inference only. The outputs
were verified against the available models on hugging face.

Dependencies are managed with `uv`.

## Example

The model name can be any of the following.

| name                |
| ------------------- |
| esm2_t48_15B_UR50D  |
| esm2_t36_3B_UR50D   |
| esm2_t33_650M_UR50D |
| esm2_t30_150M_UR50D |
| esm2_t12_35M_UR50D  |
| esm2_t6_8M_UR50D    |

On the first run with the model, weights will be downloaded from hugging face using [`transformers`](https://github.com/huggingface/transformers). The checkpoints (converted to Jax/Equinox) will be cached in the `data/weights` directory.

```python
import jax.random as jr

from esmjax import esm2

model = esm2.ESM2.from_pretrained("esm2_t30_150M_UR50D", key=jr.PRNGKey(43))

sequence = "MGSSHHHHHHSSGLVPAGSHMEEKQILCVGLVVLDIINVVDKYPEEDTDRRCLSQRWQRGGNASNSCTVLSLLGARCAFMGSLAPGHVADFVLDDLRQHSVDLRYVVLQTEGSIPTSTVIINEASGSRTILHAYRNLPDVSAKDFEKVDLTRFKWIHIEGRNASEQVKMLQRIEEHNAKQPLPQKVRVSVEIEKPREELFQLFSYGEVVFVSKDVAKHLGFQSAVEALRGLYSRVKKGATLVCAWAEEGADALGPDGQLLHSDAFPPPRVVDTLGAGDTFNASVIFSLSKGNSMQEALRFGCQVAGKKCGLQGFDGIV"

tokens = esm2.tokenize(sequence)
tokens, mask = esm2.pad_and_mask(tokens, pad_length=16) # in case you need to pad or mask

logits, embedding = model(tokens=tokens, mask=mask)

...
```
