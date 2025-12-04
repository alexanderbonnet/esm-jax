# esm-jax

Jax implementation of the esm2 protein language model, inference only. The outputs
were verified against the available models on hugging face.

The model differs slightly from the hugging face version as I have removed
token dropout from inference.

Dependencies are managed with `uv`.

## Example

### ESM2

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

model = esm2.ESM2.from_pretrained("esm2_t30_150M_UR50D")

sequence = "MGSSHHHHHHSSGLVPAGSHMEEKQILCVGLVVLDIINVVDKYPEEDTDRRCLSQRWQRGGNASNSCTVLSLLGARCAFMGSLAPGHVADFVLDDLRQHSVDLRYVVLQTEGSIPTSTVIINEASGSRTILHAYRNLPDVSAKDFEKVDLTRFKWIHIEGRNASEQVKMLQRIEEHNAKQPLPQKVRVSVEIEKPREELFQLFSYGEVVFVSKDVAKHLGFQSAVEALRGLYSRVKKGATLVCAWAEEGADALGPDGQLLHSDAFPPPRVVDTLGAGDTFNASVIFSLSKGNSMQEALRFGCQVAGKKCGLQGFDGIV"

tokens = esm2.tokenize(sequence)
tokens, mask = esm2.pad_and_mask(tokens, pad_length=16) # in case you need to pad or mask

logits, embedding = model(tokens=tokens, mask=mask)

...
```

### ESMC (Cambrian)

The repo also proposes an implementation of ESM Cambrian. The available models are
`esmc-300m-2024-12` and `esmc-600m-2024-12`. The weights are downloaded from the
[hugging face hub](https://github.com/huggingface/huggingface_hub). The
checkpoints (converted to Jax/Equinox) will be cached in the `data/weights` directory.

The model is used identically to ESM2.

```python
import jax.random as jr

from esmjax import cambrian

model = cambrian.ESMC.from_pretrained("esm2_t30_150M_UR50D")

sequence = "MGSSHHHHHHSSGLVPAGSHMEEKQILCVGLVVLDIINVVDKYPEEDTDRRCLSQRWQRGGNASNSCTVLSLLGARCAFMGSLAPGHVADFVLDDLRQHSVDLRYVVLQTEGSIPTSTVIINEASGSRTILHAYRNLPDVSAKDFEKVDLTRFKWIHIEGRNASEQVKMLQRIEEHNAKQPLPQKVRVSVEIEKPREELFQLFSYGEVVFVSKDVAKHLGFQSAVEALRGLYSRVKKGATLVCAWAEEGADALGPDGQLLHSDAFPPPRVVDTLGAGDTFNASVIFSLSKGNSMQEALRFGCQVAGKKCGLQGFDGIV"

tokens = cambrian.tokenize(sequence)
tokens, mask = cambrian.pad_and_mask(tokens, pad_length=16) # in case you need to pad or mask

logits, embedding = model(tokens=tokens, mask=mask)

...
```

## Citations

```bash
@article{lin2023evolutionary,
  title = {Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author = {Zeming Lin and Halil Akin and Roshan Rao and Brian Hie and Zhongkai Zhu and Wenting Lu and Nikita Smetanin and Robert Verkuil and Ori Kabeli and Yaniv Shmueli and Allan dos Santos Costa and Maryam Fazel-Zarandi and Tom Sercu and Salvatore Candido and Alexander Rives},
  journal = {Science},
  volume = {379},
  number = {6637},
  pages = {1123-1130},
  year = {2023},
  doi = {10.1126/science.ade2574},
  URL = {https://www.science.org/doi/abs/10.1126/science.ade2574},
  note = {Earlier versions as preprint: bioRxiv 2022.07.20.500902}
}
```

```bash
@misc{esm2024cambrian,
  author = {{ESM Team}},
  title = {ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning},
  year = {2024},
  publisher = {EvolutionaryScale Website},
  url = {https://evolutionaryscale.ai/blog/esm-cambrian},
  urldate = {2024-12-04}
}
```
