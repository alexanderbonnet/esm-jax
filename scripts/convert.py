"""Script to convert torch model weights to equinox."""

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import torch
from transformers import AutoModelForMaskedLM

import esmjax

parent = Path(__file__).parent


def build_conversion_map(name: str) -> dict:
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
    for k in range(esmjax.MODEL_HYPERPARAMS[name].num_layers):
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
    updated_path_vals = []
    count = 0
    for names, array in path_vals:
        key = ".".join(str(x).strip(".") for x in names)
        if key in conversion_map:
            weights = state_dict[conversion_map[key]]
            assert array.shape == weights.shape, f"{array.shape} != {weights.shape} for {key=}"
            updated_path_vals.append((names, jnp.asarray(weights)))
            count += 1
        else:
            updated_path_vals.append((names, array))

    updated_leaves = [v for _, v in updated_path_vals]
    updated_module = jax.tree.unflatten(treedef, updated_leaves)

    if not count == len(conversion_map):
        raise ValueError(
            f"Did not find all keys in conversion map {count=}, {len(conversion_map)=}"
        )
    return updated_module


def main(name: str, eqx_path: str | Path | None = None) -> None:
    if eqx_path is None:
        eqx_path = parent.parent / ("data/weights/" + name + ".eqx")

    eqx_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = AutoModelForMaskedLM.from_pretrained("facebook/" + name).state_dict()
    model = esmjax.ESM2(**dataclasses.asdict(esmjax.MODEL_HYPERPARAMS[name]), key=jr.PRNGKey(0))
    conversion_map = build_conversion_map(name)
    updated_model = update_eqx_with_state_dict(model, state_dict, conversion_map)
    eqx.tree_serialise_leaves(eqx_path, updated_model)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
