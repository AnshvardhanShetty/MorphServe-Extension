"""LTS, LRS, MDS, and combined LIS computation."""

import torch
import torch.nn.functional as F



def compute_lts_scores(model, inputs):
    """Run forward pass, capture per-layer input/output similarity.
    Higher LTS = layer transforms less = safer to swap."""
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def save_io(mod, inp, out, idx=i):
            layer_inputs[idx] = inp[0].detach()
            layer_outputs[idx] = out[0].detach()
        hooks.append(layer.register_forward_hook(save_io))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    num_layers = len(model.model.layers)
    lts_scores = []
    for i in range(num_layers):
        inp = layer_inputs[i].squeeze(0)
        out = layer_outputs[i]
        if out.dim() == 3:
            out = out.squeeze(0)
        lts = F.cosine_similarity(inp.float(), out.float(), dim=-1).mean().item()
        lts_scores.append(lts)

    return lts_scores, layer_outputs


def compute_lrs_scores(model_fp16, model_int4, inputs, fp16_layer_outputs):
    """Cosine sim between FP16 and INT4 layer outputs.
    Higher = quantization changes the layer less."""
    int4_layer_outputs = {}
    hooks = []

    # AWQ model has an extra .model wrapper so it's .model.model.layers
    for i, layer in enumerate(model_int4.model.model.layers):
        def save_out(mod, inp, out, idx=i):
            int4_layer_outputs[idx] = out[0].detach()
        hooks.append(layer.register_forward_hook(save_out))

    with torch.no_grad():
        model_int4(inputs["input_ids"])

    for h in hooks:
        h.remove()

    num_layers = len(model_fp16.model.layers)
    lrs_scores = []
    for i in range(num_layers):
        fp16_out = fp16_layer_outputs[i]
        int4_out = int4_layer_outputs[i]
        if fp16_out.dim() == 3:
            fp16_out = fp16_out.squeeze(0)
        if int4_out.dim() == 3:
            int4_out = int4_out.squeeze(0)
        lrs = F.cosine_similarity(fp16_out.float(), int4_out.float(), dim=-1).mean().item()
        lrs_scores.append(lrs)

    return lrs_scores


def compute_mds_scores(model_fp16, model_int4, inputs):
    """Swap each layer individually to INT4, measure final output change.
    Most expensive metric but captures cascading effects."""
    num_layers = len(model_fp16.model.layers)

    with torch.no_grad():
        baseline_output = model_fp16(**inputs, output_hidden_states=True)
        baseline_final = baseline_output.hidden_states[-1].squeeze(0).float()

    mds_scores = []
    for target_layer in range(num_layers):
        fp16_layer = model_fp16.model.layers[target_layer]
        int4_layer = model_int4.model.model.layers[target_layer]

        model_fp16.model.layers[target_layer] = int4_layer
        with torch.no_grad():
            swapped_output = model_fp16(**inputs, output_hidden_states=True)
            swapped_final = swapped_output.hidden_states[-1].squeeze(0).float()
        model_fp16.model.layers[target_layer] = fp16_layer

        mds = F.cosine_similarity(baseline_final, swapped_final, dim=-1).mean().item()
        mds_scores.append(mds)

    return mds_scores


def compute_lis_scores(lts, lrs, mds, weights=(0.25, 0.25, 0.5)):
    """Weighted combination. MDS gets 0.5 since it measures actual end-to-end impact."""
    w_lts, w_lrs, w_mds = weights
    return [w_lts * l + w_lrs * r + w_mds * m for l, r, m in zip(lts, lrs, mds)]
