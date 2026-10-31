"""Compare each post-attention stage: Keras truth (one model per tensor) vs the
numpy chain that the ONNX graph implements. The first mismatching row is the
assumption that is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

import stage1_build_deploy_onnx as B


GOLDEN = "deployment_artifacts/stage1_golden_vectors.npz"
WEIGHTS = "tiny_transformer_stage1_n_s_vf.weights.h5"
N = 8


def ln(x, g, b, eps=1e-3):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


_erf = np.vectorize(math.erf)


def gelu_exact(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def show(name, ref, got, tol=1e-4):
    a = np.asarray(ref, np.float64).ravel()
    b = np.asarray(got, np.float64).ravel()
    if a.size != b.size:
        print(f"  [BAD] {name:<10} SHAPE {a.size} vs {b.size}  ref{a.shape} got{b.shape}")
        return
    d = np.abs(a - b)
    print(f"  [{'OK ' if d.max() < tol else 'BAD'}] {name:<10} max|d|={d.max():.3e} mean|d|={d.mean():.3e}")


def main() -> None:
    model = B.build_reference_model()
    model.load_weights(WEIGHTS)
    w = B.extract_weights(model)

    with np.load(GOLDEN) as g:
        gx1 = g["x1"].astype(np.float32)[:N]
        gx2 = g["x2"].astype(np.float32)[:N]
    feed = {"x1": gx1, "x2": gx2}

    def get(layer_name):
        m = tf.keras.Model(model.inputs, model.get_layer(layer_name).output)
        return m.predict(feed, batch_size=N, verbose=0).astype(np.float64)

    names = [
        "add", "layer_normalization_1", "dense", "dense_1", "add_1", "encoder_norm",
        "global_average_pooling1d", "global_max_pooling1d", "concatenate",
        "rr_norm", "rr_embedding", "concatenate_1", "fusion",
    ]
    k = {n: get(n) for n in names}
    k["logits"] = model.predict(feed, batch_size=N, verbose=0).astype(np.float64)
    for n in names:
        print(f"  keras {n:<28} {k[n].shape}")

    def W(layer):
        return layer.kernel.numpy().astype(np.float64), layer.bias.numpy().astype(np.float64)

    W1, b1 = W(w["d_mlp1"])
    W2, b2 = W(w["d_mlp2"])
    Wr, br = W(w["d_rr"])
    Wf, bf = W(w["d_fuse"])
    Wh, bh = W(w["d_head"])
    g_mlp, be_mlp = [np.asarray(t.numpy(), np.float64) for t in w["ln_mlp"].weights]
    g_enc, be_enc = [np.asarray(t.numpy(), np.float64) for t in w["ln_enc"].weights]
    g_rr, be_rr = [np.asarray(t.numpy(), np.float64) for t in w["ln_rr"].weights]

    r1 = k["add"]
    n2 = ln(r1, g_mlp, be_mlp)
    m1pre = n2 @ W1 + b1
    m1post = gelu_exact(m1pre)
    m2 = gelu_exact(m1post @ W2 + b2)
    r2 = m2 + r1
    n3 = ln(r2, g_enc, be_enc)
    avg = n3.mean(1, keepdims=True)
    mx = n3.max(1, keepdims=True)
    morph = np.concatenate([avg, mx], 1).reshape(N, -1)
    rrn = ln(gx2.astype(np.float64), g_rr, be_rr)
    rrpre = rrn @ Wr + br
    rrpost = gelu_exact(rrpre)
    fin = np.concatenate([morph, rrpost], 1)
    fpre = fin @ Wf + bf
    fpost = gelu_exact(fpre)
    logits = fpost @ Wh + bh

    print("\nchain vs keras truth")
    show("n2", k["layer_normalization_1"], n2)
    show("mlp1_post", k["dense"], m1post, tol=1e-3)
    show("mlp2", k["dense_1"], m2)
    show("r2", k["add_1"], r2)
    show("n3", k["encoder_norm"], n3)
    show("avg", k["global_average_pooling1d"], avg)
    show("mx", k["global_max_pooling1d"], mx)
    show("morph", k["concatenate"], morph)
    show("rr_norm", k["rr_norm"], rrn)
    show("rr_post", k["rr_embedding"], rrpost, tol=1e-3)
    show("fuse_in", k["concatenate_1"], fin)
    show("fuse_post", k["fusion"], fpost, tol=1e-3)
    show("logits", k["logits"], logits)

    print("\n  keras dense_1[0,0,:6] :", np.round(k["dense_1"][0, 0, :6], 5))
    print("  chain m2    [0,0,:6]  :", np.round(m2[0, 0, :6], 5))
    print("  keras dense_1 has {:.4f}..{:.4f}".format(k["dense_1"].min(), k["dense_1"].max()))
    print("  chain m2      has {:.4f}..{:.4f}".format(m2.min(), m2.max()))
    print("  dense(k) out  has {:.4f}..{:.4f}".format(k["dense"].min(), k["dense"].max()))


if __name__ == "__main__":
    main()
