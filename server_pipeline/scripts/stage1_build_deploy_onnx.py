"""Build an ESP-DL-friendly ONNX graph for the N/S/VF Stage-1 model.

Why this exists
---------------
tf2onnx emits a faithful but deployment-hostile graph:

  * exact GELU becomes Erf            -> ESP-DL has no Erf operator
  * LayerNorm is decomposed into
    ReduceMean/Sub/Mul/Add/Sqrt/Reciprocal
                                     -> ESP-DL has no Reciprocal, and the
                                        decomposed form is 8 passes instead of 1

ESP-DL does provide a fused LayerNormalization operator, Tanh, MatMul, Gemm,
Conv, Softmax, ReduceMean, ReduceMax, Concat, Reshape and Transpose. So this
script re-expresses the very same weights with that operator set:

  * GELU  -> tanh approximation 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
  * LayerNorm -> single LayerNormalization node (scale + bias + eps=1e-3)
  * Dense -> MatMul + Add (weight is the static input1, so w8a16 can apply)
  * pooling -> ReduceMean / ReduceMax over the token axis

Nothing is retrained; every parameter is copied from the audited checkpoint.
The script then proves the result on the golden vectors and, with --full-suite,
on all 49,661 DS2 beats.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import TensorProto, helper
from tensorflow.keras import layers


NUM_HEADS = 8
KEY_DIM = 2
HEAD_DIM = KEY_DIM
EMBEDDING = 16
MLP_DIM = 128
KERNEL = 3
SEQ_LEN = 66
RR_FEATURE_COUNT = 6
LN_EPS = 1e-3

GELU_C1 = 0.7978845608028654  # sqrt(2/pi)
GELU_C2 = 0.03567740813636141  # 0.044715 * sqrt(2/pi)
MHA_SCALE = 1.0 / np.sqrt(KEY_DIM)

# Filled during build_onnx so diagnostic scripts can probe intermediate tensors.
DEBUG_TENSORS: dict[str, str] = {}

# Operator set implemented and tested by ESP-DL (operator_support_state.md).
ESP_DL_OPS = {
    "Add", "AveragePool", "Clip", "Concat", "Conv", "ConvTranspose", "DepthToSpace",
    "Div", "Elu", "Equal", "Exp", "Flatten", "Gather", "Gemm", "GlobalAveragePool",
    "Greater", "GreaterOrEqual", "GRU", "HardSigmoid", "HardSwish", "LayerNormalization",
    "LeakyRelu", "Less", "LessOrEqual", "Log", "LogSoftmax", "LpNormalization", "LSTM",
    "MatMul", "MaxPool", "Mod", "Mul", "Neg", "Pad", "Pow", "PRelu", "ReduceL1",
    "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
    "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape",
    "Resize", "ReverseSequence", "RMSNormalization", "ScatterND", "Sigmoid", "Slice",
    "Softmax", "SpaceToDepth", "Split", "Sqrt", "Squeeze", "Sub", "Swish", "Tanh",
    "Transpose", "Unsqueeze",
}


@tf.keras.utils.register_keras_serializable(package="ecg")
class AddPositionEmbs(layers.Layer):
    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init

    def build(self, inputs_shape):
        init = self.posemb_init
        if init is None or isinstance(init, dict):
            init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, inputs_shape[1], inputs_shape[2]),
            initializer=init,
            trainable=True,
        )

    def call(self, inputs, inputs_positions=None):
        return inputs + tf.cast(self.pos_embedding, inputs.dtype)


def build_reference_model() -> tf.keras.Model:
    """Byte-for-byte the architecture the checkpoint was trained with."""
    x1 = layers.Input(shape=(198, 1, 1), name="x1")
    x2 = layers.Input(shape=(RR_FEATURE_COUNT,), name="x2")
    x = layers.Conv2D(EMBEDDING, (KERNEL, 1), strides=(KERNEL, 1), padding="valid")(x1)
    x = layers.Reshape((SEQ_LEN, EMBEDDING))(x)
    x = AddPositionEmbs(name="posembed_input")(x)
    x = layers.Dropout(0.2)(x)

    attention_input = layers.LayerNormalization(dtype=x.dtype)(x)
    attention = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1
    )(attention_input, attention_input)
    x = layers.Add()([attention, x])

    mlp_input = layers.LayerNormalization(dtype=x.dtype)(x)
    y = layers.Dense(MLP_DIM, activation=tf.nn.gelu)(mlp_input)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(EMBEDDING, activation=tf.nn.gelu)(y)
    y = layers.Dropout(0.1)(y)
    x = layers.Add()([y, x])

    encoded = layers.LayerNormalization(name="encoder_norm")(x)
    avg = layers.GlobalAveragePooling1D()(encoded)
    mx = layers.GlobalMaxPooling1D()(encoded)
    morphology = layers.Concatenate()([avg, mx])

    rr = layers.LayerNormalization(name="rr_norm")(x2)
    rr = layers.Dense(16, activation=tf.nn.gelu, name="rr_embedding")(rr)
    rr = layers.Dropout(0.15)(rr)

    fused = layers.Concatenate()([morphology, rr])
    fused = layers.Dense(32, activation=tf.nn.gelu, name="fusion")(fused)
    fused = layers.Dropout(0.25)(fused)
    logits = layers.Dense(
        3, name="stage1_n_s_vf_head", kernel_initializer=tf.keras.initializers.Zeros()
    )(fused)
    return tf.keras.Model(inputs=[x1, x2], outputs=logits, name="stage1_n_s_vf")


class Graph:
    """Tiny helper so the graph definition reads like the model."""

    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: list = []
        self.counter = 0

    def const(self, arr: np.ndarray, name: str | None = None) -> str:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        name = name or f"w_{len(self.inits)}"
        self.inits.append(
            helper.make_tensor(name, TensorProto.FLOAT, list(arr.shape), arr.ravel().tolist())
        )
        return name

    def i64(self, arr, name: str | None = None) -> str:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
        name = name or f"i_{len(self.inits)}"
        self.inits.append(
            helper.make_tensor(name, TensorProto.INT64, list(arr.shape), arr.ravel().tolist())
        )
        return name

    def op(self, op_type: str, inputs, prefix: str, **attrs) -> str:
        self.counter += 1
        out = f"{prefix}_{self.counter}"
        self.nodes.append(helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    def matmul_bias(self, x: str, kernel: np.ndarray, bias: np.ndarray, prefix: str) -> str:
        """MatMul(x, W) + b. Keras kernels are already (in, out), i.e. ONNX K x N."""
        mm = self.op("MatMul", [x, self.const(kernel, f"{prefix}_W")], f"{prefix}_mm")
        return self.op("Add", [mm, self.const(bias, f"{prefix}_b")], f"{prefix}_bias")

    def layer_norm(self, x: str, gamma: np.ndarray, beta: np.ndarray, prefix: str) -> str:
        return self.op(
            "LayerNormalization",
            [x, self.const(gamma, f"{prefix}_g"), self.const(beta, f"{prefix}_be")],
            prefix,
            axis=-1,
            epsilon=LN_EPS,
        )

    def gelu(self, x: str, prefix: str) -> str:
        """0.5 * x * (1 + tanh(c1 * x + c2 * x^3))."""
        x2 = self.op("Mul", [x, x], f"{prefix}_sq")
        x3 = self.op("Mul", [x2, x], f"{prefix}_cu")
        t1 = self.op("Mul", [x3, self.const(np.array(GELU_C2, np.float32), f"{prefix}_c2")], f"{prefix}_t1")
        t2 = self.op("Mul", [x, self.const(np.array(GELU_C1, np.float32), f"{prefix}_c1")], f"{prefix}_t2")
        u = self.op("Add", [t1, t2], f"{prefix}_u")
        th = self.op("Tanh", [u], f"{prefix}_tanh")
        one_plus = self.op("Add", [th, self.const(np.array(1.0, np.float32), f"{prefix}_one")], f"{prefix}_1p")
        half_x = self.op("Mul", [x, self.const(np.array(0.5, np.float32), f"{prefix}_half")], f"{prefix}_hx")
        return self.op("Mul", [one_plus, half_x], f"{prefix}_out")


def extract_weights(model: tf.keras.Model) -> dict:
    conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][0]
    pos = [l for l in model.layers if isinstance(l, AddPositionEmbs)][0]
    mha = [l for l in model.layers if isinstance(l, layers.MultiHeadAttention)][0]
    lns = [l for l in model.layers if isinstance(l, layers.LayerNormalization)]
    denses = [l for l in model.layers if isinstance(l, layers.Dense)]
    if len(lns) != 4 or len(denses) != 5:
        raise RuntimeError(f"unexpected layer counts: {len(lns)} LN, {len(denses)} Dense")

    w = {
        "conv_kernel": np.asarray(conv.kernel.numpy(), np.float32),      # (3,1,1,16)
        "conv_bias": np.asarray(conv.bias.numpy(), np.float32),          # (16,)
        "pos": np.asarray(pos.pos_embedding.numpy(), np.float32),        # (1,66,16)
        "ln_attn": lns[0], "ln_mlp": lns[1], "ln_enc": lns[2], "ln_rr": lns[3],
        "mha": mha,
        "d_mlp1": denses[0], "d_mlp2": denses[1], "d_rr": denses[2],
        "d_fuse": denses[3], "d_head": denses[4],
    }
    for key in ("d_mlp1", "d_mlp2", "d_rr", "d_fuse", "d_head"):
        if "rr_embedding" in w[key].name and key != "d_rr":
            raise RuntimeError("Dense ordering assumption broke")
    return w


def ln_params(layer) -> tuple[np.ndarray, np.ndarray]:
    gamma, beta = layer.weights
    return np.asarray(gamma.numpy(), np.float32), np.asarray(beta.numpy(), np.float32)


def build_onnx(w: dict, opset: int = 18) -> onnx.ModelProto:
    g = Graph()

    x1_vi = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 198, 1, 1])
    x2_vi = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, RR_FEATURE_COUNT])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])

    # ---- patch embedding: 198 -> 16 channels x 66 tokens -------------------
    flat = g.op("Reshape", ["x1", g.i64([1, 1, 198], "shape_flat")], "flat")
    # (kh=3, kw=1, in=1, out=16) -> (out=16, in=1, k=3) for a 1-D convolution
    conv_kernel = np.transpose(w["conv_kernel"], (3, 2, 0, 1)).reshape(EMBEDDING, 1, KERNEL)
    conv = g.op(
        "Conv",
        [flat, g.const(conv_kernel, "conv_W"), g.const(w["conv_bias"], "conv_B")],
        "conv",
        strides=[KERNEL],
        pads=[0, 0],
        group=1,
    )
    tok = g.op("Transpose", [conv], "tokT", perm=[0, 2, 1])          # (1,66,16)
    tok = g.op("Add", [tok, g.const(w["pos"], "pos_emb")], "tok_pos")
    DEBUG_TENSORS["tok"] = tok

    # ---- transformer block -------------------------------------------------
    n1 = g.layer_norm(tok, *ln_params(w["ln_attn"]), "ln_attn")
    DEBUG_TENSORS["n1"] = n1

    qw, qb, kw_, kb, vw, vb, ow, ob = [np.asarray(t.numpy(), np.float32) for t in w["mha"].weights]
    # Keras 3 MultiHeadAttention uses EinsumDense, so kernels already carry the
    # head axis: q/k/v are (in, heads, head_dim) and o is (heads, head_dim, out).
    # Flattening (heads, head_dim) into one axis turns them into plain matrices,
    # because the einsum output layout is exactly b, t, h*head_dim+d.
    qw2, qb2 = qw.reshape(EMBEDDING, EMBEDDING), qb.reshape(EMBEDDING)
    kw2, kb2 = kw_.reshape(EMBEDDING, EMBEDDING), kb.reshape(EMBEDDING)
    vw2, vb2 = vw.reshape(EMBEDDING, EMBEDDING), vb.reshape(EMBEDDING)
    ow2 = ow.reshape(EMBEDDING, EMBEDDING)  # rows are h*head_dim+d, cols are out
    q = g.matmul_bias(n1, qw2, qb2, "q")
    k = g.matmul_bias(n1, kw2, kb2, "k")
    v = g.matmul_bias(n1, vw2, vb2, "v")

    def to_heads(t: str, prefix: str) -> str:
        r = g.op("Reshape", [t, g.i64([1, SEQ_LEN, NUM_HEADS, HEAD_DIM], f"{prefix}_sh")], f"{prefix}_rs")
        return g.op("Transpose", [r], f"{prefix}_hp", perm=[0, 2, 1, 3])  # (1,8,66,2)

    qh = to_heads(q, "qh")
    kh = to_heads(k, "kh")
    vh = to_heads(v, "vh")
    kt = g.op("Transpose", [kh], "kt", perm=[0, 1, 3, 2])               # (1,8,2,66)
    scores = g.op("MatMul", [qh, kt], "scores")
    scores = g.op("Mul", [scores, g.const(np.array(MHA_SCALE, np.float32), "mha_scale")], "scores_s")
    attn = g.op("Softmax", [scores], "attn", axis=-1)
    # ESP-PPQ recognizes MatMul -> scale -> Softmax -> MatMul as a fused
    # `Attention` operation. ESP-DL 3.3.14 can serialize that operation but
    # has no runtime module for it on ESP32-S3. Softmax is mathematically in
    # [0, 1], so this Clip is numerically neutral while deliberately keeping
    # the graph as supported primitive operators on the device.
    attn = g.op(
        "Clip",
        [
            attn,
            g.const(np.array(0.0, np.float32), "attn_clip_min"),
            g.const(np.array(1.0, np.float32), "attn_clip_max"),
        ],
        "attn_primitive_guard",
    )
    ctx = g.op("MatMul", [attn, vh], "ctx")                             # (1,8,66,2)
    ctx = g.op("Transpose", [ctx], "ctxT", perm=[0, 2, 1, 3])           # (1,66,8,2)
    ctx = g.op("Reshape", [ctx, g.i64([1, SEQ_LEN, EMBEDDING], "ctx_sh")], "ctx_rs")
    attn_out = g.matmul_bias(ctx, ow2, ob, "attn_o")
    DEBUG_TENSORS["attn_out"] = attn_out
    r1 = g.op("Add", [attn_out, tok], "res1")
    DEBUG_TENSORS["r1"] = r1

    n2 = g.layer_norm(r1, *ln_params(w["ln_mlp"]), "ln_mlp")
    DEBUG_TENSORS["n2"] = n2
    h = g.matmul_bias(n2, np.asarray(w["d_mlp1"].kernel.numpy(), np.float32),
                      np.asarray(w["d_mlp1"].bias.numpy(), np.float32), "mlp1")
    DEBUG_TENSORS["mlp1_pre"] = h
    h = g.gelu(h, "mlp1_gelu")
    DEBUG_TENSORS["mlp1_post"] = h
    h = g.matmul_bias(h, np.asarray(w["d_mlp2"].kernel.numpy(), np.float32),
                      np.asarray(w["d_mlp2"].bias.numpy(), np.float32), "mlp2")
    DEBUG_TENSORS["mlp2_pre"] = h
    # the second Dense carries a GELU too, exactly like the first one
    h = g.gelu(h, "mlp2_gelu")
    DEBUG_TENSORS["mlp2"] = h
    r2 = g.op("Add", [h, r1], "res2")
    DEBUG_TENSORS["r2"] = r2

    # ---- pooling over the token axis --------------------------------------
    n3 = g.layer_norm(r2, *ln_params(w["ln_enc"]), "ln_enc")
    DEBUG_TENSORS["n3"] = n3
    # opset 18 moved the reduction axes from an attribute to an input.
    pool_axes = g.i64([1], "pool_axes")
    avg = g.op("ReduceMean", [n3, pool_axes], "gap", keepdims=1)         # (1,1,16)
    mx = g.op("ReduceMax", [n3, pool_axes], "gmp", keepdims=1)           # (1,1,16)
    cat = g.op("Concat", [avg, mx], "pool_cat", axis=1)                  # (1,2,16)
    morph = g.op("Reshape", [cat, g.i64([1, 2 * EMBEDDING], "morph_sh")], "morph")  # (1,32)
    DEBUG_TENSORS["avg"] = avg
    DEBUG_TENSORS["mx"] = mx
    DEBUG_TENSORS["morph"] = morph

    # ---- RR branch ---------------------------------------------------------
    rrn = g.layer_norm("x2", *ln_params(w["ln_rr"]), "ln_rr")
    rr = g.matmul_bias(rrn, np.asarray(w["d_rr"].kernel.numpy(), np.float32),
                       np.asarray(w["d_rr"].bias.numpy(), np.float32), "rr")
    DEBUG_TENSORS["rr_pre"] = rr
    rr = g.gelu(rr, "rr_gelu")
    DEBUG_TENSORS["rr_post"] = rr

    # ---- fusion ------------------------------------------------------------
    fused = g.op("Concat", [morph, rr], "fuse_cat", axis=1)              # (1,48)
    fused = g.matmul_bias(fused, np.asarray(w["d_fuse"].kernel.numpy(), np.float32),
                          np.asarray(w["d_fuse"].bias.numpy(), np.float32), "fuse")
    DEBUG_TENSORS["fuse_pre"] = fused
    fused = g.gelu(fused, "fuse_gelu")
    DEBUG_TENSORS["fuse_post"] = fused
    logits = g.matmul_bias(fused, np.asarray(w["d_head"].kernel.numpy(), np.float32),
                           np.asarray(w["d_head"].bias.numpy(), np.float32), "head")
    # expose the final tensor under the graph output name
    for node in g.nodes:
        if logits in node.output:
            node.output[0] = "logits"
            break
    else:
        raise RuntimeError("could not rename the head output")

    graph = helper.make_graph(g.nodes, "stage1_n_s_vf_deploy", [x1_vi, x2_vi], [out], g.inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="ecg-stage1-deploy-builder",
    )
    # onnx>=1.23 defaults to a very new IR version that older runtimes and the
    # ESP-PPQ quantizer reject; opset 18 pairs with IR 10.
    model.ir_version = 10
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tiny_transformer_stage1_n_s_vf.weights.h5")
    ap.add_argument("--golden", default="deployment_artifacts/stage1_golden_vectors.npz")
    ap.add_argument("--splits", default="patient_independent_hierarchical_splits.npz")
    ap.add_argument("--outdir", default="deployment_artifacts")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--full-suite", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("[1/5] load checkpoint into the reference architecture")
    model = build_reference_model()
    model.load_weights(str(Path(args.weights).resolve()))
    print(f"      parameters: {model.count_params():,}")
    weights = extract_weights(model)

    print("=" * 78)
    print(f"[2/5] build ESP-DL-friendly ONNX (opset {args.opset})")
    proto = build_onnx(weights, args.opset)
    onnx.checker.check_model(proto)
    onnx_path = outdir / "stage1_n_s_vf_deploy.onnx"
    onnx.save(proto, str(onnx_path))
    print(f"      written: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KiB)")
    print("      onnx.checker: PASSED")
    print(f"      inputs : {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in proto.graph.input]}")
    print(f"      outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in proto.graph.output]}")

    print("=" * 78)
    print("[3/5] operator inventory vs ESP-DL support")
    ops = Counter(n.op_type for n in proto.graph.node)
    missing = sorted(set(ops) - ESP_DL_OPS)
    for name, count in ops.most_common():
        flag = "OK " if name in ESP_DL_OPS else "NO "
        print(f"      [{flag}] {name:<22} {count:>4}")
    print(f"      total nodes: {sum(ops.values())}   distinct ops: {len(ops)}")
    print(f"      unsupported  : {missing if missing else 'none'}")

    print("=" * 78)
    print("[4/5] numerical check against Keras on golden vectors")
    with np.load(Path(args.golden).resolve()) as g:
        gx1 = g["x1"].astype(np.float32)
        gx2 = g["x2"].astype(np.float32)
        glabels = g["labels"].astype(np.int64)

    keras_logits = model.predict({"x1": gx1, "x2": gx2}, batch_size=32, verbose=0)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    if names != ["x1", "x2"]:
        raise RuntimeError(f"unexpected onnx input names/order: {names}")

    deploy_logits = np.empty_like(keras_logits, dtype=np.float32)
    for i in range(len(gx1)):
        deploy_logits[i] = sess.run(["logits"], {"x1": gx1[i : i + 1], "x2": gx2[i : i + 1]})[0][0]

    diff = np.abs(deploy_logits - keras_logits)
    top1 = float(np.mean(np.argmax(deploy_logits, 1) == np.argmax(keras_logits, 1)))
    print(f"      max |logit diff|  : {diff.max():.3e}   (tanh-GELU vs exact GELU)")
    print(f"      mean |logit diff| : {diff.mean():.3e}")
    print(f"      top-1 agreement   : {top1 * 100:.2f}%")
    print(f"      golden accuracy   : keras={np.mean(np.argmax(keras_logits,1)==glabels):.4f}  deploy={np.mean(np.argmax(deploy_logits,1)==glabels):.4f}")

    full = None
    if args.full_suite:
        print("=" * 78)
        print("[5/5] full DS2 replay with the deploy graph")
        from sklearn.metrics import accuracy_score, f1_score

        with np.load(Path(args.splits).resolve()) as d:
            xt = d["X_test"].astype(np.float32).reshape(-1, 198, 1, 1)
            rt = d["RR_test"].astype(np.float32)
            yr = d["y_test"].astype(np.int64)
        yt = np.where(yr >= 2, 2, yr).astype(np.int64)
        preds = np.empty(len(yt), dtype=np.int64)
        for i in range(len(yt)):
            preds[i] = int(np.argmax(sess.run(["logits"], {"x1": xt[i:i+1], "x2": rt[i:i+1]})[0][0]))
            if (i + 1) % 10000 == 0:
                print(f"      {i+1}/{len(yt)}", flush=True)
        acc = accuracy_score(yt, preds)
        f1 = f1_score(yt, preds, labels=[0, 1, 2], average="macro")
        per_class = f1_score(yt, preds, labels=[0, 1, 2], average=None)
        full = {
            "samples": int(len(yt)),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "per_class_f1": per_class.tolist(),
        }
        print(f"      deploy accuracy : {acc:.4f}   (float baseline 0.9202)")
        print(f"      deploy macro-F1 : {f1:.4f}   (float baseline 0.6845)")
        print(f"      per-class F1    : N={per_class[0]:.4f} S={per_class[1]:.4f} VF={per_class[2]:.4f}")
    else:
        print("=" * 78)
        print("[5/5] full DS2 replay skipped (--full-suite)")

    report = {
        "onnx": str(onnx_path),
        "opset": args.opset,
        "size_kib": round(onnx_path.stat().st_size / 1024, 1),
        "operator_counts": dict(ops.most_common()),
        "total_nodes": int(sum(ops.values())),
        "unsupported_for_esp_dl": missing,
        "golden_vs_keras": {
            "samples": int(len(gx1)),
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
            "top1_agreement": top1,
        },
        "full_ds2": full,
    }
    (outdir / "stage1_deploy_onnx_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("=" * 78)
    print(f"report: {outdir / 'stage1_deploy_onnx_report.json'}")


if __name__ == "__main__":
    main()
