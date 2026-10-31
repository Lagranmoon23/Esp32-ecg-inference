# v1.0 服务器端流水线

本目录是服务器 `software/releases/v1.0` 的本地镜像，保存最终模型审计、流式验证、ONNX 构建、ESP-DL 量化及固件资产导出所需代码和关键产物。

## 服务器规范位置

```text
/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/MIT-BIH/mitdb-1.0.0/FPGA-based-Tiny-Transformer-model-for-ECG-Arrythmia-Classification/software/releases/v1.0
```

连接：

```powershell
ssh ecglab506@100.92.120.56
```

## Python 环境

模型审计、TensorFlow 与流式研究环境：

```bash
source /media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/MIT-BIH/mitdb-1.0.0/mcu_env/bin/activate
```

关键版本：Python 3.10.22、TensorFlow 2.21.0、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2、WFDB 4.3.1、ONNX 1.23.2、tf2onnx 1.17.0。

ESP-PPQ 量化环境：

```bash
source /media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/MIT-BIH/mitdb-1.0.0/ppq_env/bin/activate
```

关键版本：Python 3.10.22、esp-ppq 1.3.12、PyTorch 2.14.1+cpu、ONNX 1.19.1、ONNX Runtime 1.23.2。

## 数据位置

数据没有复制进版本包：

```text
MIT-BIH:
/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0

患者独立 split:
../../patient_independent_hierarchical_splits.npz

流式训练 split:
../../patient_independent_hierarchical_streaming_train_val.npz
```

服务器发布目录的 `data/` 使用符号链接指向上述 split，不重复占用磁盘。

## 正式模型

- `models/tiny_transformer_stage1_n_s_vf.weights.h5`：最终选择的原始 Stage-1 权重。
- `models/tiny_transformer_stage1_n_s_vf_streaming_selected.weights.h5`：保护逻辑选择结果，与原始权重相同。
- `models/stage1_n_s_vf.keras`：完整 Keras 模型。
- `models/stage1_n_s_vf_deploy.onnx`：基础算子部署图，无融合 Attention。
- `models/stage1_n_s_vf_w16a16.espdl`：v1.0 板端模型。
- `models/stage1_golden_vectors_w16a16.npz`：量化黄金向量。
- `models/stage1_board_eval_w16a16_900.bin`：平衡900心搏板端评估包。

v1.0 的 `.espdl` 以本地实际烧录并完成900心搏评估的文件为准，MD5 为
`0D75C5AA82AC614B40D5EDB98DF4D37C`。服务器原 `deployment_artifacts/primitive`
中较晚生成的同名文件 MD5 不同，因此没有把它作为 v1.0 正式模型，也没有改动原实验目录。

## 核心脚本

| 脚本 | 用途 |
|---|---|
| `stage1_deployment_audit.py` | 浮点模型与患者独立测试审计 |
| `stage1_streaming_pipeline_study.py` | 因果滤波、滚动标准化、R峰和端到端流式评估 |
| `stage1_streaming_train.py` | 保守流式微调与模型保护选择 |
| `stage1_build_deploy_onnx.py` | 构建 ESP-DL 兼容的基础算子 ONNX |
| `stage1_debug_stages.py` | 分阶段数值对齐 |
| `stage1_espdl_quantize.py` | ESP-PPQ量化、评估和运行时算子检查 |
| `export_firmware_assets.py` | 导出 SOS、RR统计和固件黄金数据 |

其余脚本是因果滤波与窗口化零相位消融，保留用于解释 v1.0 技术选择。

## 复现原则

- DS1 只用于训练和量化校准，DS2 只用于最终测试。
- R峰阈值只能在 train/validation 上调节，冻结后再运行 test。
- 实时固件必须使用因果滤波，不得以 `sosfiltfilt` 冒充实时实现。
- PPQ模拟结果不能冒充真实ESP32板端结果。
- v1.0 使用 W16A16 基础算子模型，不使用不受 ESP-DL 3.3.14 支持的融合 Attention。

## 已冻结结果

- 平衡900心搏主机参考：Accuracy 0.6889，Macro-F1 0.6779。
- 平衡900心搏真实ESP32：Accuracy 0.6989，Macro-F1 0.6963。
- 板端平均推理：90 ms/心搏，失败 0/900。
- 主机与板端预测一致率：86.11%。

这些数字的机器可读副本位于 `artifacts/board_evaluation_v1.0.json`。

除非开始 v1.1 实验，否则不要覆盖 v1.0 文件。新的实验应写入新的输出目录。
