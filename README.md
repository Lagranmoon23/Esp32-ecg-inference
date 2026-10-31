# Edge ECG Tiny Transformer v1.0

这是截至 2026-10-31 已验证代码与模型的只读发布快照，包含 ESP32-S3 实时固件、Windows 可视化、服务器端模型审计/转换/量化/流式验证代码，以及关键部署产物。

本版本不会替代原开发目录；原文件没有被移动或删除。

## 发布结论

- 硬件：ESP32-S3 N16R8 + AD8232
- 任务：单导联 ECG 心搏 N / S / (V+F) 三分类
- 板端模型：ESP-DL W16A16
- 实时链路：360 Hz ADC、0.5–40 Hz 因果带通、50 Hz 陷波、10 秒滚动标准化、R 峰检测、RR 特征、板端推理
- 900 心搏真实板端结果：Accuracy 0.6989，Macro-F1 0.6963，平均 90 ms/心搏，推理失败 0/900
- Windows：实时 ECG、心率、类别与概率可视化

该系统是研究和工程原型，不是医疗器械，不能用于医疗诊断。

## 目录

```text
esp32_ecg_inference_v1.0/
├─ VERSION
├─ README.md
├─ firmware/                 # 可直接用 ESP-IDF 构建的固件快照
│  ├─ main/                  # 实时 DSP、R 峰、特征和 ESP-DL 推理
│  ├─ evaluation/            # 900 心搏板端评估入口备份
│  ├─ tools/                 # 固件资产导出与量化脚本副本
│  └─ plot_live_ecg.py       # Windows 实时可视化
├─ server_pipeline/          # 服务器端代码和最终部署产物镜像
│  ├─ scripts/
│  ├─ models/
│  ├─ artifacts/
│  └─ README.md
└─ docs/
   └─ RELEASE_MANIFEST.md
```

## 快速使用固件

在 ESP-IDF PowerShell 中：

```powershell
cd G:\esp-projects\esp32_ecg_inference_v1.0\firmware
idf.py set-target esp32s3
idf.py build
idf.py -p COM9 flash monitor
```

目标已是 ESP32-S3 时不需要重复运行 `set-target`。启动可视化前用 `Ctrl + ]` 关闭 monitor：

```powershell
python -m pip install pyserial matplotlib
python .\plot_live_ecg.py COM9
```

接线、串口协议、安全说明和板端指标见 [firmware/README.md](firmware/README.md)。

## 最终模型身份

```text
firmware/main/models/ecg.espdl
MD5: 0D75C5AA82AC614B40D5EDB98DF4D37C
```

服务器规范副本：

```text
server_pipeline/models/stage1_n_s_vf_w16a16.espdl
```

两者应保持逐字节一致。

## 版本边界

v1.0 收录当前实际使用或用于复现最终部署结果的代码。以下内容刻意不复制：

- ESP-IDF `build/` 和 `managed_components/`；可由依赖清单重新生成。
- Python 虚拟环境、`__pycache__` 和日志。
- 失败的外部预训练、四分类、F-detector 与候选微调模型。
- MIT-BIH、SVDB、INCART 原始数据和大型 split 缓存；服务器保留原路径。
- 早期损坏或已禁用的实时固件草稿。

## 下一版本重点

1. 加入 ECG 信号质量评估 SQI，质量差时拒绝分类。
2. 加入 MPU6050/ICM-42688 加速度计进行运动门控。
3. 比较加速度辅助 LMS/NLMS 消噪前后的 R 峰与 Macro-F1。
4. 增加 3–5 心搏时间平滑和低置信度拒识。
5. 将单次 ADC 读取升级为 continuous/DMA 并测量采样抖动。

详细清单见 [docs/RELEASE_MANIFEST.md](docs/RELEASE_MANIFEST.md)。
