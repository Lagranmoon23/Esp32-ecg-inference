# v1.0 发布清单

生成日期：2026-10-31

## 本地来源

```text
G:\esp-projects\esp32_ecg_inference
```

复制了当前实时固件源文件、配置、依赖锁、分区表、W16A16模型、900心搏评估数据、资产导出工具和Windows可视化。没有复制 `build`、`managed_components`、`sdkconfig.old`、`.vscode` 和损坏的旧实时入口。

900心搏评估入口以 `firmware/evaluation/app_main_board_eval.cpp` 保存，不参与默认实时固件编译。

## 服务器来源

```text
/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/MIT-BIH/mitdb-1.0.0/FPGA-based-Tiny-Transformer-model-for-ECG-Arrythmia-Classification/software
```

服务器规范发布目录：

```text
software/releases/v1.0
```

## 身份校验

```text
ESP-DL模型 MD5:
0D75C5AA82AC614B40D5EDB98DF4D37C

board_eval.bin SHA256:
E8B28550FFB650892E7D2FEE0F6BC71993B58AFF4B784CC854BB6E8E48669594
```

整理时发现服务器原 `deployment_artifacts/primitive/stage1_n_s_vf_w16a16.espdl`
的 MD5 为 `BBDFCEB07E2EC2368CDD9BA6D2F17454`，与实际板端文件不一致。
v1.0 已统一采用完成真实板端评估的 `0D75...` 文件；原服务器实验产物未被删除或覆盖。

## 明确排除

- 所有Python虚拟环境和包缓存。
- 所有ESP-IDF构建中间文件。
- MIT-BIH、SVDB、INCART原始数据。
- 大型split缓存的物理副本。
- baseline、四分类、F detector、外部预训练和未被选中的流式候选权重。
- 图片、Notebook历史备份和训练日志。

这些内容仍保留在原服务器目录，没有删除。排除它们是为了让v1.0只表达最终可部署链路，而不是完整实验垃圾场。

## v1.0 后续计划

下一开发版本从SQI开始，然后加入MPU6050/ICM-42688同步加速度：先做运动检测和分类门控，再评估LMS/NLMS自适应消噪。任何预处理变化必须重新进行主机与板端验证。
