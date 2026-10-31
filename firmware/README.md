# ESP32-S3 单导联 ECG 实时分析系统

本项目使用 AD8232 采集单导联 ECG，由 ESP32-S3 完成实时因果滤波、R 峰检测、心率计算和 Tiny Transformer 三分类，并通过串口在 Windows 上实时显示波形、分类概率与心率。

> 当前定位：科研与工程原型，用于心律异常筛查演示，不是医疗器械，不能用于疾病诊断。

## 1. 当前状态

- [x] AD8232 三电极单导联 ECG 采集
- [x] 360 Hz 实时采样
- [x] 0.5–40 Hz 因果 Butterworth 带通滤波
- [x] 50 Hz 陷波
- [x] 10 秒滚动中位数/MAD 标准化（每秒更新）
- [x] 自适应 R 峰检测与心率计算
- [x] 198 点心搏片段与 6 个 RR 特征构造
- [x] W16A16 ESP-DL Tiny Transformer 板端推理
- [x] N / S / V+F 三分类
- [x] Windows 实时波形、概率和心率可视化
- [x] 900 个平衡测试心搏的真实板端评估
- [ ] 信号质量评估（SQI）和运动伪差抑制
- [ ] 加速度传感器辅助运动识别

```text
AD8232 OUT
  -> ESP32-S3 ADC1 @ 360 Hz
  -> 0.5–40 Hz 因果带通 + 50 Hz 陷波
  -> 滚动稳健标准化 -> R 峰检测
  -> 198 点心搏 + 6 个 RR 特征
  -> ESP-DL Tiny Transformer
  -> N / S / V+F + 心率
  -> USB 串口 -> Windows 实时可视化
```

## 2. 分类含义

| 标签 | 含义 |
|---|---|
| N | 正常类心搏 |
| S | 室上性异常心搏 |
| V/F | 室性异常心搏与融合搏的合并类别 |

这里的 **V/F 不是心室颤动（Ventricular Fibrillation）**，而是数据集中 V 类与 F 类的合并标签。界面和项目说明中应写成 `V/F` 或 `V+F`，避免误解。

## 3. 硬件与接线

已验证平台：ESP32-S3 N16R8（16 MB Flash、8 MB PSRAM）、ESP-IDF 6.1、ESP-DL 3.3.14、AD8232。

| AD8232 | ESP32-S3 | 用途 |
|---|---|---|
| 3.3V | 3V3 | 模块供电，禁止接 5V |
| GND | GND | 共地 |
| OUT | GPIO1 / ADC1_CH0 | ECG 模拟信号 |
| LO+ | GPIO2 | 正导联脱落检测 |
| LO- | GPIO3 | 负导联脱落检测 |

人体电极使用 RA、LA、RL 三根线。线色可能不同，应以模块和导联标签为准。建议将 RA、LA 放在左右锁骨下方，RL 放在右下腹。手腕活动较多，更容易产生运动伪差。RL 是右腿驱动参考电极，有助于降低共模与市电干扰，不建议省略。

### 人体连接安全

- 人体佩戴电极时，优先使用笔记本电池供电或合规 USB 隔离器。
- 不要同时连接接地示波器、台式电源等非隔离设备。
- 本项目不具备医疗电气安全认证。
- 参考：[Analog Devices AD8232 数据手册](https://www.analog.com/media/en/technical-documentation/data-sheets/ad8232.pdf)。

## 4. 编译、烧录与监视

在 ESP-IDF PowerShell 中执行：

```powershell
cd G:\esp-projects\esp32_ecg_inference
idf.py set-target esp32s3
idf.py build
idf.py -p COM9 flash monitor
```

`set-target` 只需在目标芯片错误或首次配置时执行。普通修改后直接 `idf.py build`，不要反复 `fullclean`。使用 `Ctrl + ]` 退出串口监视器。

COM9 同一时间只能被一个程序占用。启动 Python 可视化前必须退出 ESP-IDF Monitor。

## 5. Windows 实时可视化

首次安装依赖：

```powershell
python -m pip install pyserial matplotlib
```

运行：

```powershell
cd G:\esp-projects\esp32_ecg_inference
python .\plot_live_ecg.py COM9
```

上图显示滤波 ECG、R 峰分类、电极状态、ADC 与心率；下图显示 N、S、V/F 概率。连接电极后约需 10 秒建立滚动统计。分类还需等待后一个 R 峰得到 post-RR 特征，因此存在约一个心搏的算法延迟。

## 6. 串口协议

```text
ECG,sample_index,adc_raw,filtered,normalized,lead_off
PRED,r_sample,class,bpm,logit_N,logit_S,logit_VF,pre_rr,post_rr,local_rr
```

内部采样率为 360 Hz。为降低串口和绘图负担，每 4 个采样点输出一行波形（约 90 行/秒），滤波、R 峰检测和模型输入仍使用完整 360 Hz 数据。

## 7. 模型与真实板端结果

- 任务：N / S / (V+F) 三分类
- 部署格式：ESP-DL `.espdl`
- 量化：W16A16
- 文件：`main/models/ecg.espdl`
- 已验证 MD5：`0D75C5AA82AC614B40D5EDB98DF4D37C`

固定平衡测试集包含 900 个心搏，每类 300 个。真实 ESP32-S3 结果：

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.6989 |
| Macro-F1 | 0.6963 |
| N F1 | 0.7252 |
| S F1 | 0.6296 |
| V/F F1 | 0.7340 |
| 推理失败 | 0 / 900 |
| 总耗时 | 81.0 s |
| 平均耗时 | 90.0 ms / 心搏 |
| 主机与板端预测一致率 | 86.11% |

板端混淆矩阵（行是真实标签，列是预测标签）：

```text
          Pred N  Pred S  Pred V/F
True N       281      19         0
True S       123     170         7
True V/F      71      51       178
```

同一批样本的主机参考结果为 Accuracy 0.6889、Macro-F1 0.6779。简历和演示应报告真实板端结果，不能把 PPQ 模拟结果写成板端结果。

## 8. 为什么移动干扰明显

ECG 是毫伏级信号。电极移动会改变电极—皮肤接触阻抗并产生远大于 ECG 的瞬态电压；导线晃动会引入感应噪声，肌肉收缩产生的肌电又与 ECG 频段重叠。因此，仅靠高通、低通或陷波不能彻底消除运动伪差。

优先改善方法：使用 Ag/AgCl 湿电极、清洁皮肤、缩短并固定导线、使用胸部/躯干位置、可靠连接 RL，并在静止状态下采集。

## 9. 后续改进路线

### P0：信号质量评估（下一步优先）

在推理前计算实时 SQI：

- 导联脱落状态；
- ADC 接近 0/4095 的饱和比例；
- 短时幅度跳变与异常斜率；
- 基线漂移和高频噪声能量；
- R 峰间隔合理性；
- 有效心搏比例。

输出 `GOOD / POOR / LEAD_OFF`。质量差时继续显示波形，但暂停心率更新和分类，界面提示“运动干扰，请保持静止”。这比强行给出高置信度类别更可靠。

### P1：加入运动传感器

建议增加 MPU6050 或 ICM-42688 三轴加速度计，通过 I2C 接入 ESP32-S3，分阶段实现：

1. **同步采集**：ECG 360 Hz，加速度 50–100 Hz，统一 ESP32 时间戳。
2. **运动强度计算**：计算三轴合加速度、短时方差和 jerk（加速度变化率）。
3. **运动门控**：超过阈值时标记运动污染，暂停分类并记录事件。
4. **联合 SQI**：将 ECG 质量指标和加速度指标融合为 0–100 的质量分数。
5. **自适应消噪实验**：以加速度为参考输入，对比 LMS/NLMS 前后的 R 峰召回率和 Macro-F1。
6. **数据闭环**：保存静止、走动、抬手和晃动导线等数据，建立个人运动伪差测试集。

运动传感器的首要作用是判断“当前结果是否可信”，而不是保证还原所有受污染波形。肌电与 ECG 频谱重叠，即使有加速度计也不能无损恢复全部信号。

### P2：分类决策

- 最近 3–5 个心搏投票或指数平滑；
- 连续多次异常才提示，单次异常标记“待复核”；
- 信号质量差时冻结分类；
- 增加低置信度/未知类拒识；
- 保存异常心搏前后数秒数据供电脑端复核。

### P3：采集链路

- 将定时器单次 ADC 读取升级为 ADC continuous/DMA，并测量采样抖动；
- 评估更高线性度的外置 ADC；
- 改善 PCB、外壳、导线固定与屏蔽；
- 若修改带通范围或加入滤波器，必须用相同预处理重新验证或训练模型。

## 10. 已知限制

- `LO+ / LO-` 只能检测导联脱落，不能证明 ECG 质量良好。
- 当前没有运动传感器与 SQI 拒识，移动尖峰可能被误判为 R 峰。
- 单导联只能进行有限节律筛查，不能替代 12 导联 ECG。
- 当前 S 类较弱，板端 S 类 F1 为 0.6296。
- 主机模拟与 ESP-DL 板端预测一致率为 86.11%，两者不能混为同一结果。
- 输出是单次心搏类别，不是疾病诊断结论。

## 11. 主要文件

```text
esp32_ecg_inference/
├─ main/
│  ├─ app_main.cpp                 # 实时采集与双核任务调度
│  ├─ ecg_dsp.cpp/.hpp             # 因果滤波、标准化、R 峰检测
│  ├─ ecg_pipeline.cpp/.hpp        # 心搏与 RR 特征构造
│  ├─ ecg_model.cpp/.hpp           # ESP-DL 推理与离线评估
│  ├─ ecg_generated_config.hpp     # 滤波系数、统计量、测试向量
│  ├─ ppg_max30102.cpp/.hpp        # MAX30102 血氧/心率驱动（独立，未接入主链路）
  ├─ ppg_analyzer.cpp/.hpp        # PPG 峰值/心率/SpO2/PAT 上层分析（独立，未接入主链路）
│  ├─ models/ecg.espdl             # W16A16 部署模型
│  └─ data/board_eval.bin          # 900 心搏评估数据
├─ tools/
│  ├─ export_firmware_assets.py    # 固件配置与测试资产导出
│  └─ stage1_espdl_quantize.py     # ESP-DL 量化导出
├─ plot_live_ecg.py                # Windows 实时可视化（matplotlib 版）
└─ plot_live_ecg_v2.py             # Windows 实时可视化（pyqtgraph 增强版）
```

日常实时固件默认不嵌入 900 心搏评估数据，以减小固件并缩短编译时间。

## 12. MAX30102（PPG）驱动说明

`ppg_max30102.cpp/.hpp` 提供 MAX30102 血氧/心率传感器的底层驱动，与 ECG 主链路完全独立，默认不参与编译行为（文件已加入构建，但 `ECG_PPG_ENABLE` 开关默认关闭，未在 `app_main` 中实例化）。

### 接线（本项目实际连接）

| MAX30102 | ESP32-S3 | 说明 |
|---|---|---|
| VIN / VCC | 3V3 | 供电 |
| GND | GND | 共地 |
| SDA | GPIO8 | I2C 数据 |
| SCL | GPIO9 | I2C 时钟 |
| INT | 暂不接（或 GPIO4） | 驱动使用轮询，无需中断 |

### 关键参数

- I2C 地址：`0x57`（7 位）
- 采样率：100 Hz（SpO2 模式）
- LED 电流：约 7.2 mA（红 / 红外）
- 波特率：100 kHz（保守，稳定后可上 400 kHz）

### 注意

- **GPIO8 复用风险**：GPIO8 默认是 FSPICLK。若板子使用 Octal/OPI Flash 或 PSRAM，GPIO8 会被占用。Quad Flash 模块（常见 N16R8 配置）下 GPIO8/GPIO9 可用。若 `init()` 读到 PART ID 异常，优先排查此项。
- 驱动仅做「初始化 + 自检 + FIFO 读取」，**心率 / SpO₂ 计算、与 ECG 的 PAT 对齐、真峰验证**等上层逻辑尚未实现，属于后续融合工作。

### 快速验证

在 `app_main` 中临时实例化并读取 PART ID 即可验证接线：

```cpp
ecg::Max30102 ppg;
if (ppg.init() == ESP_OK) {
    ecg::PpgSample sample;
    while (true) {
        if (ppg.read(sample)) {
            ESP_LOGI("PPG", "red=%lu ir=%lu", (unsigned long)sample.red,
                     (unsigned long)sample.ir);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
```

依赖：`esp_driver_i2c`（已在 `main/CMakeLists.txt` 的 `PRIV_REQUIRES` 中加入）。

## 13. PPG 上层分析（ppg_analyzer）

`ppg_analyzer.cpp/.hpp` 在 MAX30102 驱动之上实现：

- **PPG 峰值检测**：基于 IR 交流分量的上升沿 + 回落确认，带最小峰间隔（300 ms）抑制重搏切迹和运动尖峰。
- **心率**：由相邻 PPG 峰峰间隔计算，生理范围过滤（30–220 BPM）。
- **SpO₂ 估算**：用经典线性化公式 `SpO₂ ≈ 110 − 25·R`，其中 `R = (AC_red/DC_red)/(AC_ir/DC_ir)`。**仅为粗略估计，未经参考血氧仪标定**。
- **PAT（脉搏到达时间）对齐**：`align_to_ecg()` 在 ECG R 峰之后 150–500 ms 的生理窗口内找最近的 PPG 峰，返回 PAT。这是 ECG+PPG 联合才有的指标，可用于血压趋势 / 血管弹性研究，以及 ECG「真峰 vs 伪峰」判别。

### 时间基准（集成时务必注意）

PPG 用 `esp_timer_get_time()`（微秒）打时间戳，ECG 侧目前用整数采样序号（`sample_index`）。要做 PAT 对齐，**ECG 的 R 峰必须以同一 `esp_timer_get_time()` 时钟打点**（即 `sample_index * 1000000 / 360`，或在 R 峰确认时刻直接取 `esp_timer_get_time()`）。两个时钟不统一会导致 PAT 系统性偏移。

### 集成示例

```cpp
ecg::Max30102 ppg;
ecg::PpgAnalyzer analyzer;
if (ppg.init() != ESP_OK) { /* 处理失败 */ }

// PPG 读取任务（建议独立任务，100 Hz 轮询）
while (true) {
    ecg::PpgSample sample;
    if (ppg.read(sample)) {
        analyzer.push(sample);
        if (auto hr = analyzer.heart_rate(); hr.valid) {
            // hr.bpm 可用
        }
        if (auto spo2 = analyzer.spo2(); spo2.valid) {
            // spo2.percent 可用
        }
    }
    vTaskDelay(pdMS_TO_TICKS(10));
}
```

ECG 侧拿到 R 峰后（需带微秒时间戳）：

```cpp
ecg::EcgRPeak r_peak;
r_peak.timestamp_us = /* R 峰的 esp_timer 时间戳 */;
r_peak.valid = true;
ecg::PatMeasurement pat = analyzer.align_to_ecg(r_peak);
// pat.valid 为 true 时，pat.pat_us 即脉搏到达时间
```

> 说明：以上「真峰验证」的完整闭环（PAT 超窗 → 标记 ECG 伪峰）尚未在 `app_main` 中实现，属于后续融合工作。当前 ppg_analyzer 是独立的、可直接单测的分析模块。

## 14. PPG 集成（ppg_integration）

`ppg_integration.cpp/.hpp` 把驱动 + 分析器 + ECG 推理任务串成完整闭环：

- 创建一个独立 PPG 采样任务（100 Hz，CPU0）；
- 在推理任务收到心搏时，用 `validate_r_peak(r_sample)` 做「真峰验证」：把 ECG R 峰采样序号换算成微秒时间戳，在 PPG 峰环形缓冲里找 150–500 ms 窗口内的对应脉搏峰；
- 若找到 → 该心搏被 PPG 佐证（`peak_corroborated=true`），否则标记为「未验证」。

### 启用方式

在 `idf.py menuconfig` 里打开 `PPG (MAX30102) → Enable MAX30102 PPG sensor`，或在 `sdkconfig` 里加 `CONFIG_ECG_PPG_ENABLE=y`。默认关闭，固件行为不变。

### PRED 串口输出变化（仅启用时）

启用后 `PRED` 行会从原来的 10 个字段扩展为 14 个字段，追加：

```text
...,corroborated,pat_us,ppg_bpm,ppg_spo2
```

- `corroborated`：1 = PPG 佐证了该 R 峰，0 = 未佐证；
- `pat_us`：脉搏到达时间（微秒），未佐证时为 0；
- `ppg_bpm` / `ppg_spo2`：PPG 侧心率和 SpO₂。

**主机端若启用 PPG，需要同步更新 `plot_live_ecg.py` / `plot_live_ecg_v2.py` 的 `PRED` 解析字段数**（当前是 10，需改为 14），否则解析会静默失败。

### 当前定位

`corroborated=0` 目前只是**信息性标记**，不会丢弃分类结果。这样设计是为了先积累「ECG 伪峰 vs PPG 佐证」的对齐数据，观察 PAT 窗口是否合理，之后再决定「是否用 corroborated 来冻结分类」。这与 README 第 9 节 P2「信号质量差时冻结分类」的路线一致——先量化，再决定门控策略。
