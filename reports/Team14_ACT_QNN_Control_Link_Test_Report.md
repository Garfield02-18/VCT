# Team14 项目 ACT/QNN 替代后端控制链路测试报告

## 1. 测试目的

本次测试的目标不是复现 Team14 原项目的 ManiFlow handover 实验结果，而是在缺少 Team14 原始训练权重和 QNN 部署产物的情况下，验证 Team14 项目的控制链路是否能够接入一个真实可运行的 QNN/HTP 后端并完成端到端调用。

Team14 仓库中缺少以下关键产物：

```text
checkpoints/<task>/maniflow_<task>_best.pt
checkpoints/<task>/normalizer.json
deploy/onnx/maniflow_handover_1step.onnx
deploy/qnn/*.dlc
deploy/qnn/context_binary/*.bin
```

因此，本次测试使用本地已有的 ACT/QNN-friendly action policy 作为替代后端，测试 Team14 的 `PolicyRunner`、动作后处理、安全检查和 QNN 调用链路是否可行。

## 2. 测试环境

测试平台：Radxa AIRbox Q900

操作系统：Ubuntu 24.04.4 LTS, aarch64

Team14 项目路径：

```bash
/home/radxa/Team14/AIR5051-Team14-code
```

替代 QNN 后端路径：

```bash
/home/radxa/VLA/convertibility/qnn_friendly_policy_bundle
```

QAIRT 路径：

```bash
/home/radxa/qairt/2.47.1
```

QNN/HTP context binary：

```bash
/home/radxa/VLA/convertibility/qnn_friendly_policy_bundle/htp_context/qnn_friendly_policy_htp.bin
```


## 3. 模型规模与精度信息

本次替代测试使用的 ACT/QNN-friendly action policy 来自本地 VLA QNN bundle。根据 `model_io.json` 与 ONNX initializer 统计结果，模型规模如下：

| 项目 | 数值 |
| --- | ---: |
| ONNX 参数量 | 478,365 |
| 参数量规模 | 0.478 M |
| ONNX 权重估算大小 | 1.82 MiB |
| ONNX initializer 数量 | 14 |
| ONNX node 数量 | 15 |
| HTP context binary 大小 | 约 1.1 MB |

输入输出精度如下：

| Tensor | Shape | 外部 IO dtype | Raw bytes |
| --- | --- | --- | ---: |
| `image` | `[1, 3, 480, 640]` | float32 | 3,686,400 |
| `qpos` | `[1, 14]` | float32 | 56 |
| `task_embedding` | `[1, 32]` | float32 | 128 |
| `actions` | `[1, 100, 14]` | float32 | 5,600 |

说明：这里的“精度”指模型对外暴露的输入/输出 tensor dtype 为 float32。QNN HTP context 内部的算子执行格式由 QAIRT/QNN 编译器和 HTP backend 决定，本报告不将其等同为完整 float32 计算或完整 int8 计算，只记录外部 IO 精度和实际运行结果。

任务层面的准确率/成功率未评估，因为当前测试没有使用 Team14 原始 handover 数据、真实机器人闭环和原始 ManiFlow 权重。本报告中的“精度”主要指模型数值精度/IO dtype，而不是任务成功率。

## 4. 替代模型接口说明

Team14 原始模型接口设计如下：

```text
输入:
rgb          [1, 3, 224, 224] float32
robot_state  [1, 14] float32

输出:
action_chunk [1, 16, 14] float32
```

本次使用的 ACT/QNN-friendly policy 接口如下：

```text
输入:
image          [1, 3, 480, 640] float32
qpos           [1, 14] float32
task_embedding [1, 32] float32

输出:
actions        [1, 100, 14] float32
```

两者都属于“根据视觉和机器人状态预测未来动作序列”的 action chunking policy，但 tensor contract 不完全一致。因此测试中新增了 adapter，将 ACT/QNN 输出适配为 Team14 控制链路可使用的动作格式。

适配关系如下：

```text
Team14 rgb [3,224,224]
        -> resize 到 ACT image [1,3,480,640]

Team14 robot_state [14]
        -> ACT qpos [1,14]

固定 task_embedding [1,32]
        -> 来自 ACT bundle sample_inputs/task_embedding.raw

ACT 输出 actions [1,100,14]
        -> 裁剪前 16 步，得到 [16,14]
        -> Team14 PolicyRunner 再取第 1 步，得到 [1,14]
```

## 5. 新增测试脚本

本次新增脚本：

```bash
/home/radxa/Team14/AIR5051-Team14-code/scripts/smoke_test_act_qnn_backend.py
```

脚本功能：

1. 构造 Team14 形状的 synthetic RGB 和 robot state。
2. 将 Team14 输入转换为 ACT/QNN 模型输入。
3. 调用真实 `qnn-net-run` 和 HTP context binary。
4. 读取 ACT/QNN 输出 `actions.raw`。
5. 将 `[1,100,14]` 输出裁剪为 Team14 需要的动作 chunk。
6. 接入 Team14 `PolicyRunner`。
7. 输出最终动作 `[1,14]`。
8. 使用 Team14 `SafetyMonitor` 做动作安全检查。

## 6. 测试命令

### 6.1 使用 ACT bundle 自带 sample image

```bash
cd /home/radxa/Team14/AIR5051-Team14-code
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py --use-sample-image
```

测试结果：

```text
ACTION_SHAPE=(1, 14)
LATENCY_MS=221.22
ACTION_FIRST14=[0.0746918, -0.0335884, 0.0460815, 0.0938416, -0.188293, -0.0352287, 0.141296, 0.00252485, 0.00400782, -0.000525475, -0.110626, 0.0116158, -0.0666428, -0.149765]
SAFETY_OK=True REASON=ok
WORK_DIR=/home/radxa/Team14/AIR5051-Team14-code/tmp/act_qnn_team14
[done] Team14 PolicyRunner successfully used ACT/QNN backend
```

### 6.2 使用 Team14 形状输入并 resize 到 ACT 输入尺寸

```bash
cd /home/radxa/Team14/AIR5051-Team14-code
PYTHONPATH=. python3 scripts/smoke_test_act_qnn_backend.py
```

测试结果：

```text
ACTION_SHAPE=(1, 14)
LATENCY_MS=242.54
ACTION_FIRST14=[0.0727844, -0.0348854, 0.0453949, 0.0957489, -0.188599, -0.0364685, 0.13916, 0.00401497, 0.00248909, -0.000452757, -0.109787, 0.00956535, -0.0682449, -0.149841]
SAFETY_OK=True REASON=ok
WORK_DIR=/home/radxa/Team14/AIR5051-Team14-code/tmp/act_qnn_team14
[done] Team14 PolicyRunner successfully used ACT/QNN backend
```

## 7. 测试产物

临时输入与 QNN 输出目录：

```bash
/home/radxa/Team14/AIR5051-Team14-code/tmp/act_qnn_team14
```

主要文件包括：

```text
qnn_run/image.raw
qnn_run/qpos.raw
qnn_run/input_list.txt
qnn_run/qnn_output_*/execution_metadata.yaml
team14_identity_normalizer.json
```

其中 `team14_identity_normalizer.json` 是为了让 Team14 `PolicyRunner` 可以完整执行归一化/反归一化流程而生成的临时 normalizer。



## 8. 性能测试数据

本次补充测试分别统计了两类性能数据：

1. 纯 QNN/HTP 后端多轮推理性能：直接使用 ACT bundle 的 `qnn-net-run --num_inferences 100`。
2. Team14 adapter 端到端性能：从 Team14 形状输入开始，完成 resize、raw 文件写入、`qnn-net-run` 调用、读取 `actions.raw`、裁剪动作并返回 `PolicyRunner` 输出。

### 8.1 纯 QNN/HTP 后端性能

测试命令：

```bash
cd /home/radxa/VLA/convertibility/qnn_friendly_policy_bundle
NUM_INFERENCES=100 ./run_on_airbox_htp.sh
```

QNN profiling 结果：

| 指标 | 数值 |
| --- | ---: |
| inferences_completed | 100 |
| NetRun IPS | 136.7716 inf/sec |
| 平均 NetRun execute | 5.842 ms |
| 平均 QNN execute | 5.814 ms |
| 平均 QNN accelerator execute | 4.583 ms |
| 平均 Accelerator execute | 4.545 ms |
| HVX threads | 4 |
| 最小 NetRun execute | 5.284 ms |
| 最大 NetRun execute | 6.733 ms |

按平均 NetRun execute 计算：

```text
FPS = 1000 / 5.842 ~= 171.17 FPS
```

但 QNN profile viewer 给出的整体 NetRun IPS 为：

```text
136.7716 inf/sec
```

两者差异来自统计口径不同：平均 execute 时间只统计单图执行阶段，而 NetRun IPS 包含 IO 和其他运行时开销。因此报告中优先使用 profile viewer 的 `NetRun IPS` 作为纯 QNN 后端吞吐参考。

### 8.2 Team14 adapter 端到端性能

测试方法：连续执行 10 次 Team14 adapter 调用，测试范围包含：

```text
Team14 synthetic RGB/state
    -> resize 到 ACT image [1,3,480,640]
    -> 写入 image.raw/qpos.raw/input_list.txt
    -> 调用 qnn-net-run
    -> 读取 actions.raw
    -> 裁剪 [1,100,14] 到 [16,14]
    -> Team14 PolicyRunner 输出 [1,14]
```

测试结果：

| 指标 | 数值 |
| --- | ---: |
| 测试次数 | 10 |
| 输出动作 shape | `(1, 14)` |
| 平均端到端延迟 | 256.65 ms |
| 最小端到端延迟 | 231.78 ms |
| 最大端到端延迟 | 275.38 ms |
| P50 延迟 | 258.01 ms |
| P90 延迟 | 270.18 ms |
| 端到端 FPS | 3.90 FPS |

端到端 FPS 低于纯 QNN 后端 FPS，主要原因是当前 adapter 每次调用都会启动一次 `qnn-net-run` 子进程，并进行 raw 文件写入/读取。这种方式适合 smoke test 和链路验证，但不适合最终实时部署。

### 8.3 参数量、精度与 FPS 对照

| 模型/测试路径 | 参数量 | 外部 IO 精度 | 输出动作 | 平均延迟 | FPS/IPS |
| --- | ---: | --- | --- | ---: | ---: |
| ACT/QNN 纯 HTP 后端 | 0.478 M | float32 IO | `[1,100,14]` | NetRun avg 5.842 ms | NetRun IPS 136.77 |
| Team14 adapter 端到端 | 0.478 M | float32 IO | `[1,14]` 裁剪后 | 256.65 ms | 3.90 FPS |

结论：在约 0.478M 参数、float32 外部 IO 的替代动作模型下，ACT/QNN 模型本身在 HTP 上具备较高推理吞吐；当前 Team14 adapter 端到端 FPS 主要受 Python subprocess 和文件 IO 影响，而不是模型本体计算量限制。

## 9. 测试结论

本次测试表明，在缺少 Team14 原始 QNN 部署产物的情况下，可以使用本地 ACT/QNN-friendly action policy 作为替代后端，成功接入 Team14 的控制链路。

已验证通过的链路包括：

```text
Team14 synthetic observation
    -> ACT/QNN adapter
    -> qnn-net-run + HTP context binary
    -> actions.raw [1,100,14]
    -> Team14 action chunk adapter
    -> Team14 PolicyRunner
    -> final action [1,14]
    -> SafetyMonitor
```

关键结果：

```text
最终动作 shape: (1, 14)
QNN 后端调用成功: 是
SafetyMonitor 检查: 通过
Team14 PolicyRunner 接入 ACT/QNN 后端: 成功
```

因此，Team14 项目的控制链路可以接入真实 QNN/HTP 后端进行工程级联调。该测试满足“验证整体项目控制链路可行性”的目标。

## 10. 限制与注意事项

本测试不代表 Team14 原项目复现成功，原因如下：

1. 使用的 QNN 后端不是 Team14 原始 ManiFlow 模型。
2. ACT/QNN 模型的训练任务、数据分布和 Team14 handover 任务不同。
3. 本次输入为 synthetic observation 或 ACT sample image，并非真实 handover 场景数据。
4. `task_embedding` 使用固定 sample embedding，并未与 Team14 任务语义对齐。
5. 输出动作通过安全检查只能说明数值范围合理，不能说明真实机器人执行有效。
6. Team14 仓库中的真实机器人接口和真实 observation source 仍然是占位实现，尚未连接实际相机和机械臂控制接口。

## 11. 后续建议

如果目标是继续做工程联调，建议下一步：

1. 将 `smoke_test_act_qnn_backend.py` 中的 synthetic RGB 替换为真实相机帧。
2. 将 synthetic `state` 替换为真实机器人 joint state。
3. 将固定 `task_embedding.raw` 替换为由任务解析模块生成的 embedding。
4. 将 adapter 封装成 Team14 的正式 backend，例如 `ActQnnBackend`。
5. 在上位机控制前继续保留 `SafetyMonitor`，并加入更严格的动作限幅和急停逻辑。
6. 如果未来获得 Team14 原始 ManiFlow checkpoint，再单独导出 ONNX/QNN 产物进行原模型后端测试。

## 12. 总体评价

在缺少 Team14 原始 checkpoint 和 QNN context binary 的情况下，本次测试通过 ACT/QNN-friendly policy 成功替代 Team14 模型后端，验证了 AIRbox 上真实 QNN/HTP 推理与 Team14 控制链路之间的可接入性。

该结果说明：

```text
Team14 控制链路具备接入真实 QNN 后端的工程可行性，
但当前测试仅验证链路，不验证原任务性能。
```
