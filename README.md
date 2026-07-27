# Online Video LLM Testbed

面向在线长视频理解研究的可运行 Testbed。系统既支持本地视频离线问答，也支持把本地视频按真实播放时钟通过 WebRTC 发送到服务端，在仅使用已经到达帧的前提下进行滑动窗口推理。

## 能力

- 本地视频离线问答，支持 GPU、采样帧数和最大输出长度配置
- WebRTC 在线视频传输，无需开启摄像头
- 基于媒体时间的滑动窗口与定时/手动触发
- 持久化单 GPU 模型 Worker，避免每次请求重复加载权重
- 传输状态、缓冲窗口、码率、推理延迟和回答时间线
- 服务端接收画面预览，可直接核对 WebRTC 实际传输结果
- WebRTC 编码面板，可在线调整码率上限、最大帧率和分辨率缩放
- VideoMME 小规模快速评测与逐题可追溯结果
- Mock 模式，可在不加载模型的情况下验证传输与界面

## 系统结构

```text
浏览器本地视频
  -> HTML 视频时钟
  -> WebRTC 视频轨
  -> 服务端帧采样与滑动窗口
  -> Online Video LLM Worker
  -> 时间戳回答、延迟指标与评测记录
```

当前实现是可测量的在线推理基线：每次触发都会对最新观测窗口执行一次推理，暂未跨窗口复用感知、决策和反应模块的隐藏状态。

## 环境

- Python 3.10
- CUDA 11.8
- PyTorch 2.2
- Transformers 4.41
- FFmpeg
- 两块 GPU 可并行承载独立会话；单次模型推理由一块 GPU 完成

安装在线传输依赖：

```bash
.venv/bin/pip install -r web_demo/requirements-online.txt
```

## 启动

真实模型：

```bash
scripts/run_online_testbed.sh
```

只验证 WebRTC 和界面：

```bash
scripts/run_online_testbed.sh --mock-model
```

默认地址为 `http://localhost:7860`。可通过以下变量覆盖：

```bash
ONLINE_VIDEO_LLM_HOST=0.0.0.0 ONLINE_VIDEO_LLM_PORT=7860 \
  scripts/run_online_testbed.sh
```

模型路径可通过 `ONLINE_VIDEO_LLM_MODEL_PATH` 指定。未指定时，服务会在 `checkpoints/` 下自动寻找兼容的长视频模型。

## 命令行推理

```bash
python inference.py \
  --model_path /path/to/model \
  --video_path /path/to/video.mp4 \
  --prompt "Please describe what happens in this video."
```

## VideoMME 快速评测

快速评测数据模板位于 `playground/data/`。配置视频路径和模型路径后执行：

```bash
bash scripts/eval/videomme.sh
```

网页评测看板会展示准确率、有效回答率、任务类型、每道题的全部选项、模型原始输出、预测答案和标准答案。

## 目录

```text
online_video_llm/   模型架构、视频预处理与评测代码
web_demo/           WebRTC 服务、模型 Worker 与前端
scripts/            启动、下载和评测脚本
playground/data/    快速评测配置
inference.py        单视频命令行推理入口
```

更完整的在线协议和实现边界见 [`web_demo/ONLINE_TESTBED.md`](web_demo/ONLINE_TESTBED.md)。

## 许可证与上游

本仓库基于 Apache License 2.0 发布。模型权重和数据集可能适用各自的许可证，使用前请分别确认。

本项目包含在开源视频语言模型实现基础上的修改，并新增了 WebRTC 传输、滑动窗口调度、持久化 Worker、可观测指标、评测看板和工程化运行入口。上游来源、论文引用和修改声明见 [`NOTICE`](NOTICE)。
