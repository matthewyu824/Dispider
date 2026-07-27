# Online Video LLM Testbed

This testbed turns a local video into a real-time browser media source without
requesting camera access. Chrome/Chromium plays the selected file, captures the
player's media track, and sends it to the server over WebRTC. The server keeps
only a sliding window of frames that have already arrived.

## Architecture

```text
local video -> HTML video clock -> WebRTC video track -> aiortc receiver
            -> 4 FPS frame sampler -> sliding window -> video LLM worker
            -> timestamped answer + latency metrics -> browser dashboard
```

The model worker is persistent and bound to one GPU, so repeated windows do not
reload the checkpoint. GPU requests are serialized per device to avoid the
out-of-memory failures caused by overlapping inference.

The browser exposes live encoder controls backed by
`RTCRtpSender.setParameters()`. Bitrate caps, maximum frame rate, and
resolution scaling can be changed while a session is running. The panel also
shows the actual outbound resolution, frame rate, bitrate, and browser-reported
quality limitation reason.

This is an online evaluation baseline, not an exact reconstruction of the
paper's asynchronous perception, decision, and reaction state. Each reaction
currently performs a new inference over the latest observed window.

## Install

The existing model environment is required. Add the online transport dependencies:

```bash
.venv/bin/pip install -r web_demo/requirements-online.txt
```

## Run

Real model:

```bash
scripts/run_online_testbed.sh
```

Transport/UI validation without loading the checkpoint:

```bash
scripts/run_online_testbed.sh --mock-model
```

The default URL is `http://localhost:7860`. Set `ONLINE_VIDEO_LLM_PORT` or
`ONLINE_VIDEO_LLM_HOST` to override it. Set `ONLINE_VIDEO_LLM_MODEL_PATH` when
the checkpoint is not under `checkpoints/`.

## Online protocol

- `POST /api/online/sessions`: create a windowed inference session.
- `POST /api/rtc/offer`: negotiate the browser video track and telemetry data
  channel.
- `GET /api/online/sessions/{id}`: read transport, buffer, inference, and
  response state.
- `GET /api/online/sessions/{id}/preview.jpg`: inspect the latest frame
  decoded and sampled on the server.
- `POST /api/online/sessions/{id}/query`: trigger an immediate reaction.
- `POST /api/online/sessions/{id}/stop`: close the peer connection.

The dashboard displays the latest server-received frame and records media-time
windows, received and sampled frame counts, sender bitrate, transport lag,
WebRTC state, total response latency, model latency, and the ordered event
timeline.
