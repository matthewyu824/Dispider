const $ = (selector) => document.querySelector(selector);
const state = {
  sample: "test",
  file: null,
  job: null,
  poll: null,
  health: null,
  onlineFile: null,
  onlineObjectUrl: null,
  onlineSession: null,
  onlinePeer: null,
  onlineSender: null,
  onlineChannel: null,
  onlinePoll: null,
  telemetryTimer: null,
  lastRtcBytes: 0,
  lastRtcTimestamp: 0,
  lastPreviewFrame: 0,
};
const stageOrder = ["queued", "preparing", "transcoding", "loading", "completed"];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function showVideo(url, label) {
  const player = $("#videoPlayer");
  player.src = url;
  $("#dropZone").classList.add("has-video");
  $("#videoMeta").textContent = label;
  player.load();
}

function selectSample(id, name) {
  state.sample = id;
  state.file = null;
  document.querySelectorAll(".sample").forEach((button) => {
    button.classList.toggle("active", button.dataset.id === id);
  });
  showVideo(`/sample/${id}`, name);
}

function setStage(stage, message) {
  const normalized = stage === "transcoding" ? "preparing" : stage;
  const current = stageOrder.indexOf(normalized);
  document.querySelectorAll("#stageList div").forEach((node) => {
    const index = stageOrder.indexOf(node.dataset.stage);
    node.classList.toggle("done", index >= 0 && index < current);
    node.classList.toggle("current", node.dataset.stage === normalized);
  });
  if (message) $("#runtime").textContent = message;
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    state.health = health;
    $("#systemText").textContent = health.model_ready
      ? `${health.gpus.length} GPU · 模型就绪`
      : "模型未就绪";
    $("#sampleStrip").innerHTML =
      health.samples
        .filter((item) => item.available)
        .map(
          (item) =>
            `<button class="sample" data-id="${escapeHtml(item.id)}">${escapeHtml(item.name)}</button>`,
        )
        .join("") +
      `<button class="sample" id="uploadButton">＋ 上传视频</button>`;
    document.querySelectorAll(".sample[data-id]").forEach((button) => {
      button.onclick = () => selectSample(button.dataset.id, button.textContent);
    });
    $("#uploadButton").onclick = () => $("#videoInput").click();
    const available = health.samples.filter((item) => item.available);
    $("#onlineSample").innerHTML = available
      .map(
        (item) =>
          `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`,
      )
      .join("");
    const first =
      available.find((item) => item.id === "test") || available[0];
    if (first) {
      selectSample(first.id, first.name);
      selectOnlineSample(first.id, first.name);
      $("#onlineSample").value = first.id;
    }
  } catch (error) {
    $("#systemText").textContent = "服务连接失败";
    $("#onlineError").textContent = error.message;
  }
}

async function loadEvaluation() {
  try {
    const data = await api("/api/evaluation");
    $("#metricAccuracy").textContent = `${data.accuracy}%`;
    $("#metricCorrect").textContent = `${data.correct}/${data.questions}`;
    $("#metricAnswered").textContent =
      `${Math.round((data.answered / Math.max(data.questions, 1)) * 100)}%`;
    $("#taskBreakdown").innerHTML = data.tasks
      .map(
        (task) =>
          `<div class="task-chip">${escapeHtml(task.name)} · ${task.correct}/${task.total} · ${task.accuracy}%</div>`,
      )
      .join("");
    $("#resultRows").innerHTML = data.rows
      .map((row) => {
        const options = row.options
          .map((option) => {
            const letter = option[0];
            const classes = [
              letter === row.prediction ? "predicted" : "",
              letter === row.answer ? "answer" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return `<li class="${classes}">${escapeHtml(option)}</li>`;
          })
          .join("");
        return `<tr><td><details><summary class="question-summary"><strong>视频 ${escapeHtml(row.video_id)} · ${escapeHtml(row.question_id)}</strong><span>${escapeHtml(row.question)}</span></summary><div class="question-detail"><ul class="option-list">${options}</ul><div class="raw-output">模型原始输出：${escapeHtml(row.response || "（空）")}</div></div></details></td><td>${escapeHtml(row.task_type)}</td><td class="answer-cell ${row.correct ? "correct" : "wrong"}"><b>${escapeHtml(row.prediction || "—")}</b><span>${escapeHtml(row.prediction_text)}</span></td><td class="answer-cell"><b>${escapeHtml(row.answer)}</b><span>${escapeHtml(row.answer_text)}</span></td><td class="${row.correct ? "status-good" : "status-bad"}">${row.correct ? "✓ 正确" : "× 错误"}</td></tr>`;
      })
      .join("");
  } catch (error) {
    $("#resultRows").innerHTML =
      `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function pollJob(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    state.job = job;
    setStage(job.stage, job.message);
    if (job.status === "completed") {
      clearInterval(state.poll);
      state.poll = null;
      $("#answer").textContent = job.answer;
      $("#answer").classList.add("ready");
      $("#runtime").textContent = `GPU ${job.gpu} · ${job.elapsed_seconds}s`;
      $("#runButton").disabled = false;
      $("#runButton span").textContent = "再次推理";
    } else if (job.status === "failed") {
      clearInterval(state.poll);
      state.poll = null;
      $("#errorText").textContent = job.error;
      $("#answer").textContent = "任务未完成";
      $("#runButton").disabled = false;
      $("#runButton span").textContent = "重试";
    }
  } catch (error) {
    $("#errorText").textContent = error.message;
  }
}

$("#inferForm").onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData();
  form.append("prompt", $("#prompt").value);
  form.append("gpu", $("#gpu").value);
  form.append("max_new_tokens", $("#tokens").value);
  form.append("max_clips", $("#clips").value);
  if (state.file) form.append("video", state.file);
  else form.append("sample_id", state.sample);
  $("#runButton").disabled = true;
  $("#runButton span").textContent = "任务运行中";
  $("#answer").textContent = "模型正在观察视频…";
  $("#answer").classList.remove("ready");
  $("#errorText").textContent = "";
  setStage("queued", "正在创建任务");
  try {
    const job = await api("/api/infer", { method: "POST", body: form });
    state.job = job;
    state.poll = setInterval(() => pollJob(job.id), 1200);
    pollJob(job.id);
  } catch (error) {
    $("#errorText").textContent = error.message;
    $("#runButton").disabled = false;
  }
};

$("#videoInput").onchange = (event) => {
  const file = event.target.files[0];
  if (!file) return;
  state.file = file;
  state.sample = "";
  document
    .querySelectorAll(".sample")
    .forEach((button) => button.classList.remove("active"));
  showVideo(
    URL.createObjectURL(file),
    `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`,
  );
};
$("#dropZone").onclick = (event) => {
  if (
    event.target.tagName !== "VIDEO" &&
    !state.file &&
    !state.sample
  )
    $("#videoInput").click();
};
$("#dropZone").ondragover = (event) => event.preventDefault();
$("#dropZone").ondrop = (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) {
    state.file = file;
    state.sample = "";
    showVideo(URL.createObjectURL(file), file.name);
  }
};

function showOnlineVideo(url, label, isObjectUrl = false) {
  const player = $("#onlineVideoPlayer");
  if (state.onlineObjectUrl) URL.revokeObjectURL(state.onlineObjectUrl);
  state.onlineObjectUrl = isObjectUrl ? url : null;
  player.src = url;
  player.load();
  $("#onlineDropZone").classList.add("has-video");
  $("#onlineSourceMeta").textContent = label;
}

function selectOnlineSample(id, name) {
  if (!id || state.onlineSession) return;
  state.onlineFile = null;
  showOnlineVideo(`/sample/${id}`, name);
}

function selectOnlineFile(file) {
  if (!file || state.onlineSession) return;
  state.onlineFile = file;
  showOnlineVideo(
    URL.createObjectURL(file),
    `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`,
    true,
  );
}

$("#onlineUploadButton").onclick = () => $("#onlineVideoInput").click();
$("#onlineVideoInput").onchange = (event) =>
  selectOnlineFile(event.target.files[0]);
$("#onlineSample").onchange = (event) => {
  const option = event.target.selectedOptions[0];
  selectOnlineSample(event.target.value, option?.textContent || "示例视频");
};
$("#onlineDropZone").ondragover = (event) => event.preventDefault();
$("#onlineDropZone").ondrop = (event) => {
  event.preventDefault();
  selectOnlineFile(event.dataTransfer.files[0]);
};

function setOnlineState(label, className = "idle") {
  const badge = $("#onlineState");
  badge.textContent = label;
  badge.className = `state-badge ${className}`;
}

function resetReceiverPreview() {
  state.lastPreviewFrame = 0;
  $("#receiverPreview").removeAttribute("src");
  $("#receiverStage").classList.remove("has-frame");
  $("#receiverMeta").textContent = "等待首帧";
  $("#receiverFrames").textContent = "0";
  $("#receiverSampled").textContent = "0";
  $("#receiverTime").textContent = "0.0s";
  $("#receiverLag").textContent = "—";
}

function readEncodingSettings() {
  return {
    bitrateKbps: Number($("#encodingBitrate").value),
    maxFramerate: Number($("#encodingFramerate").value),
    scaleResolutionDownBy: Number($("#encodingScale").value),
  };
}

function encodingSummary(settings) {
  const bitrate = settings.bitrateKbps
    ? `${settings.bitrateKbps} Kbps`
    : "自动码率";
  const framerate = settings.maxFramerate
    ? `${settings.maxFramerate} FPS`
    : "源帧率";
  const resolution = settings.scaleResolutionDownBy === 1
    ? "原始分辨率"
    : `${Math.round(100 / settings.scaleResolutionDownBy)}% 分辨率`;
  return `${bitrate} · ${framerate} · ${resolution}`;
}

async function applyEncodingSettings() {
  const settings = readEncodingSettings();
  const status = $("#encodingStatus");
  if (!state.onlineSender) {
    status.textContent = `已保存：${encodingSummary(settings)}，将在会话启动时应用`;
    status.className = "";
    return false;
  }

  const button = $("#applyEncodingButton");
  button.disabled = true;
  status.textContent = "正在应用编码参数…";
  status.className = "";
  try {
    const parameters = state.onlineSender.getParameters();
    if (!parameters.encodings?.length) {
      throw new Error("浏览器编码器尚未就绪");
    }
    const encoding = parameters.encodings[0];
    if (settings.bitrateKbps > 0) {
      encoding.maxBitrate = settings.bitrateKbps * 1000;
    } else {
      delete encoding.maxBitrate;
    }
    if (settings.maxFramerate > 0) {
      encoding.maxFramerate = settings.maxFramerate;
    } else {
      delete encoding.maxFramerate;
    }
    encoding.scaleResolutionDownBy = settings.scaleResolutionDownBy;
    await state.onlineSender.setParameters(parameters);
    status.textContent = `已应用：${encodingSummary(settings)}`;
    status.className = "encoding-applied";
    return true;
  } catch (error) {
    status.textContent = `应用失败：${error.message}`;
    status.className = "encoding-error";
    return false;
  } finally {
    button.disabled = false;
  }
}

function waitForVideo(player) {
  if (player.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("视频加载超时")),
      15000,
    );
    player.addEventListener(
      "loadeddata",
      () => {
        clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
    player.addEventListener(
      "error",
      () => {
        clearTimeout(timeout);
        reject(new Error("浏览器无法读取该视频"));
      },
      { once: true },
    );
  });
}

function waitForIceGathering(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (peer.iceGatheringState === "complete") {
        peer.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", check);
  });
}

async function rtcTelemetry() {
  if (!state.onlinePeer || !state.onlineSession) return;
  const stats = await state.onlinePeer.getStats();
  let videoStats = null;
  stats.forEach((report) => {
    if (
      report.type === "outbound-rtp" &&
      !report.isRemote &&
      (report.kind === "video" || report.mediaType === "video")
    ) {
      videoStats = report;
    }
  });
  if (!videoStats) return;
  let bitrate = 0;
  if (
    state.lastRtcTimestamp &&
    videoStats.timestamp > state.lastRtcTimestamp
  ) {
    bitrate =
      ((videoStats.bytesSent - state.lastRtcBytes) * 8) /
      (videoStats.timestamp - state.lastRtcTimestamp);
  }
  state.lastRtcBytes = videoStats.bytesSent;
  state.lastRtcTimestamp = videoStats.timestamp;
  const frameWidth = Number(videoStats.frameWidth || 0);
  const frameHeight = Number(videoStats.frameHeight || 0);
  const actualFps = Number(videoStats.framesPerSecond || 0);
  const resolution = frameWidth && frameHeight
    ? `${frameWidth}×${frameHeight}`
    : "分辨率待定";
  const qualityReason = videoStats.qualityLimitationReason &&
    videoStats.qualityLimitationReason !== "none"
    ? ` · 限制原因 ${videoStats.qualityLimitationReason}`
    : "";
  $("#encodingActual").textContent =
    `实际编码：${resolution} · ${actualFps ? `${actualFps.toFixed(0)} FPS` : "帧率待定"} · ${bitrate ? `${Math.round(bitrate)} Kbps` : "码率采集中"}${qualityReason}`;
  if (state.onlineChannel?.readyState === "open") {
    const player = $("#onlineVideoPlayer");
    state.onlineChannel.send(
      JSON.stringify({
        type: "telemetry",
        sent_at: Date.now(),
        duration: Number.isFinite(player.duration) ? player.duration : 0,
        playhead: player.currentTime,
        fps: videoStats.framesPerSecond || 0,
        bitrate_kbps: bitrate,
        packets_sent: videoStats.packetsSent || 0,
      }),
    );
  }
}

function renderOnlineSession(session) {
  state.onlineSession = session;
  const metrics = session.metrics;
  const preview = session.preview || {};
  $("#currentOnlineQuestion").textContent = session.prompt || "—";
  $("#transportMetric").textContent = session.transport_state;
  $("#playheadMetric").textContent = `${session.media_time.toFixed(1)}s`;
  $("#bufferMetric").textContent = `${metrics.buffered_seconds.toFixed(1)}s`;
  $("#bitrateMetric").textContent = Math.round(metrics.bitrate_kbps);
  $("#latencyMetric").textContent = metrics.last_inference_ms
    ? `${(metrics.last_inference_ms / 1000).toFixed(1)}s`
    : "—";
  $("#responseMetric").textContent = session.responses.length;
  $("#receiverFrames").textContent = metrics.received_frames;
  $("#receiverSampled").textContent = metrics.sampled_frames;
  $("#receiverTime").textContent =
    `${Number(preview.media_time || 0).toFixed(1)}s`;
  $("#receiverLag").textContent = metrics.transport_lag_ms
    ? `${Math.round(metrics.transport_lag_ms)} ms`
    : "—";
  $("#receiverMeta").textContent = preview.available
    ? `${Number(preview.media_time).toFixed(1)}s · 第 ${preview.sampled_frame} 个采样帧`
    : "等待首帧";
  if (
    preview.available &&
    preview.sampled_frame !== state.lastPreviewFrame
  ) {
    const image = $("#receiverPreview");
    image.onload = () => $("#receiverStage").classList.add("has-frame");
    image.src = `${preview.url}?frame=${preview.sampled_frame}`;
    state.lastPreviewFrame = preview.sampled_frame;
  }
  if (session.inference_running) {
    setOnlineState("模型推理中", "working");
  } else if (
    ["streaming", "connected"].includes(session.status) ||
    session.transport_state === "connected"
  ) {
    setOnlineState("正在接收", "live");
  } else if (session.status === "failed") {
    setOnlineState("连接失败", "failed");
  } else if (session.status === "stopped") {
    setOnlineState("已停止", "idle");
  } else {
    setOnlineState(session.transport_state, "working");
  }
  $("#onlineAnswers").innerHTML = session.responses.length
    ? [...session.responses]
        .reverse()
        .map((response) => {
          const answered = Boolean(response.answer);
          const answerText = answered
            ? response.answer
            : "模型在这个时间窗结束时未生成可见文本。该窗口已完成，并非仍在推理。";
          return `<article class="online-answer-card ${answered ? "" : "silent"}">
            <div><strong>#${response.sequence}</strong><span>${response.media_start.toFixed(1)}–${response.media_end.toFixed(1)}s · ${response.frames} 帧 · ${(response.total_latency_ms / 1000).toFixed(1)}s</span><em>${answered ? "已回答" : "保持静默"}</em></div>
            <small>问题：${escapeHtml(response.question || session.prompt)}</small>
            <p>${escapeHtml(answerText)}</p>
          </article>`;
        })
        .join("")
    : `<p class="empty-copy">${session.inference_running ? "模型正在处理第一个时间窗…" : "视频开始播放后，服务端将积累在线窗口。"}</p>`;
  $("#onlineTimeline").innerHTML = session.events.length
    ? [...session.events]
        .reverse()
        .map(
          (event) =>
            `<li class="event-${escapeHtml(event.kind)}"><time>${event.media_time.toFixed(1)}s</time><span>${escapeHtml(event.message)}</span></li>`,
        )
        .join("")
    : "<li><time>—</time><span>等待事件</span></li>";
}

async function pollOnlineSession() {
  if (!state.onlineSession) return;
  try {
    const session = await api(
      `/api/online/sessions/${state.onlineSession.id}`,
    );
    renderOnlineSession(session);
  } catch (error) {
    $("#onlineError").textContent = error.message;
  }
}

async function stopOnlineSession() {
  const session = state.onlineSession;
  clearInterval(state.onlinePoll);
  clearInterval(state.telemetryTimer);
  state.onlinePoll = null;
  state.telemetryTimer = null;
  $("#onlineVideoPlayer").pause();
  if (state.onlinePeer) state.onlinePeer.close();
  state.onlinePeer = null;
  state.onlineSender = null;
  state.onlineChannel = null;
  if (session) {
    try {
      const stopped = await api(`/api/online/sessions/${session.id}/stop`, {
        method: "POST",
      });
      renderOnlineSession(stopped);
    } catch (error) {
      $("#onlineError").textContent = error.message;
    }
  }
  state.onlineSession = null;
  resetReceiverPreview();
  $("#encodingStatus").textContent = "将在下次会话启动时应用";
  $("#encodingStatus").className = "";
  $("#encodingActual").textContent = "实际编码：等待 WebRTC 统计";
  $("#onlineStartButton").disabled = false;
  $("#onlineStartButton span").textContent = "启动在线会话";
  $("#onlineStopButton").disabled = true;
  $("#manualQueryButton").disabled = true;
  $("#onlineUploadButton").disabled = false;
  $("#onlineSample").disabled = false;
}

$("#onlineForm").onsubmit = async (event) => {
  event.preventDefault();
  $("#onlineError").textContent = "";
  const player = $("#onlineVideoPlayer");
  const captureStream = player.captureStream || player.mozCaptureStream;
  if (!player.src) {
    $("#onlineError").textContent = "请先选择本地视频或示例视频";
    return;
  }
  if (!captureStream) {
    $("#onlineError").textContent =
      "当前浏览器不支持 HTMLVideoElement.captureStream，请使用 Chrome 或 Chromium。";
    return;
  }
  $("#onlineStartButton").disabled = true;
  $("#onlineStartButton span").textContent = "正在建立 WebRTC";
  $("#onlineUploadButton").disabled = true;
  $("#onlineSample").disabled = true;
  resetReceiverPreview();
  setOnlineState("建立连接", "working");
  try {
    await waitForVideo(player);
    const session = await api("/api/online/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("#onlinePrompt").value,
        gpu: Number($("#onlineGpu").value),
        max_new_tokens: Number($("#onlineTokens").value),
        window_seconds: Number($("#onlineWindow").value),
        interval_seconds: Number($("#onlineInterval").value),
        max_clips: 1,
        auto_query: $("#onlineAutoQuery").checked,
      }),
    });
    state.onlineSession = session;
    const peer = new RTCPeerConnection();
    state.onlinePeer = peer;
    const channel = peer.createDataChannel("telemetry");
    state.onlineChannel = channel;
    peer.onconnectionstatechange = () => {
      $("#transportMetric").textContent = peer.connectionState;
      if (peer.connectionState === "failed") {
        $("#onlineError").textContent = "WebRTC 连接失败，请检查服务器网络或防火墙。";
      }
    };
    await player.play();
    const stream = captureStream.call(player);
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) throw new Error("没有从视频播放器捕获到视频轨道");
    state.onlineSender = peer.addTrack(videoTrack, stream);
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGathering(peer);
    const answer = await api("/api/rtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session.id,
        sdp: peer.localDescription.sdp,
        type: peer.localDescription.type,
      }),
    });
    await peer.setRemoteDescription(answer);
    await applyEncodingSettings();
    state.lastRtcBytes = 0;
    state.lastRtcTimestamp = 0;
    state.onlinePoll = setInterval(pollOnlineSession, 1000);
    state.telemetryTimer = setInterval(rtcTelemetry, 1000);
    $("#onlineStartButton span").textContent = "在线会话运行中";
    $("#onlineStopButton").disabled = false;
    $("#manualQueryButton").disabled = false;
    renderOnlineSession(session);
    pollOnlineSession();
  } catch (error) {
    $("#onlineError").textContent = error.message;
    await stopOnlineSession();
  }
};

$("#onlineStopButton").onclick = stopOnlineSession;
$("#applyEncodingButton").onclick = applyEncodingSettings;
["#encodingBitrate", "#encodingFramerate", "#encodingScale"].forEach(
  (selector) => {
    $(selector).onchange = () => {
      const settings = readEncodingSettings();
      $("#encodingStatus").textContent = state.onlineSender
        ? `待应用：${encodingSummary(settings)}`
        : `已保存：${encodingSummary(settings)}，将在会话启动时应用`;
      $("#encodingStatus").className = "";
    };
  },
);
$("#manualQueryButton").onclick = async () => {
  if (!state.onlineSession) return;
  $("#onlineError").textContent = "";
  try {
    await api(`/api/online/sessions/${state.onlineSession.id}/query`, {
      method: "POST",
    });
    pollOnlineSession();
  } catch (error) {
    $("#onlineError").textContent = error.message;
  }
};

function switchView(view) {
  document.querySelectorAll(".nav-link").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  $("#studioView").classList.toggle("active", view === "studio");
  $("#onlineView").classList.toggle("active", view === "online");
  $("#evaluationView").classList.toggle("active", view === "evaluation");
  if (view === "evaluation") loadEvaluation();
}

document.querySelectorAll(".nav-link").forEach((button) => {
  button.onclick = () => switchView(button.dataset.view);
});

window.addEventListener("beforeunload", () => {
  if (state.onlinePeer) state.onlinePeer.close();
});

loadHealth();
loadEvaluation();
