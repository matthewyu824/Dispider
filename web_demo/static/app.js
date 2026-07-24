const $ = (s) => document.querySelector(s);
const state = { sample: "test", file: null, job: null, poll: null };
const stageOrder = ["queued", "preparing", "transcoding", "loading", "completed"];

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
  state.sample = id; state.file = null;
  document.querySelectorAll(".sample").forEach(b => b.classList.toggle("active", b.dataset.id === id));
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
    $("#systemText").textContent = health.model_ready ? "2 GPU · 模型就绪" : "模型未就绪";
    $("#sampleStrip").innerHTML = health.samples.filter(x => x.available).map(x => `<button class="sample" data-id="${x.id}">${x.name}</button>`).join("") + `<button class="sample" id="uploadButton">＋ 上传视频</button>`;
    document.querySelectorAll(".sample[data-id]").forEach(b => b.onclick = () => selectSample(b.dataset.id, b.textContent));
    $("#uploadButton").onclick = () => $("#videoInput").click();
    const first = health.samples.find(x => x.id === "test" && x.available) || health.samples.find(x => x.available);
    if (first) selectSample(first.id, first.name);
  } catch (error) { $("#systemText").textContent = "服务连接失败"; }
}

async function loadEvaluation() {
  try {
    const data = await api("/api/evaluation");
    $("#metricAccuracy").textContent = `${data.accuracy}%`;
    $("#metricCorrect").textContent = `${data.correct}/${data.questions}`;
    $("#metricAnswered").textContent = `${Math.round(data.answered / Math.max(data.questions, 1) * 100)}%`;
    $("#taskBreakdown").innerHTML = data.tasks.map(task => `<div class="task-chip">${task.name} · ${task.correct}/${task.total} · ${task.accuracy}%</div>`).join("");
    $("#resultRows").innerHTML = data.rows.map(row => {
      const options = row.options.map(option => { const letter=option[0]; const classes=[letter===row.prediction?"predicted":"",letter===row.answer?"answer":""].filter(Boolean).join(" "); return `<li class="${classes}">${option}</li>`; }).join("");
      return `<tr><td><details><summary class="question-summary"><strong>视频 ${row.video_id} · ${row.question_id}</strong><span>${row.question}</span></summary><div class="question-detail"><ul class="option-list">${options}</ul><div class="raw-output">模型原始输出：${row.response || "（空）"}</div></div></details></td><td>${row.task_type}</td><td class="answer-cell ${row.correct ? "correct" : "wrong"}"><b>${row.prediction || "—"}</b><span>${row.prediction_text}</span></td><td class="answer-cell"><b>${row.answer}</b><span>${row.answer_text}</span></td><td class="${row.correct ? "status-good" : "status-bad"}">${row.correct ? "✓ 正确" : "× 错误"}</td></tr>`;
    }).join("");
  } catch (error) { $("#resultRows").innerHTML = `<tr><td colspan="5">${error.message}</td></tr>`; }
}

async function pollJob(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    state.job = job; setStage(job.stage, job.message);
    if (job.status === "completed") {
      clearInterval(state.poll); state.poll = null;
      $("#answer").textContent = job.answer; $("#answer").classList.add("ready");
      $("#runtime").textContent = `GPU ${job.gpu} · ${job.elapsed_seconds}s`;
      $("#runButton").disabled = false; $("#runButton span").textContent = "再次推理";
    } else if (job.status === "failed") {
      clearInterval(state.poll); state.poll = null;
      $("#errorText").textContent = job.error; $("#answer").textContent = "任务未完成";
      $("#runButton").disabled = false; $("#runButton span").textContent = "重试";
    }
  } catch (error) { $("#errorText").textContent = error.message; }
}

$("#inferForm").onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData(); form.append("prompt", $("#prompt").value); form.append("gpu", $("#gpu").value); form.append("max_new_tokens", $("#tokens").value); form.append("max_clips", $("#clips").value);
  if (state.file) form.append("video", state.file); else form.append("sample_id", state.sample);
  $("#runButton").disabled = true; $("#runButton span").textContent = "任务运行中"; $("#answer").textContent = "Dispider 正在观察视频…"; $("#answer").classList.remove("ready"); $("#errorText").textContent = ""; setStage("queued", "正在创建任务");
  try { const job = await api("/api/infer", { method: "POST", body: form }); state.job = job; state.poll = setInterval(() => pollJob(job.id), 1200); pollJob(job.id); }
  catch (error) { $("#errorText").textContent = error.message; $("#runButton").disabled = false; }
};

$("#videoInput").onchange = (event) => { const file = event.target.files[0]; if (!file) return; state.file = file; state.sample = ""; document.querySelectorAll(".sample").forEach(b => b.classList.remove("active")); showVideo(URL.createObjectURL(file), `${file.name} · ${(file.size/1048576).toFixed(1)} MB`); };
$("#dropZone").onclick = (event) => { if (event.target.tagName !== "VIDEO" && !state.file && !state.sample) $("#videoInput").click(); };
$("#dropZone").ondragover = (e) => { e.preventDefault(); };
$("#dropZone").ondrop = (e) => { e.preventDefault(); const file=e.dataTransfer.files[0]; if(file){ state.file=file; state.sample=""; showVideo(URL.createObjectURL(file), file.name); } };
document.querySelectorAll(".nav-link").forEach(button => button.onclick = () => { document.querySelectorAll(".nav-link").forEach(x => x.classList.toggle("active", x===button)); $("#studioView").classList.toggle("active", button.dataset.view==="studio"); $("#evaluationView").classList.toggle("active", button.dataset.view==="evaluation"); if(button.dataset.view==="evaluation") loadEvaluation(); });

loadHealth(); loadEvaluation();
