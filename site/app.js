(function () {
  if (typeof ort === "undefined") {
    $("dot").className = "dot off";
    $("stat").textContent = "Could not load the model runtime. Check your connection and reload.";
    return;
  }
  ort.env.wasm.numThreads = 1;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225], S = 224;
  let session = null;

  async function loadModel() {
    $("dot").className = "dot";
    $("stat").textContent = "Loading the model (about 51 MB, first visit only)";
    try {
      session = await ort.InferenceSession.create("model/damage.onnx", { executionProviders: ["wasm"] });
      modelReady = true;
      $("dot").className = "dot on";
      $("stat").textContent = "Model ready, runs in your browser";
    } catch (e) {
      $("dot").className = "dot off";
      $("stat").textContent = "Could not load the model: " + (e.message || e);
    }
    refresh();
  }

  async function toTensor(file) {
    let bmp;
    try { bmp = await createImageBitmap(file); }
    catch (e) { throw new Error("Your browser cannot read this image. Use a PNG or JPEG."); }
    const c = document.createElement("canvas");
    c.width = c.height = S;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, S, S);
    const d = ctx.getImageData(0, 0, S, S).data;
    const out = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) {
      for (let ch = 0; ch < 3; ch++) out[ch * S * S + i] = (d[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    }
    return new ort.Tensor("float32", out, [1, 3, S, S]);
  }

  $("go").onclick = async () => {
    const out = $("result");
    out.className = "show";
    out.textContent = "Analysing";
    $("go").disabled = true;
    try {
      const [a, b] = await Promise.all([toTensor(files.pre), toTensor(files.post)]);
      const res = await session.run({ pre: a, post: b });
      const logits = Array.from(res.logits.data);
      const m = Math.max(...logits);
      const ex = logits.map(v => Math.exp(v - m));
      const sum = ex.reduce((x, y) => x + y, 0);
      const names = ["no-damage", "minor-damage", "major-damage", "destroyed"];
      const probs = ex.map(v => v / sum);
      const best = probs.indexOf(Math.max(...probs));
      const cls = names[best];
      const bars = names.map((k, i) =>
        `<div class="bar"><span>${LABELS[k]}</span><div class="track"><div class="fill" data-w="${(probs[i] * 100).toFixed(1)}" style="background:${COLORS[k]}"></div></div><span>${(probs[i] * 100).toFixed(1)}%</span></div>`).join("");
      out.innerHTML = `<div class="verdict"><h3 style="color:${COLORS[cls]}">${LABELS[cls]}</h3><span>${(probs[best] * 100).toFixed(1)}% confidence</span></div><div class="bars">${bars}</div>`;
      requestAnimationFrame(() => out.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%"));
    } catch (e) {
      out.innerHTML = `<p class="err">${e.message || e}</p>`;
    }
    refresh();
  };

  loadModel();
})();