(() => {
  "use strict";

  // ── DOM refs ──
  const video        = document.getElementById("webcam");
  const overlay      = document.getElementById("video-overlay");
  const btnStart     = document.getElementById("btn-start");
  const btnCapture   = document.getElementById("btn-capture");
  const btnClear     = document.getElementById("btn-clear");
  const btnStitch    = document.getElementById("btn-stitch");
  const btnDownload  = document.getElementById("btn-download");
  const btnRestart   = document.getElementById("btn-restart");
  const btnRetry     = document.getElementById("btn-retry");
  const thumbsGrid   = document.getElementById("thumbs-grid");
  const frameCount   = document.getElementById("frame-count");
  const sectionThumbs  = document.getElementById("thumbs-section");
  const sectionLoader  = document.getElementById("loader-section");
  const sectionResult  = document.getElementById("result-section");
  const sectionError   = document.getElementById("error-section");
  const resultImg      = document.getElementById("result-img");
  const errorMsg       = document.getElementById("error-msg");

  // Sample demo refs
  const btnSampleStitch  = document.getElementById("btn-sample-stitch");
  const btnSampleDownload = document.getElementById("btn-sample-download");
  const btnSampleReset   = document.getElementById("btn-sample-reset");
  const sampleLoader     = document.getElementById("sample-loader");
  const sampleResult     = document.getElementById("sample-result");
  const sampleResultImg  = document.getElementById("sample-result-img");
  const sampleControls   = document.getElementById("sample-controls");

  // Feature visualization refs
  const sampleViz        = document.getElementById("sample-viz");
  const sampleKpGrid     = document.getElementById("sample-kp-grid");
  const sampleMatchGrid  = document.getElementById("sample-match-grid");
  const vizSection       = document.getElementById("viz-section");
  const customKpGrid     = document.getElementById("custom-kp-grid");
  const customMatchGrid  = document.getElementById("custom-match-grid");

  const MAX_FRAMES = 6;
  let frames = [];   // base64 strings
  let stream = null;

  // ── Helper: smooth show/hide with animation ──
  function smoothShow(el, displayType = "block") {
    el.style.display = displayType;
    el.classList.remove("fade-in");
    // Trigger reflow so animation replays
    void el.offsetWidth;
    el.classList.add("fade-in");
  }

  function smoothHide(el) {
    el.style.display = "none";
    el.classList.remove("fade-in");
  }

  // ── Helper: convert image URL to base64 data URL ──
  function imgUrlToBase64(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext("2d").drawImage(img, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  // ── Helper: render visualization results into target grids ──
  function renderVizResults(data, kpGrid, matchGrid) {
    kpGrid.innerHTML = "";
    matchGrid.innerHTML = "";

    if (data.keypoints) {
      data.keypoints.forEach((src) => {
        const div = document.createElement("div");
        div.className = "viz-item";
        div.innerHTML = `<img src="${src}" alt="Feature keypoints"/>`;
        kpGrid.appendChild(div);
      });
    }

    if (data.matches) {
      data.matches.forEach((src) => {
        const div = document.createElement("div");
        div.className = "viz-match-item";
        div.innerHTML = `<img src="${src}" alt="Feature matches"/>`;
        matchGrid.appendChild(div);
      });
    }
  }

  // ══════ Sample Demo ══════
  async function stitchSamples() {
    smoothHide(sampleControls);
    smoothHide(sampleResult);
    smoothHide(sampleViz);
    smoothShow(sampleLoader, "flex");

    try {
      // Convert sample images to base64
      const sampleUrls = [
        "/static/samples/retina_1.jpg",
        "/static/samples/retina_2.jpg",
        "/static/samples/retina_3.jpg"
      ];
      const b64Images = await Promise.all(sampleUrls.map(imgUrlToBase64));

      // Step 1: Stitch first — show result immediately
      const resp = await fetch("/stitch-retinal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: b64Images })
      });

      const data = await resp.json();
      smoothHide(sampleLoader);

      if (data.success) {
        sampleResultImg.src = data.image;
        btnSampleDownload.href = data.image;
        smoothShow(sampleResult);
        // Add success glow to the sample card
        const sampleCard = document.getElementById("sample-section");
        sampleCard.classList.add("success-glow");
        setTimeout(() => sampleCard.classList.remove("success-glow"), 1500);
      } else {
        smoothShow(sampleControls, "flex");
        alert("Sample stitching failed: " + (data.error || "Unknown error"));
        return;
      }

      // Step 2: Load feature visualization after result is shown
      const vizResp = await fetch("/visualize-features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: b64Images })
      });
      const vizData = await vizResp.json();

      if (vizData.success) {
        renderVizResults(vizData, sampleKpGrid, sampleMatchGrid);
        smoothShow(sampleViz);
      }
    } catch (err) {
      smoothHide(sampleLoader);
      smoothShow(sampleControls, "flex");
      alert("Network error – is the server running?");
      console.error(err);
    }
  }

  function resetSample() {
    smoothHide(sampleResult);
    smoothHide(sampleLoader);
    smoothHide(sampleViz);
    smoothShow(sampleControls, "flex");
    sampleResultImg.src = "";
    sampleKpGrid.innerHTML = "";
    sampleMatchGrid.innerHTML = "";
  }

  btnSampleStitch.addEventListener("click", stitchSamples);
  btnSampleReset.addEventListener("click", resetSample);

  // ── Webcam ──
  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
      });
      video.srcObject = stream;
      overlay.classList.add("hidden");
      btnCapture.disabled = false;
      btnStart.textContent = "Camera Running";
      btnStart.disabled = true;
    } catch (err) {
      overlay.querySelector("span").textContent = "Camera access denied.";
      console.error(err);
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  // ── Capture frame ──
  function captureFrame() {
    if (frames.length >= MAX_FRAMES) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    frames.push(dataUrl);

    // Flash effect
    video.classList.add("flash");
    setTimeout(() => video.classList.remove("flash"), 300);

    renderThumbnails();
  }

  // ── Thumbnails ──
  function renderThumbnails() {
    thumbsGrid.innerHTML = "";
    frames.forEach((src, i) => {
      const div = document.createElement("div");
      div.className = "thumb";
      div.innerHTML = `<img src="${src}" alt="Frame ${i + 1}"/>
        <button class="remove-btn" data-idx="${i}">&times;</button>`;
      thumbsGrid.appendChild(div);
    });

    frameCount.textContent = frames.length;

    if (frames.length) {
      smoothShow(sectionThumbs);
    } else {
      smoothHide(sectionThumbs);
    }

    btnStitch.disabled = frames.length < 2;
    btnClear.disabled = frames.length === 0;
    btnCapture.disabled = !stream || frames.length >= MAX_FRAMES;
  }

  thumbsGrid.addEventListener("click", e => {
    const btn = e.target.closest(".remove-btn");
    if (!btn) return;
    frames.splice(Number(btn.dataset.idx), 1);
    renderThumbnails();
  });

  // ── Stitch ──
  async function stitch() {
    showSection("loader");
    smoothHide(vizSection);

    try {
      // Step 1: Stitch first — show result immediately
      const resp = await fetch("/stitch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: frames })
      });

      const data = await resp.json();

      if (data.success) {
        resultImg.src = data.image;
        btnDownload.href = data.image;
        showSection("result");
        // Add success glow to result card
        sectionResult.classList.add("success-glow");
        setTimeout(() => sectionResult.classList.remove("success-glow"), 1500);
      } else {
        errorMsg.textContent = data.error || "Unknown error.";
        showSection("error");
        return;
      }

      // Step 2: Load feature visualization after result is shown
      const vizResp = await fetch("/visualize-features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: frames })
      });
      const vizData = await vizResp.json();

      if (vizData.success) {
        renderVizResults(vizData, customKpGrid, customMatchGrid);
        smoothShow(vizSection);
      }
    } catch (err) {
      errorMsg.textContent = "Network error – is the server running?";
      showSection("error");
      console.error(err);
    }
  }

  // ── Section visibility helpers ──
  function showSection(name) {
    if (name === "loader") {
      smoothShow(sectionLoader, "flex");
    } else {
      smoothHide(sectionLoader);
    }

    if (name === "result") {
      smoothShow(sectionResult);
    } else {
      smoothHide(sectionResult);
    }

    if (name === "error") {
      smoothShow(sectionError);
    } else {
      smoothHide(sectionError);
    }
  }

  function resetAll() {
    frames = [];
    renderThumbnails();
    showSection(null);
    smoothHide(vizSection);
    customKpGrid.innerHTML = "";
    customMatchGrid.innerHTML = "";
    resultImg.src = "";
    btnDownload.href = "#";
  }

  // ── Event listeners ──
  btnStart.addEventListener("click", startCamera);
  btnCapture.addEventListener("click", captureFrame);
  btnClear.addEventListener("click", resetAll);
  btnStitch.addEventListener("click", stitch);
  btnRestart.addEventListener("click", resetAll);
  btnRetry.addEventListener("click", () => showSection(null));
})();
