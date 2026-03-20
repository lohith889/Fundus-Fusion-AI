(() => {
  "use strict";

  const REQUIRED = 3;

  const LOADING_STEPS = [
    "Extracting retinal masks…",
    "Detecting SIFT features…",
    "Computing pairwise homographies…",
    "Chaining transforms to centre frame…",
    "Feather-blending images…",
    "Cropping final mosaic…",
  ];

  // ── DOM refs ──
  const dropzone        = document.getElementById("dropzone");
  const fileInput       = document.getElementById("file-input");
  const slotsGrid       = document.getElementById("slots-grid");
  const slotFill        = document.getElementById("slot-fill");
  const slotCount       = document.getElementById("slot-count");
  const reorderHint     = document.getElementById("reorder-hint");
  const btnClear        = document.getElementById("btn-clear");
  const btnStitch       = document.getElementById("btn-stitch");
  const loadingStep     = document.getElementById("loading-step");

  const sectionUpload   = document.getElementById("upload-section");
  const sectionLoader   = document.getElementById("loader-section");
  const sectionResult   = document.getElementById("result-section");
  const sectionViz      = document.getElementById("viz-section");
  const sectionError    = document.getElementById("error-section");

  const resultImg       = document.getElementById("result-img");
  const btnDownload     = document.getElementById("btn-download");
  const btnRestart      = document.getElementById("btn-restart");
  const btnRetry        = document.getElementById("btn-retry");
  const errorMsg        = document.getElementById("error-msg");

  const customKpGrid    = document.getElementById("custom-kp-grid");
  const customMatchGrid = document.getElementById("custom-match-grid");

  // ── Screening refs (custom upload) ──
  const btnScreen               = document.getElementById("btn-screen");
  const screenLoaderSection     = document.getElementById("screen-loader-section");
  const screenResultSection     = document.getElementById("screen-result-section");
  const screenResultImg         = document.getElementById("screen-result-img");
  const screenClass             = document.getElementById("screen-class");
  const screenConfidence        = document.getElementById("screen-confidence");
  const screenConfidenceFill    = document.getElementById("screen-confidence-fill");
  const screenProbs             = document.getElementById("screen-probs");
  const btnScreenDownload       = document.getElementById("btn-screen-download");
  const btnRestartFromScreen    = document.getElementById("btn-restart-from-screen");

  // ── Sample demo refs ──
  const btnSampleStitch         = document.getElementById("btn-sample-stitch");
  const sampleControls          = document.getElementById("sample-controls");
  const sampleLoader            = document.getElementById("sample-loader");
  const sampleLoadingStep       = document.getElementById("sample-loading-step");
  const sampleResult            = document.getElementById("sample-result");
  const btnSampleDownload       = document.getElementById("btn-sample-download");
  const btnSampleReset          = document.getElementById("btn-sample-reset");
  const sampleResultImg         = document.getElementById("sample-result-img");
  const sampleViz               = document.getElementById("sample-viz");
  const sampleKpGrid            = document.getElementById("sample-kp-grid");
  const sampleMatchGrid         = document.getElementById("sample-match-grid");

  // ── Sample screening refs ──
  const btnSampleScreen             = document.getElementById("btn-sample-screen");
  const sampleScreenLoader          = document.getElementById("sample-screen-loader");
  const sampleScreenResult          = document.getElementById("sample-screen-result");
  const sampleScreenImg             = document.getElementById("sample-screen-img");
  const sampleScreenClass           = document.getElementById("sample-screen-class");
  const sampleScreenConfidence      = document.getElementById("sample-screen-confidence");
  const sampleScreenConfidenceFill  = document.getElementById("sample-screen-confidence-fill");
  const sampleScreenProbs           = document.getElementById("sample-screen-probs");
  const btnSampleScreenDownload     = document.getElementById("btn-sample-screen-download");

  // Engine toggle
  const toggleBtns = document.querySelectorAll(".toggle-btn");
  let engineMode = "retinal";

  // ── State ──
  let images = Array(REQUIRED).fill(null);
  let lastStitchedB64 = null;       // stores the stitched result for screening
  let sampleStitchedB64 = null;     // stores the sample stitched result

  // ── Helpers ──
  function show(el, display = "block") {
    if (!el) return;
    el.style.display = display;
    el.classList.remove("fade-in");
    void el.offsetWidth;
    el.classList.add("fade-in");
  }

  function hide(el) {
    if (!el) return;
    el.style.display = "none";
    el.classList.remove("fade-in");
  }

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = e => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function imgUrlToBase64(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        c.getContext("2d").drawImage(img, 0, 0);
        resolve(c.toDataURL("image/png"));
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  function renderVizResults(data, kpGrid, matchGrid) {
    kpGrid.innerHTML = "";
    matchGrid.innerHTML = "";
    (data.keypoints || []).forEach(src => {
      const d = document.createElement("div");
      d.className = "viz-item";
      d.innerHTML = `<img src="${src}" alt="Keypoints" loading="lazy"/>`;
      kpGrid.appendChild(d);
    });
    (data.matches || []).forEach(src => {
      const d = document.createElement("div");
      d.className = "viz-match-item";
      d.innerHTML = `<img src="${src}" alt="Matches" loading="lazy"/>`;
      matchGrid.appendChild(d);
    });
  }

  // ── Screening result renderer ──
  function renderScreeningResult(data, els) {
    const pred = data.prediction;
    const isGlaucoma = pred.class === "Glaucoma";

    els.img.src = data.annotated_image;
    els.cls.textContent = pred.class;
    els.cls.className = "screen-class " + (isGlaucoma ? "screen-class-danger" : "screen-class-safe");
    els.confidence.textContent = pred.confidence + "%";
    els.confidenceFill.style.width = pred.confidence + "%";
    els.confidenceFill.className = "screen-confidence-fill " +
      (isGlaucoma ? "screen-confidence-danger" : "screen-confidence-safe");

    // Probability breakdown
    els.probs.innerHTML = "";
    for (const [name, pct] of Object.entries(pred.probs)) {
      const row = document.createElement("div");
      row.className = "screen-prob-row";
      const isThis = name === pred.class;
      row.innerHTML = `
        <span class="screen-prob-name">${name}</span>
        <div class="screen-prob-bar-track">
          <div class="screen-prob-bar-fill ${name === 'Glaucoma' ? 'prob-danger' : 'prob-safe'}"
               style="width:${pct}%"></div>
        </div>
        <span class="screen-prob-value ${isThis ? 'active' : ''}">${pct}%</span>
      `;
      els.probs.appendChild(row);
    }

    if (els.download) {
      els.download.href = data.annotated_image;
    }
  }

  // ── Loading steps animation ──
  let _stepTimer = null;
  function startLoadingSteps(labelEl) {
    if (!labelEl) return;
    let idx = 0;
    labelEl.textContent = LOADING_STEPS[0];
    _stepTimer = setInterval(() => {
      idx = (idx + 1) % LOADING_STEPS.length;
      labelEl.textContent = LOADING_STEPS[idx];
    }, 1800);
  }

  function stopLoadingSteps() {
    clearInterval(_stepTimer);
    _stepTimer = null;
  }

  // ── Slots ──
  function buildSlots() {
    if (!slotsGrid) return;
    slotsGrid.innerHTML = "";
    images.forEach((img, i) => {
      const slot = document.createElement("div");
      slot.className = img ? "img-slot filled" : "img-slot";
      slot.dataset.idx = i;

      if (img) {
        slot.innerHTML = `
          <img class="slot-img" src="${img.dataUrl}" alt="Image ${i + 1}" draggable="false"/>
          <span class="slot-order">img ${i + 1}</span>
          <button class="slot-remove" data-idx="${i}" title="Remove">✕</button>
        `;
        slot.draggable = true;
        slot.addEventListener("dragstart", onDragStart);
        slot.addEventListener("dragover", onDragOver);
        slot.addEventListener("dragleave", onDragLeave);
        slot.addEventListener("drop", onDrop);
        slot.addEventListener("dragend", onDragEnd);
      } else {
        slot.innerHTML = `
          <span class="slot-empty-icon">+</span>
          <span class="slot-empty-num">slot ${i + 1}</span>
        `;
      }

      slotsGrid.appendChild(slot);
    });

    slotsGrid.querySelectorAll(".slot-remove").forEach(btn => {
      btn.addEventListener("click", e => {
        e.stopPropagation();
        images[parseInt(btn.dataset.idx)] = null;
        syncUI();
      });
    });

    updateProgress();
  }

  function updateProgress() {
    const filled = images.filter(Boolean).length;
    if (slotFill) slotFill.style.width = `${(filled / REQUIRED) * 100}%`;
    if (slotCount) slotCount.textContent = `${filled} / ${REQUIRED} images selected`;
    if (btnStitch) btnStitch.disabled = filled !== REQUIRED;
    if (btnClear) btnClear.disabled = filled === 0;
    if (reorderHint) reorderHint.style.display = filled > 0 ? "block" : "none";
  }

  function syncUI() { buildSlots(); }

  // ── Drag-to-reorder ──
  let dragSrcIdx = null;
  function onDragStart(e) {
    dragSrcIdx = parseInt(e.currentTarget.dataset.idx);
    e.currentTarget.classList.add("drag-src");
    e.dataTransfer.effectAllowed = "move";
  }
  function onDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    e.currentTarget.classList.add("drag-over");
  }
  function onDragLeave(e) { e.currentTarget.classList.remove("drag-over"); }
  function onDrop(e) {
    e.preventDefault();
    const targetIdx = parseInt(e.currentTarget.dataset.idx);
    e.currentTarget.classList.remove("drag-over");
    if (dragSrcIdx === null || dragSrcIdx === targetIdx) return;
    [images[dragSrcIdx], images[targetIdx]] = [images[targetIdx], images[dragSrcIdx]];
    syncUI();
  }
  function onDragEnd(e) {
    e.currentTarget.classList.remove("drag-src");
    dragSrcIdx = null;
  }

  // ── File ingestion ──
  async function ingestFiles(files) {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
    if (!imageFiles.length) return;
    const emptySlots = images.map((v, i) => v === null ? i : -1).filter(i => i >= 0);
    const toFill = Math.min(imageFiles.length, emptySlots.length);
    for (let i = 0; i < toFill; i++) {
      const dataUrl = await fileToDataUrl(imageFiles[i]);
      images[emptySlots[i]] = { dataUrl, name: imageFiles[i].name };
    }
    syncUI();
  }

  // ── Drop zone events ──
  if (dropzone) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag-over"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", e => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
      ingestFiles(e.dataTransfer.files);
    });
  }
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      ingestFiles(fileInput.files);
      fileInput.value = "";
    });
  }

  // ── Engine toggle ──
  toggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      toggleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      engineMode = btn.dataset.mode;
    });
  });

  // ── Clear / Restart ──
  function resetCustom() {
    images = Array(REQUIRED).fill(null);
    lastStitchedB64 = null;
    syncUI();
    hide(sectionResult);
    hide(sectionViz);
    hide(sectionError);
    hide(screenLoaderSection);
    hide(screenResultSection);
  }

  if (btnClear) btnClear.addEventListener("click", resetCustom);
  if (btnRestart) btnRestart.addEventListener("click", () => { resetCustom(); show(sectionUpload); });
  if (btnRestartFromScreen) btnRestartFromScreen.addEventListener("click", () => { resetCustom(); show(sectionUpload); });
  if (btnRetry) btnRetry.addEventListener("click", () => { hide(sectionError); show(sectionUpload); });

  // ══════════════════════════════════════════════════════════════════════════
  // CUSTOM UPLOAD — STITCH
  // ══════════════════════════════════════════════════════════════════════════
  if (btnStitch) {
    btnStitch.addEventListener("click", async () => {
      const filled = images.filter(Boolean);
      if (filled.length !== REQUIRED) {
        alert(`Please fill all ${REQUIRED} image slots before stitching.`);
        return;
      }

      hide(sectionResult); hide(sectionViz); hide(sectionError);
      hide(screenLoaderSection); hide(screenResultSection);
      show(sectionLoader, "flex");
      startLoadingSteps(loadingStep);

      const endpoint = engineMode === "retinal" ? "/stitch-retinal" : "/stitch";
      const b64Images = images.map(img => img.dataUrl);

      try {
        const resp = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ images: b64Images }),
        });
        const data = await resp.json();
        stopLoadingSteps();
        hide(sectionLoader);

        if (!data.success) {
          errorMsg.textContent = data.error || "Unknown error.";
          show(sectionError);
          return;
        }

        lastStitchedB64 = data.image;
        resultImg.src = data.image;
        btnDownload.href = data.image;
        show(sectionResult);
        sectionResult.classList.add("success-glow");
        setTimeout(() => sectionResult.classList.remove("success-glow"), 1500);

        // Feature viz (non-blocking)
        try {
          const vizResp = await fetch("/visualize-features", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: b64Images }),
          });
          const vizData = await vizResp.json();
          if (vizData.success) {
            renderVizResults(vizData, customKpGrid, customMatchGrid);
            show(sectionViz);
          }
        } catch (_) {}

      } catch (err) {
        stopLoadingSteps();
        hide(sectionLoader);
        errorMsg.textContent = "Network error — is the server running?";
        show(sectionError);
        console.error(err);
      }
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CUSTOM UPLOAD — SCREEN FOR DISEASE
  // ══════════════════════════════════════════════════════════════════════════
  if (btnScreen) {
    btnScreen.addEventListener("click", async () => {
      if (!lastStitchedB64) {
        alert("No stitched image available. Please stitch first.");
        return;
      }

      hide(screenResultSection);
      show(screenLoaderSection, "flex");

      try {
        const resp = await fetch("/screen", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: lastStitchedB64 }),
        });
        const data = await resp.json();
        hide(screenLoaderSection);

        if (!data.success) {
          alert("Screening failed: " + (data.error || "Unknown error"));
          return;
        }

        renderScreeningResult(data, {
          img: screenResultImg,
          cls: screenClass,
          confidence: screenConfidence,
          confidenceFill: screenConfidenceFill,
          probs: screenProbs,
          download: btnScreenDownload,
        });
        show(screenResultSection);
        screenResultSection.classList.add("success-glow");
        setTimeout(() => screenResultSection.classList.remove("success-glow"), 1500);

      } catch (err) {
        hide(screenLoaderSection);
        alert("Network error — is the server running?");
        console.error(err);
      }
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SAMPLE DEMO — STITCH
  // ══════════════════════════════════════════════════════════════════════════
  const SAMPLE_URLS = [
    "/static/samples/retina_1.jpg",
    "/static/samples/retina_2.jpg",
    "/static/samples/retina_3.jpg",
  ];

  if (btnSampleStitch) {
    btnSampleStitch.addEventListener("click", async () => {
      hide(sampleControls);
      hide(sampleResult);
      hide(sampleViz);
      hide(sampleScreenLoader);
      hide(sampleScreenResult);
      show(sampleLoader, "flex");
      startLoadingSteps(sampleLoadingStep);

      try {
        const b64Images = await Promise.all(SAMPLE_URLS.map(imgUrlToBase64));

        const resp = await fetch("/stitch-retinal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ images: b64Images }),
        });
        const data = await resp.json();
        stopLoadingSteps();
        hide(sampleLoader);

        if (!data.success) {
          show(sampleControls, "flex");
          alert("Sample stitching failed: " + (data.error || "Unknown error"));
          return;
        }

        sampleStitchedB64 = data.image;
        sampleResultImg.src = data.image;
        btnSampleDownload.href = data.image;
        show(sampleResult);
        document.getElementById("sample-section").classList.add("success-glow");
        setTimeout(() => document.getElementById("sample-section").classList.remove("success-glow"), 1500);

        // Feature viz (non-blocking)
        try {
          const vizResp = await fetch("/visualize-features", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ images: b64Images }),
          });
          const vizData = await vizResp.json();
          if (vizData.success) {
            renderVizResults(vizData, sampleKpGrid, sampleMatchGrid);
            show(sampleViz);
          }
        } catch (_) {}

      } catch (err) {
        stopLoadingSteps();
        hide(sampleLoader);
        show(sampleControls, "flex");
        alert("Network error — is the server running?");
        console.error(err);
      }
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SAMPLE DEMO — SCREEN FOR DISEASE
  // ══════════════════════════════════════════════════════════════════════════
  if (btnSampleScreen) {
    btnSampleScreen.addEventListener("click", async () => {
      if (!sampleStitchedB64) {
        alert("No stitched sample available.");
        return;
      }

      hide(sampleScreenResult);
      show(sampleScreenLoader, "flex");

      try {
        const resp = await fetch("/screen", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: sampleStitchedB64 }),
        });
        const data = await resp.json();
        hide(sampleScreenLoader);

        if (!data.success) {
          alert("Screening failed: " + (data.error || "Unknown error"));
          return;
        }

        renderScreeningResult(data, {
          img: sampleScreenImg,
          cls: sampleScreenClass,
          confidence: sampleScreenConfidence,
          confidenceFill: sampleScreenConfidenceFill,
          probs: sampleScreenProbs,
          download: btnSampleScreenDownload,
        });
        show(sampleScreenResult);

      } catch (err) {
        hide(sampleScreenLoader);
        alert("Network error — is the server running?");
        console.error(err);
      }
    });
  }

  // ── Sample Reset ──
  if (btnSampleReset) {
    btnSampleReset.addEventListener("click", () => {
      hide(sampleResult);
      hide(sampleLoader);
      hide(sampleViz);
      hide(sampleScreenLoader);
      hide(sampleScreenResult);
      show(sampleControls, "flex");
      sampleResultImg.src = "";
      sampleKpGrid.innerHTML = "";
      sampleMatchGrid.innerHTML = "";
      sampleStitchedB64 = null;
      stopLoadingSteps();
    });
  }

  // ── Init ──
  buildSlots();
})();
