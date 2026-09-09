// Hiver AI Support Agent — demo frontend. Vanilla JS, no build step.
(() => {
  "use strict";

  const API = ""; // same-origin

  const INTENT_COLORS = {
    order_delivery: "#2a78d6",
    returns_refund: "#4a3aa7",
    billing_payment: "#e34948",
    account_access: "#eda100",
    product_inquiry: "#1baf7a",
    complaint_escalation: "#e87ba4",
    compliment: "#008300",
    unclear_other: "#898781",
  };

  const MODE_LABEL = { trivial: "Trivial", simple: "Simple (ML)", llm: "LLM + RAG" };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function intentBadge(intent) {
    const color = INTENT_COLORS[intent] || "#898781";
    const label = (intent || "unknown").replace(/_/g, " ");
    return `<span class="intent-badge"><span class="intent-dot" style="background:${color}"></span>${escapeHtml(label)}</span>`;
  }

  // ---------------------------------------------------------------
  // Mobile nav
  // ---------------------------------------------------------------
  $("#navToggle")?.addEventListener("click", () => {
    const bar = $(".topbar");
    const open = bar.classList.toggle("nav-open");
    $("#navToggle").setAttribute("aria-expanded", String(open));
  });
  $$(".topnav a").forEach((a) => a.addEventListener("click", () => $(".topbar").classList.remove("nav-open")));

  // ---------------------------------------------------------------
  // Try it live
  // ---------------------------------------------------------------
  const messageInput = $("#messageInput");
  const charCount = $("#charCount");
  const analyzeBtn = $("#analyzeBtn");
  const resultsArea = $("#resultsArea");
  const errorBanner = $("#errorBanner");

  messageInput.addEventListener("input", () => {
    charCount.textContent = `${messageInput.value.length} / 500`;
  });

  async function loadExamples() {
    try {
      const res = await fetch(`${API}/api/examples`);
      const examples = await res.json();
      const wrap = $("#exampleChips");
      wrap.innerHTML = examples.map((ex, i) =>
        `<button class="chip" type="button" data-idx="${i}">${escapeHtml(ex.label)}</button>`
      ).join("");
      wrap.querySelectorAll(".chip").forEach((btn) => {
        btn.addEventListener("click", () => {
          messageInput.value = examples[Number(btn.dataset.idx)].text;
          messageInput.dispatchEvent(new Event("input"));
          messageInput.focus();
        });
      });
    } catch (e) {
      console.warn("Could not load examples", e);
    }
  }

  function activeModes() {
    return $$('#modeToggles input[type="checkbox"]').filter((c) => c.checked).map((c) => c.value);
  }

  function renderSkeletons(modes) {
    resultsArea.innerHTML = modes.map((mode) => `
      <div class="result-card" data-mode="${mode}">
        <div class="result-card-head">
          <span class="mode-badge ${mode}">${MODE_LABEL[mode]}</span>
        </div>
        <div class="result-card-body">
          <div class="skeleton" style="width:60%"></div>
          <div class="skeleton" style="width:40%"></div>
          <div class="skeleton" style="height:60px"></div>
        </div>
      </div>
    `).join("");
  }

  function renderResultCard(mode, data) {
    if (!data || data.error) {
      return `
        <div class="result-card card-error" data-mode="${mode}">
          <div class="result-card-head"><span class="mode-badge ${mode}">${MODE_LABEL[mode]}</span></div>
          <div class="result-card-body">Could not run this mode: ${escapeHtml(data?.error || "unknown error")}</div>
        </div>`;
    }

    const confPct = data.intent_confidence != null ? Math.round(data.intent_confidence * 100) : null;
    const escalate = !!data.escalate;
    const groundedItems = (data.retrieved_examples || []).slice(0, 3);

    return `
      <div class="result-card" data-mode="${mode}">
        <div class="result-card-head">
          <span class="mode-badge ${mode}">${MODE_LABEL[mode]}</span>
          ${data.latency_s ? `<span class="mode-name">${data.latency_s.toFixed(2)}s</span>` : ""}
        </div>
        <div class="result-card-body">
          <div class="intent-row">
            ${intentBadge(data.intent)}
            ${confPct != null ? `<span class="confidence-text">${confPct}% confidence</span>` : ""}
          </div>
          ${confPct != null ? `
            <div class="confidence-track"><div class="confidence-fill" style="width:${confPct}%"></div></div>
          ` : ""}

          <div class="status-pill ${escalate ? "escalate" : "auto"}">
            <span class="status-dot"></span>
            ${escalate ? "Escalate to human" : "Auto-handle"}
          </div>
          <div class="escalation-reason">${escapeHtml(data.escalation_reason || "")}</div>

          <div class="reply-bubble ${data.drafted_reply ? "" : "empty"}">
            ${data.drafted_reply ? escapeHtml(data.drafted_reply) : "(no reply drafted)"}
          </div>

          ${groundedItems.length ? `
            <button class="grounded-toggle" type="button">Grounded in ${groundedItems.length} similar past case${groundedItems.length > 1 ? "s" : ""} ▾</button>
            <div class="grounded-list" hidden>
              ${groundedItems.map((g) => `
                <div class="grounded-item">
                  <span class="grounded-sim">sim ${(g.similarity ?? 0).toFixed(2)}</span>
                  <b>Customer:</b> ${escapeHtml((g.customer_text || "").slice(0, 110))}<br/>
                  <b>Brand replied:</b> ${escapeHtml((g.brand_reply || "").slice(0, 110))}
                </div>
              `).join("")}
            </div>
          ` : ""}
        </div>
      </div>`;
  }

  async function runAnalysis() {
    const message = messageInput.value.trim();
    errorBanner.hidden = true;
    if (!message) {
      errorBanner.textContent = "Type a message first (or pick an example above).";
      errorBanner.hidden = false;
      return;
    }
    const modes = activeModes();
    renderSkeletons(modes);
    analyzeBtn.disabled = true;
    $(".btn-label", analyzeBtn).textContent = "Analyzing…";
    $(".btn-spinner", analyzeBtn).hidden = false;

    try {
      const res = await fetch(`${API}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, modes }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      resultsArea.innerHTML = modes.map((m) => renderResultCard(m, data[m])).join("");
      $$(".grounded-toggle", resultsArea).forEach((btn) => {
        btn.addEventListener("click", () => {
          const list = btn.nextElementSibling;
          const open = !list.hidden;
          list.hidden = open;
          btn.textContent = btn.textContent.replace(open ? "▴" : "▾", open ? "▾" : "▴");
        });
      });
    } catch (e) {
      errorBanner.textContent = `Something went wrong: ${e.message}`;
      errorBanner.hidden = false;
      resultsArea.innerHTML = "";
    } finally {
      analyzeBtn.disabled = false;
      $(".btn-label", analyzeBtn).textContent = "Analyze";
      $(".btn-spinner", analyzeBtn).hidden = true;
    }
  }

  analyzeBtn.addEventListener("click", runAnalysis);
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAnalysis();
  });

  // ---------------------------------------------------------------
  // Dashboard
  // ---------------------------------------------------------------
  const CHART_FONT = { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12 };
  Chart.defaults.font = CHART_FONT;
  Chart.defaults.color = "#52514e";
  Chart.defaults.borderColor = "#e1e0d9";

  function statTile(label, value, delta) {
    return `
      <div class="stat-tile">
        <div class="stat-label">${escapeHtml(label)}</div>
        <div class="stat-value">${escapeHtml(value)}</div>
        ${delta ? `<div class="stat-delta ${delta.dir}">${escapeHtml(delta.text)}</div>` : ""}
      </div>`;
  }

  function pct(x) { return `${Math.round(x * 100)}%`; }

  function renderConfusionMatrix(intentMetrics) {
    const labels = intentMetrics.labels;
    const shortLabels = labels.map((l) => l.replace(/_/g, " "));
    const cm = intentMetrics.confusion_matrix;
    const maxVal = Math.max(1, ...cm.flat());

    const cells = [`<div class="cm-corner"></div>`];
    shortLabels.forEach((l) => cells.push(`<div class="cm-collabel">${escapeHtml(l)}</div>`));

    cm.forEach((row, r) => {
      cells.push(`<div class="cm-rowlabel">${escapeHtml(shortLabels[r])}</div>`);
      row.forEach((v, c) => {
        const alpha = v === 0 ? 0 : 0.14 + 0.72 * (v / maxVal);
        const bg = v === 0 ? "transparent" : `rgba(42,120,214,${alpha.toFixed(2)})`;
        const title = `gold=${labels[r]} → predicted=${labels[c]}: ${v}`;
        cells.push(`<div class="cm-cell ${v === 0 ? "zero" : ""}" style="background:${bg}" title="${escapeHtml(title)}">${v}</div>`);
      });
    });

    $("#confusionMatrix").innerHTML = cells.join("");
  }

  async function loadDashboard() {
    let metrics;
    try {
      const res = await fetch(`${API}/api/metrics`);
      if (!res.ok) throw new Error("no metrics");
      metrics = await res.json();
    } catch {
      $("#metricsMissing").hidden = false;
      return;
    }

    const modes = ["trivial", "simple", "llm"].filter((m) => metrics[m]);
    const llm = metrics.llm, simple = metrics.simple, trivial = metrics.trivial;

    const tiles = [];
    if (llm) {
      tiles.push(statTile("Headline intent accuracy", pct(llm.intent.accuracy),
        simple ? { dir: llm.intent.accuracy >= simple.intent.accuracy ? "up" : "down",
                   text: `${llm.intent.accuracy >= simple.intent.accuracy ? "+" : ""}${Math.round((llm.intent.accuracy - simple.intent.accuracy) * 100)}pp vs simple` } : null));
      tiles.push(statTile("Headline escalation F1", llm.escalation.f1.toFixed(2),
        simple ? { dir: llm.escalation.f1 >= simple.escalation.f1 ? "up" : "down",
                   text: `${llm.escalation.f1 >= simple.escalation.f1 ? "+" : ""}${(llm.escalation.f1 - simple.escalation.f1).toFixed(2)} vs simple` } : null));
    }
    if (simple) tiles.push(statTile("Simple baseline intent accuracy", pct(simple.intent.accuracy)));
    if (trivial) tiles.push(statTile("Trivial baseline intent accuracy", pct(trivial.intent.accuracy)));
    $("#statGrid").innerHTML = tiles.join("");

    // comparison chart: intent macro-F1 vs escalation F1 per mode
    const seriesColor = { trivial: "#898781", simple: "#eb6834", llm: "#2a78d6" };
    new Chart($("#comparisonChart"), {
      type: "bar",
      data: {
        labels: modes.map((m) => MODE_LABEL[m]),
        datasets: [
          {
            label: "Intent macro-F1",
            data: modes.map((m) => metrics[m].intent.macro_f1),
            backgroundColor: modes.map((m) => seriesColor[m]),
            borderRadius: 4,
            barPercentage: 0.55,
          },
          {
            label: "Escalation F1",
            data: modes.map((m) => metrics[m].escalation.f1),
            backgroundColor: modes.map((m) => seriesColor[m] + "80"),
            borderRadius: 4,
            barPercentage: 0.55,
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { min: 0, max: 1, grid: { color: "#e1e0d9" } }, x: { grid: { display: false } } },
        plugins: { legend: { position: "bottom" } },
      },
    });

    // confusion matrix — plain HTML/CSS grid (see cm-grid in styles.css)
    const cmMode = llm ? "llm" : modes[modes.length - 1];
    $("#confusionTitle").textContent = `Intent confusion matrix — ${MODE_LABEL[cmMode]} system`;
    renderConfusionMatrix(metrics[cmMode].intent);

    // judge scores — grouped bar (not radar: radar distorts area/angle and
    // makes non-adjacent axes hard to compare, see dataviz skill guidance)
    try {
      const jres = await fetch(`${API}/api/judge-summary`);
      if (jres.ok) {
        const judge = await jres.json();
        const jModes = ["trivial", "simple", "llm"].filter((m) => judge[m]);
        const dims = ["grounding_score", "helpfulness_score", "tone_score", "clarity_score", "overall_score"];
        const dimLabels = ["Grounding", "Helpfulness", "Tone", "Clarity", "Overall"];
        new Chart($("#judgeChart"), {
          type: "bar",
          data: {
            labels: dimLabels,
            datasets: jModes.map((m) => ({
              label: MODE_LABEL[m] || m,
              data: dims.map((d) => judge[m][d]),
              backgroundColor: seriesColor[m] || "#2a78d6",
              borderRadius: 4,
              barPercentage: 0.8,
              categoryPercentage: 0.7,
            })),
          },
          options: {
            responsive: true,
            scales: { y: { min: 0, max: 5, ticks: { stepSize: 1 }, grid: { color: "#e1e0d9" } }, x: { grid: { display: false } } },
            plugins: { legend: { position: "bottom" } },
          },
        });
      } else {
        $("#judgeChart").parentElement.innerHTML = '<div class="chart-card-title">LLM-as-judge reply-quality scores (1–5)</div><div class="empty-state">Run scripts/09_llm_judge.py to populate this chart.</div>';
      }
    } catch { /* leave chart canvas empty */ }

    // judge/human agreement
    try {
      const ares = await fetch(`${API}/api/judge-agreement`);
      if (ares.ok) {
        const agreement = await ares.json();
        $("#agreementCard").hidden = false;
        $("#agreementBody").innerHTML = `
          <p class="section-sub" style="margin-bottom:12px">${escapeHtml(agreement.description || "")}</p>
          <div class="agreement-stats">
            <div class="agreement-stat"><div class="stat-label">n examples</div><div class="stat-value">${agreement.n}</div></div>
            <div class="agreement-stat"><div class="stat-label">Mean abs. diff (overall score)</div><div class="stat-value">${agreement.mean_abs_diff_overall.toFixed(2)}</div></div>
            <div class="agreement-stat"><div class="stat-label">Within ±1 point</div><div class="stat-value">${pct(agreement.within_one_point_rate)}</div></div>
            <div class="agreement-stat"><div class="stat-label">Spearman corr.</div><div class="stat-value">${agreement.spearman_corr.toFixed(2)}</div></div>
          </div>`;
      }
    } catch { /* optional section */ }
  }

  // ---------------------------------------------------------------
  // Failure analysis
  // ---------------------------------------------------------------
  let failureData = null;

  function renderFailureTab(tab) {
    const list = failureData ? failureData[tab] : [];
    const container = $("#failureList");
    if (!list || !list.length) {
      container.innerHTML = '<div class="empty-state">No examples to show yet — run the evaluation scripts first.</div>';
      return;
    }
    container.innerHTML = list.map((row) => `
      <div class="failure-card">
        <div class="failure-text">${escapeHtml(row.customer_text)}</div>
        <div class="failure-meta">
          ${tab === "wrong_intent"
            ? `<span><b>Predicted:</b> ${escapeHtml(row.pred_intent)}</span><span><b>Gold:</b> ${escapeHtml(row.gold_intent)}</span>`
            : `<span><b>Predicted:</b> ${row.pred_escalate ? "Escalate" : "Auto-handle"}</span><span><b>Gold:</b> ${row.gold_escalate ? "Escalate" : "Auto-handle"} — ${escapeHtml(row.gold_escalation_reason || "")}</span>`}
        </div>
      </div>
    `).join("");
  }

  async function loadFailures() {
    try {
      const res = await fetch(`${API}/api/failure-examples?mode=llm&n=8`);
      if (!res.ok) throw new Error("not found");
      failureData = await res.json();
    } catch {
      failureData = { wrong_intent: [], wrong_escalate: [] };
    }
    renderFailureTab("wrong_intent");
  }

  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      renderFailureTab(btn.dataset.tab);
    });
  });

  // ---------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------
  loadExamples();
  loadDashboard();
  loadFailures();
})();
