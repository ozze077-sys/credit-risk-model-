/*
 * static_api.js - answers the dashboard's /api/... requests inside the browser.
 * ============================================================================
 * Used only by the GitHub Pages version (index.html at the repo root, built by
 * build_static_site.py). The page code in web/index.html is unchanged: it still calls
 * fetch('/api/...'), and this file intercepts those calls.
 *
 * * Answers that never change (exploration charts, model performance, data quality...)
 *   were computed by server.py at build time and are looked up directly.
 * * Answers that depend on the sidebar thresholds, filters, a customer ID or the
 *   new-customer form are computed here with the same formulas as server.py /
 *   prediction.py, from the embedded customer table and model coefficients.
 * * Saving files and retraining need Python, so they return a clear message.
 */
"use strict";

function createStaticApi(D) {
  const C = D.customers;            // columnar customer table
  const N = C.ID.length;
  const M = D.model;
  const L = D.labels;
  const SEGMENTS = ["Low risk", "Medium risk", "High risk"];
  const SEGMENT_COLORS = D.segment_colors;
  const UP_COLOR = D.colors.up, DOWN_COLOR = D.colors.down;

  class ApiError extends Error {
    constructor(message, status = 400) { super(message); this.status = status; }
  }

  // ---- Python-style number formatting -----------------------------------------------
  const clone = (o) => JSON.parse(JSON.stringify(o));
  /** round() with banker's rounding, like Python's round(). */
  const pyRound = (x) => {
    const f = Math.floor(x), diff = x - f;
    if (diff === 0.5) return f % 2 === 0 ? f : f + 1;
    return Math.round(x);
  };
  const grp = (n) => Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });      // f"{n:,}"
  const pct0 = (v) => (v == null || Number.isNaN(v) ? "n/a" : pyRound(v * 100) + "%");     // f"{v:.0%}"
  const money = (v) => (v == null || Number.isNaN(v) ? "n/a" : `NT$ ${grp(v)}`);
  const in100 = (p) => `${pyRound(p * 100)} out of 100`;
  /** f"{v:,}" for a Python float, e.g. 5.0 -> "5.0", 123456.5 -> "123,456.5". */
  const pyFloat = (v) => Number.isInteger(v) ? `${grp(v)}.0`
    : v.toLocaleString("en-US", { maximumFractionDigits: 17 });
  const signed0 = (x) => { const s = x.toFixed(0); return s.startsWith("-") ? s : "+" + s; };  // f"{x:+.0f}"

  // ---- Labels (data_loader.py) ------------------------------------------------------
  const friendly = (col) => {
    col = String(col);
    if (col.includes(" = ")) { const [c, v] = col.split(" = ", 2); return `${L.friendly[c] ?? c}: ${v}`; }
    return L.friendly[col] ?? col;
  };
  const payStatus = (code) => {
    if (code == null || Number.isNaN(code)) return "Missing";
    code = Math.trunc(code);
    if (code === -2) return "No bill that month";
    if (code === -1) return "Paid in full";
    if (code === 0) return "Paid the minimum";
    if (code >= 1) return `${code} month${code > 1 ? "s" : ""} late`;
    return `Unknown code ${code}`;
  };
  const categoryLabel = (col, code) => {
    if (code == null || Number.isNaN(code)) return "Missing";
    code = Math.trunc(code);
    if (D.pay_status_cols.includes(col)) return payStatus(code);
    const map = { SEX: L.sex, EDUCATION: L.education, MARRIAGE: L.marriage }[col] || {};
    return map[code] ?? `Undocumented code ${code}`;
  };
  const MONEY_COLS = new Set(["LIMIT_BAL", ...D.bill_cols, ...D.pay_amt_cols]);
  const displayValue = (f, v) => D.categorical_features.includes(f) ? categoryLabel(f, v)
    : MONEY_COLS.has(f) ? `NT$ ${grp(v)}` : grp(v);

  // ---- Thresholds and segments (prediction.py) --------------------------------------
  function thresholds(q) {
    const num = (key, dflt) => {
      if (!q.has(key)) return dflt;
      const v = Number(q.get(key));
      if (q.get(key).trim() === "" || Number.isNaN(v)) throw new ApiError(`'${key}' must be a number.`);
      return v;
    };
    const medium = num("medium", D.defaults.medium), high = num("high", D.defaults.high);
    if (!(0 < medium && medium < high && high < 100)) throw new ApiError("Thresholds must satisfy 0 < medium < high < 100.");
    return [medium / 100, high / 100];
  }
  const segmentOf = (p, medium, high) => (p >= high ? "High risk" : p >= medium ? "Medium risk" : "Low risk");
  const segmentsFor = (medium, high) => C.prob.map((p) => segmentOf(p, medium, high));

  function segmentSummary(seg, rows = null) {
    const acc = Object.fromEntries(SEGMENTS.map((s) => [s, { n: 0, p: 0, d: 0 }]));
    const idx = rows || [...Array(N).keys()];
    for (const i of idx) { const a = acc[seg[i]]; a.n++; a.p += C.prob[i]; a.d += C.actual[i]; }
    const total = Math.max(idx.length, 1);
    return SEGMENTS.map((s) => {
      const a = acc[s];
      return { Risk_Segment: s, Customers: a.n, Avg_Probability: a.n ? a.p / a.n : null, Share: a.n / total,
               Actual_Default_Rate: a.n ? a.d / a.n : null };
    });
  }

  // ---- Charts that change with the thresholds (charts.py) ----------------------------
  function segmentBar(seg, medium, high) {
    const fig = clone(D.templates.segment_bar);
    const t = fig.data[0];
    t.x = seg.map((r) => r.Risk_Segment);
    t.y = seg.map((r) => r.Customers);
    t.marker.color = seg.map((r) => SEGMENT_COLORS[r.Risk_Segment]);
    t.text = seg.map((r) => `${grp(r.Customers)} (${pct0(r.Share)})`);
    t.customdata = seg.map((r) => [r.Share, r.Avg_Probability]);
    fig.layout.title.text = fig.layout.title.text.replace(/Low below .*?(<\/span>)/,
      `Low below ${pct0(medium)} · Medium ${pct0(medium)}–${pct0(high)} · High ${pct0(high)} and above$1`);
    return fig;
  }
  function probabilityHistogram(medium, high) {
    const fig = clone(D.templates.probability);
    const lines = [[medium, "Medium"], [high, "High"]];
    lines.forEach(([t, label], i) => {
      fig.layout.shapes[i].x0 = fig.layout.shapes[i].x1 = t;
      fig.layout.annotations[i].x = t;
      fig.layout.annotations[i].text = `${label} from ${pct0(t)}`;
    });
    return fig;
  }
  function contributionBar(exp) {
    const fig = clone(D.templates.contribution);
    const e = exp.slice().reverse();
    const t = fig.data[0];
    t.x = e.map((r) => r.Contribution);
    t.y = e.map((r) => `${friendly(r.Feature)}: ${r.Value}`);
    t.marker.color = e.map((r) => (r.Contribution > 0 ? UP_COLOR : DOWN_COLOR));
    t.customdata = e.map((r) => (r.Contribution > 0 ? "Raises the risk" : "Lowers the risk"));
    return fig;
  }

  // ---- The model (logistic regression pipeline from model_training.py) ---------------
  /** Same steps as the scikit-learn pipeline: impute, standardise / one-hot, in the same column order. */
  function transform(values) {
    const out = [];
    M.num_features.forEach((f, j) => {
      const v = values[f] == null || Number.isNaN(values[f]) ? M.num_median[j] : values[f];
      out.push((v - M.num_mean[j]) / M.num_scale[j]);
    });
    M.cat_features.forEach((f, j) => {
      const v = values[f] == null || Number.isNaN(values[f]) ? M.cat_mode[j] : values[f];
      for (const cat of M.categories[j]) out.push(v === cat ? 1 : 0);  // unknown code -> all zeros
    });
    return out;
  }
  function predictProbability(values) {
    const x = transform(values);
    let z = M.intercept;
    for (let i = 0; i < x.length; i++) z += M.coef[i] * x[i];
    return 1 / (1 + Math.exp(-z));
  }
  /** contribution = coefficient x (value - average training value), summed per original input; top 6 by size. */
  function explain(values, topN = 6) {
    const x = transform(values);
    const per = new Map();
    x.forEach((v, i) => {
      const f = M.sources[i];
      per.set(f, (per.get(f) || 0) + M.coef[i] * (v - M.train_mean[i]));
    });
    return [...per.entries()]
      .map(([Feature, Contribution]) => ({ Feature, Contribution }))
      .sort((a, b) => Math.abs(b.Contribution) - Math.abs(a.Contribution))
      .slice(0, topN)
      .map((r) => ({ ...r, Value: displayValue(r.Feature, values[r.Feature]) }));
  }

  function predictionResult(values, proba, medium, high) {
    const exp = explain(values);
    const overall = D.overall;
    const reason = (r) => `${friendly(r.Feature)} (${r.Value})`;
    const up = exp.filter((r) => r.Contribution > 0).slice(0, 3).map(reason);
    const down = exp.filter((r) => r.Contribution < 0).slice(0, 3).map(reason);
    const ratio = proba / overall;
    let compare;
    if (ratio >= 1.15) compare = `That is about ${ratio.toFixed(1)} times the average customer (${in100(overall)}).`;
    else if (ratio <= 0.87) compare = `That is lower than the average customer (${in100(overall)}).`;
    else compare = `That is close to the average customer (${in100(overall)}).`;
    return {
      probability: proba,
      segment: segmentOf(proba, medium, high),
      segment_colors: SEGMENT_COLORS,
      overall_rate: overall,
      headline: `About ${in100(proba)} customers like this one missed their next payment.`,
      compare,
      summary: { direction: proba > overall ? "higher" : "lower", up, down },
      explanation_chart: contributionBar(exp),
    };
  }

  const rowValues = (i) => Object.fromEntries(D.feature_cols.map((f) => [f, C[f][i]]));
  const kpi = (label, value, note = "", kind = "hist") => ({ label, value, note, kind });

  // ---- Endpoints (same names and answers as server.py) -------------------------------
  function overview(q) {
    const [medium, high] = thresholds(q);
    const seg = segmentsFor(medium, high);
    const summary = segmentSummary(seg);
    const counts = Object.fromEntries(summary.map((r) => [r.Risk_Segment, r.Customers]));
    const d = clone(D.templates.overview);
    d.low_share = counts["Low risk"] / N;
    d.summary[1] = `The model sorted every customer into a risk group: ${grp(counts["Low risk"])} low, ` +
      `${grp(counts["Medium risk"])} medium and ${grp(counts["High risk"])} high risk.`;
    const k = d.kpis.find((x) => x.label === "Flagged as high risk");
    k.value = grp(counts["High risk"]);
    k.note = `${pct0(counts["High risk"] / N)} of customers`;
    d.charts.segments = segmentBar(summary, medium, high);
    d.charts.probability = probabilityHistogram(medium, high);
    return d;
  }

  function segmentation(q) {
    const [medium, high] = thresholds(q);
    const seg = segmentSummary(segmentsFor(medium, high));
    return {
      summary_text: seg.filter((r) => r.Customers).map((r) => `<b>${r.Risk_Segment}</b>: ${grp(r.Customers)} customers. ` +
        `The model gives them on average a ${pct0(r.Avg_Probability)} chance of missing a payment, and ` +
        `${pct0(r.Actual_Default_Rate)} of them really did.`)
        .concat(["When the two numbers are close, the estimates can be trusted."]),
      cards: seg.map((r) => kpi(r.Risk_Segment, grp(r.Customers),
        `${pct0(r.Share)} of customers · typical chance ` + (r.Avg_Probability == null ? "–" : pct0(r.Avg_Probability)),
        "pred")),
      histogram: probabilityHistogram(medium, high),
      summary: { columns: ["Risk group", "Customers", "Share", "Estimated chance (average)", "Actually missed"],
                 rows: seg.map((r) => [r.Risk_Segment, r.Customers, r.Share, r.Avg_Probability, r.Actual_Default_Rate]),
                 formats: { "Customers": "int", "Share": "pct", "Estimated chance (average)": "pct", "Actually missed": "pct" } },
      filters: D.templates.segmentation_filters,
    };
  }

  /** Rows matching the filters (missing filter = everything; empty = nothing), like filtered_predictions(). */
  function filtered(q) {
    const [medium, high] = thresholds(q);
    const seg = segmentsFor(medium, high);
    const list = (key, allowed, sep = ",") => {
      if (!q.has(key)) return new Set(allowed);
      const raw = q.get(key);
      return new Set(raw ? raw.split(sep).filter((v) => allowed.includes(v)) : []);
    };
    const segs = list("segment", SEGMENTS), ages = list("age", D.age_labels), limits = list("limit", D.limit_labels);
    const splits = list("split", ["Train", "Test"]), sources = list("source", D.sources, "|");
    const rows = [];
    for (let i = 0; i < N; i++) {
      if (segs.has(seg[i]) && ages.has(D.age_labels[C.age_group[i]]) && limits.has(D.limit_labels[C.limit_range[i]])
          && splits.has(C.split[i] ? "Test" : "Train") && sources.has(D.sources[C.source[i]])) rows.push(i);
    }
    return { rows, seg };
  }

  function segmentCustomers(q) {
    const { rows, seg } = filtered(q);
    let top = q.has("top") ? Number(q.get("top")) : 25;
    if (Number.isNaN(top)) throw new ApiError("'top' must be a number.");
    top = Math.trunc(Math.min(Math.max(top, 1), 500));
    // Highest probability first; ties keep the original order (pandas nlargest keep="first").
    const best = rows.slice().sort((a, b) => C.prob[b] - C.prob[a] || a - b).slice(0, top);
    return {
      count: rows.length, total: N,
      table: {
        columns: ["Customer ID", "Estimated chance", "Risk group", "Actually missed?", "Credit limit", "Age",
                  "Payment status – Sep", "Bill – Sep", "Amount paid – Sep", "Used for", "Data file"],
        rows: best.map((i) => [C.ID[i], C.prob[i], seg[i], C.actual[i] ? "Yes" : "No", C.LIMIT_BAL[i], C.AGE[i],
                               payStatus(C.PAY_0[i]), C.BILL_AMT1[i], C.PAY_AMT1[i], C.split[i] ? "Testing" : "Learning",
                               D.sources[C.source[i]]]),
        formats: { "Estimated chance": "pct", "Credit limit": "int", "Bill – Sep": "int", "Amount paid – Sep": "int" },
      },
    };
  }

  function customerDetails(i) {
    const items = [["Credit limit", money(C.LIMIT_BAL[i])], ["Age", C.AGE[i].toFixed(0)],
                   ["Gender", categoryLabel("SEX", C.SEX[i])], ["Education", categoryLabel("EDUCATION", C.EDUCATION[i])],
                   ["Marital status", categoryLabel("MARRIAGE", C.MARRIAGE[i])]];
    D.pay_status_cols.forEach((c) => items.push([`Payment status – ${L.pay_status_month[c]}`, payStatus(C[c][i])]));
    D.bill_cols.forEach((b, k) => items.push([`Bill / amount paid – ${L.months[k + 1]}`,
                                              `${money(C[b][i])} / ${money(C[D.pay_amt_cols[k]][i])}`]));
    return items;
  }

  const ID_INDEX = new Map(C.ID.map((id, i) => [id, i]));
  function customer(q) {
    const [medium, high] = thresholds(q);
    const raw = (q.get("id") || "").trim();
    if (!/^[+-]?\d+$/.test(raw)) throw new ApiError("Customer ID must be a whole number.");
    const cid = parseInt(raw, 10);
    const i = ID_INDEX.get(cid);
    if (i === undefined) throw new ApiError(`No customer with ID ${cid} was found.`, 404);
    const result = predictionResult(rowValues(i), C.prob[i], medium, high);
    return { ...result, id: cid, details: [["Source file", D.sources[C.source[i]]], ...customerDetails(i)],
             actual: L.target[C.actual[i]], split: C.split[i] ? "Test" : "Train" };
  }

  function validate(values) {
    const errors = [];
    for (const col of D.feature_cols) {
      const v = values[col];
      if (v == null || Number.isNaN(v)) { errors.push(`${col} is required.`); continue; }
      if (col in D.input_rules) {
        const [lo, hi] = D.input_rules[col];
        if (!(lo <= v && v <= hi)) errors.push(`${col} must be between ${grp(lo)} and ${grp(hi)} (got ${pyFloat(v)}).`);
      } else if (col in D.allowed_codes && !D.allowed_codes[col].includes(Math.trunc(v))) {
        errors.push(`${col} has an invalid code ${Number.isInteger(v) ? v.toFixed(1) : v}.`);
      }
    }
    return errors;
  }

  function predict(body) {
    const q = new URLSearchParams();
    for (const k of ["medium", "high"]) if (k in body) q.set(k, String(body[k]));
    const [medium, high] = thresholds(q);
    const raw = body.values || {};
    const values = {};
    let errors = [];
    for (const col of D.feature_cols) {
      const r = raw[col];
      if (r == null || r === "") { values[col] = null; continue; }
      const v = Number(String(r).trim());
      if (String(r).trim() === "" || Number.isNaN(v)) errors.push(`${col} must be a number.`);
      else values[col] = v;
    }
    if (!errors.length) errors = validate(values);
    if (errors.length) {
      errors = errors.map((e) => { const s = e.indexOf(" "); return s >= 0 ? `${friendly(e.slice(0, s))} ${e.slice(s + 1)}` : e; });
      throw new ApiError("Please fix the following: " + errors.join(" | "));
    }
    return predictionResult(values, predictProbability(values), medium, high);
  }

  function evaluate(t) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const i of TEST_ROWS) {
      const pred = C.prob[i] >= t, act = C.actual[i] === 1;
      if (pred && act) tp++; else if (pred) fp++; else if (act) fn++; else tn++;
    }
    const n = tp + fp + tn + fn;
    return { accuracy: (tp + tn) / n, precision: tp + fp ? tp / (tp + fp) : 0, recall: tp + fn ? tp / (tp + fn) : 0,
             f1: 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 0, confusion_matrix: { tn, fp, fn, tp } };
  }
  const TEST_ROWS = [...Array(N).keys()].filter((i) => C.split[i] === 1);

  function threshold(q) {
    const t = q.has("t") ? Number(q.get("t")) : 0.5;
    if (Number.isNaN(t)) throw new ApiError("'t' must be a number.");
    if (!(0 < t && t < 1)) throw new ApiError("t must be between 0 and 1.");
    const mt = evaluate(t), base = D.base_metrics;
    const same = Math.abs(t - D.decision_threshold) < 1e-9;
    const delta = (key) => (same ? "" : `${signed0((mt[key] - base[key]) * 100)} points vs the 50% level`);
    const cards = [
      kpi("Missed payments caught", pct0(mt.recall), delta("recall") || "recall", "plain"),
      kpi("Warnings that were right", pct0(mt.precision), delta("precision") || "precision", "plain"),
      kpi("Overall correct calls", pct0(mt.accuracy), delta("accuracy") || "accuracy", "plain"),
      kpi("Balance score", pct0(mt.f1), delta("f1") || "F1 score", "plain"),
    ];
    return { cards, confusion: mt.confusion_matrix, threshold: t };
  }

  // ---- CSV downloads -----------------------------------------------------------------
  const csvCell = (v) => { const s = String(v ?? ""); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const probCsv = (p) => { const r = Number(p.toFixed(4)); return Number.isInteger(r) ? r.toFixed(1) : String(r); };
  function predictionsCsv(rows, seg, extraCorrect) {
    const cols = D.export_columns.concat(extraCorrect ? ["Correct_Prediction"] : []);
    const lines = [cols.join(",")];
    for (const i of rows) {
      const predicted = C.prob[i] >= D.decision_threshold ? 1 : 0;
      const cell = {
        ID: C.ID[i], Source_File: D.sources[C.source[i]], Data_Split: C.split[i] ? "Test" : "Train",
        Actual_Default: C.actual[i], Predicted_Default_Probability: probCsv(C.prob[i]), Predicted_Default: predicted,
        Risk_Segment: seg[i], Correct_Prediction: predicted === C.actual[i] ? 1 : 0,
      };
      lines.push(cols.map((c) => csvCell(c in cell ? cell[c] : C[c][i])).join(","));
    }
    return lines.join("\n") + "\n";
  }
  function download(path, query) {
    const q = new URLSearchParams(query || "");
    if (path === "/api/segmentation/export.csv") {
      const { rows, seg } = filtered(q);
      return { name: "customer_predictions_filtered.csv", type: "text/csv", body: predictionsCsv(rows, seg, false) };
    }
    if (path === "/api/download/test_predictions.csv") {
      const [medium, high] = thresholds(q);
      return { name: "test_predictions.csv", type: "text/csv", body: predictionsCsv(TEST_ROWS, segmentsFor(medium, high), true) };
    }
    if (path === "/api/download/metrics.json") return { name: "model_metrics.json", type: "application/json", body: D.downloads.metrics_json };
    if (path === "/api/download/data_quality.csv") return { name: "data_quality_summary.csv", type: "text/csv", body: D.downloads.quality_csv };
    throw new ApiError("Unknown download.", 404);
  }

  const READ_ONLY = "Not available in the online version, which cannot change files. " +
    "Run the dashboard on your own computer (python server.py) to save results or teach the model again.";
  const GET = {
    "/api/overview": overview, "/api/segmentation": segmentation, "/api/segmentation/customers": segmentCustomers,
    "/api/customer": customer, "/api/performance/threshold": threshold,
  };
  const POST = { "/api/predict": predict, "/api/segmentation/save": () => { throw new ApiError(READ_ONLY); },
                 "/api/retrain": () => { throw new ApiError(READ_ONLY); } };

  /** Answer one request: returns {status, data}. Errors become {"error": "..."} like server.py. */
  function handle(path, query, method = "GET", body = null) {
    try {
      const q = new URLSearchParams(query || "");
      if (method === "POST") {
        if (!POST[path]) throw new ApiError("Unknown endpoint.", 404);
        return { status: 200, data: POST[path](body || {}) };
      }
      if (GET[path]) return { status: 200, data: GET[path](q) };
      const fixed = D.responses[path] || D.responses[`${path}?${q.toString()}`];
      if (fixed) return { status: 200, data: fixed };
      if (path === "/api/exploration/bill-payment") throw new ApiError("stat must be Mean or Median.");
      if (path === "/api/exploration/distribution") throw new ApiError(`Unknown variable '${q.get("var")}'.`);
      throw new ApiError("Not found.", 404);
    } catch (e) {
      if (e instanceof ApiError) return { status: e.status, data: { error: e.message } };
      return { status: 500, data: { error: `Unexpected error: ${e.message}` } };
    }
  }

  return { handle, download, predictProbability, explain, ApiError };
}

// ---- Wire it into the page: replace fetch('/api/...') and the download links ------------
if (typeof window !== "undefined") {
  const D = JSON.parse(document.getElementById("static-data").textContent);
  const API = createStaticApi(D);
  window.STATIC_API = API;

  // Hide the Plotly toolbar (zoom/pan icons) on every chart.
  if (window.Plotly) {
    const react = window.Plotly.react.bind(window.Plotly);
    window.Plotly.react = (el, data, layout, config = {}) => react(el, data, layout, { ...config, displayModeBar: false });
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, options = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, options);
    const method = (options.method || "GET").toUpperCase();
    const body = options.body ? JSON.parse(options.body) : null;
    const { status, data } = API.handle(url.pathname, url.search.slice(1), method, body);
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  };

  // Download buttons point at /api/... links; build the file here instead.
  document.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a) return;
    const url = new URL(a.getAttribute("href"), location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return;
    e.preventDefault();
    try {
      const file = API.download(url.pathname, url.search.slice(1));
      const blobUrl = URL.createObjectURL(new Blob([file.body], { type: file.type }));
      const link = Object.assign(document.createElement("a"), { href: blobUrl, download: file.name });
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (err) {
      alert(err.message);
    }
  }, true);
}

if (typeof module !== "undefined") module.exports = { createStaticApi };
