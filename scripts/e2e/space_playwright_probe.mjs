#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { chromium } from "playwright";

const DEFAULTS = {
  url: "http://127.0.0.1:7860/",
  artifactDir: "/tmp/space-e2e-artifacts",
  navigationTimeoutMs: 30000,
  settleTimeoutMs: 10000,
  contextTimeoutMs: 12000,
  preloadRows: 30,
  fetchRows: 15,
};

const toInt = (value, fallback) => {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const config = {
  url: process.env.SPACE_E2E_URL || DEFAULTS.url,
  artifactDir: process.env.SPACE_E2E_ARTIFACT_DIR || DEFAULTS.artifactDir,
  navigationTimeoutMs: toInt(process.env.SPACE_E2E_NAV_TIMEOUT_MS, DEFAULTS.navigationTimeoutMs),
  settleTimeoutMs: toInt(process.env.SPACE_E2E_SETTLE_TIMEOUT_MS, DEFAULTS.settleTimeoutMs),
  contextTimeoutMs: toInt(process.env.SPACE_E2E_CONTEXT_TIMEOUT_MS, DEFAULTS.contextTimeoutMs),
  preloadRows: toInt(process.env.SPACE_E2E_PRELOAD_ROWS, DEFAULTS.preloadRows),
  fetchRows: toInt(process.env.SPACE_E2E_FETCH_ROWS, DEFAULTS.fetchRows),
};

const startEpochMs = Date.now();
const failures = [];
const warnings = [];
const consoleEvents = [];
const networkEvents = [];

const nowOffsetMs = () => Date.now() - startEpochMs;

const isRelevantUrl = (url) =>
  url.includes("/gradio_api/") || url.includes("/data?") || url.includes("/queue/");

const isHeartbeatUrl = (url) => url.includes("/heartbeat/");

const normalizeUrl = (raw) => {
  try {
    const parsed = new URL(raw);
    return parsed.pathname + parsed.search;
  } catch {
    return raw;
  }
};

const recordFailure = (name, details) => {
  failures.push({ name, details });
};

const ensure = (condition, name, details) => {
  if (!condition) {
    recordFailure(name, details);
    return false;
  }
  return true;
};

const inspectUiState = async (page) =>
  page.evaluate(() => {
    const app = document.querySelector("gradio-app");
    const root = app && app.shadowRoot ? app.shadowRoot : document;

    const panel = root.querySelector("#space-eval-records-panel");
    const table = panel ? panel.querySelector("table") : null;
    const rows = table
      ? Array.from(table.querySelectorAll("tbody tr")).map((row) =>
          Array.from(row.querySelectorAll("td")).map((cell) => (cell.textContent || "").trim()),
        )
      : [];

    const contextButtons = panel
      ? Array.from(panel.querySelectorAll("button.space-context-open")).map((btn) => (btn.textContent || "").trim())
      : [];

    const nextPageButton = root.querySelector("#space-load-next-page-btn button, button#space-load-next-page-btn");
    const wrongOnlyInput = root.querySelector("#space-wrong-only-toggle input[type=checkbox]");
    const scoreCell = root.querySelector("td[data-clickable=1][data-cell-id]");
    const modal = root.querySelector("#space-context-modal");
    const modalBody = root.querySelector("#space-context-modal-body");

    return {
      scoreCellExists: Boolean(scoreCell),
      scoreCellId: scoreCell ? scoreCell.getAttribute("data-cell-id") : null,
      panelExists: Boolean(panel),
      panelHasLoading: Boolean(panel && panel.querySelector(".space-loading-progress")),
      panelHasTable: Boolean(table),
      panelHasEmptyState: Boolean(panel && panel.querySelector(".space-table-empty")),
      panelText: panel ? (panel.innerText || "").trim() : "",
      rowCount: rows.length,
      rows,
      contextButtonCount: contextButtons.length,
      contextPreviewTexts: contextButtons,
      nextPageExists: Boolean(nextPageButton),
      nextPageDisabled: nextPageButton ? Boolean(nextPageButton.disabled) : null,
      nextPageText: nextPageButton ? (nextPageButton.textContent || "").trim() : null,
      wrongOnlyExists: Boolean(wrongOnlyInput),
      wrongOnlyChecked: wrongOnlyInput ? Boolean(wrongOnlyInput.checked) : null,
      modalExists: Boolean(modal),
      modalOpen: Boolean(modal && modal.classList.contains("open")),
      modalBodyText: modalBody ? (modalBody.textContent || "") : "",
    };
  });

const waitForEvalPanelToSettle = async (page, timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  let latest = await inspectUiState(page);
  while (Date.now() < deadline) {
    latest = await inspectUiState(page);
    if (!latest.panelExists) {
      await page.waitForTimeout(150);
      continue;
    }

    const hasTerminalState =
      latest.panelHasTable ||
      latest.panelHasEmptyState ||
      latest.panelText.includes("读取 Eval 记录失败") ||
      latest.panelText.includes("加载下一页失败");

    if (hasTerminalState && !latest.panelHasLoading) {
      return latest;
    }

    await page.waitForTimeout(180);
  }
  return latest;
};

const countActionRequests = (startIndex) => {
  const events = networkEvents.slice(startIndex);
  return events.filter((event) => event.type === "request" && !isHeartbeatUrl(event.url)).length;
};

const sampleNetworkSummary = () => {
  const summary = {};
  for (const event of networkEvents) {
    if (event.type !== "request") {
      continue;
    }
    const key = `${event.method} ${normalizeUrl(event.url)}`;
    summary[key] = (summary[key] || 0) + 1;
  }
  return summary;
};

const validateSortOrder = (rows) => {
  let prevSample = -1;
  let prevRepeat = -1;

  for (const row of rows) {
    const sample = Number.parseInt(row[0] || "", 10);
    const repeat = Number.parseInt(row[1] || "", 10);
    if (!Number.isFinite(sample) || !Number.isFinite(repeat)) {
      return {
        ok: false,
        reason: "sample_index/repeat_index 无法解析为数字",
        row,
      };
    }

    if (sample < prevSample || (sample === prevSample && repeat < prevRepeat)) {
      return {
        ok: false,
        reason: "排序不符合 sample_index asc, repeat_index asc",
        row,
        previous: [prevSample, prevRepeat],
      };
    }

    prevSample = sample;
    prevRepeat = repeat;
  }

  return { ok: true };
};

const report = {
  started_at: new Date(startEpochMs).toISOString(),
  finished_at: null,
  config,
  steps: {},
  failures,
  warnings,
  console_events: consoleEvents,
  network_events: networkEvents,
  network_summary: {},
  artifacts: {},
};

await mkdir(config.artifactDir, { recursive: true });

let browser;
let context;
let page;

try {
  browser = await chromium.launch({ headless: true });
  context = await browser.newContext({ viewport: { width: 1680, height: 1200 } });
  await context.tracing.start({ screenshots: true, snapshots: true });
  page = await context.newPage();

  page.on("console", (msg) => {
    const type = msg.type();
    const text = msg.text();
    const event = {
      type,
      text,
      time_offset_ms: nowOffsetMs(),
    };
    consoleEvents.push(event);
    if (type === "error") {
      recordFailure("browser_console_error", event);
    }
    if (type === "warning" && text.includes("@import rules are not allowed here")) {
      warnings.push({ name: "css_import_warning", details: event });
    }
  });

  page.on("pageerror", (error) => {
    const event = {
      type: "pageerror",
      message: error.message,
      stack: error.stack,
      time_offset_ms: nowOffsetMs(),
    };
    consoleEvents.push(event);
    recordFailure("page_runtime_error", event);
  });

  page.on("request", (request) => {
    const url = request.url();
    if (!isRelevantUrl(url)) {
      return;
    }
    networkEvents.push({
      type: "request",
      url,
      method: request.method(),
      time_offset_ms: nowOffsetMs(),
    });
  });

  page.on("requestfailed", (request) => {
    const url = request.url();
    if (!isRelevantUrl(url)) {
      return;
    }
    const failure = request.failure();
    const event = {
      type: "request_failed",
      url,
      method: request.method(),
      error_text: failure ? failure.errorText : "unknown",
      time_offset_ms: nowOffsetMs(),
    };
    networkEvents.push(event);
    recordFailure("network_request_failed", event);
  });

  page.on("response", async (response) => {
    const url = response.url();
    if (!isRelevantUrl(url)) {
      return;
    }
    const status = response.status();
    const event = {
      type: "response",
      url,
      method: response.request().method(),
      status,
      time_offset_ms: nowOffsetMs(),
    };

    if (status >= 400) {
      try {
        event.body_preview = (await response.text()).slice(0, 400);
      } catch {
        event.body_preview = "<unavailable>";
      }
      recordFailure("network_response_error", event);
    }

    networkEvents.push(event);
  });

  await page.goto(config.url, {
    waitUntil: "domcontentloaded",
    timeout: config.navigationTimeoutMs,
  });
  await page.waitForTimeout(1200);

  const initialState = await inspectUiState(page);
  report.steps.initial_state = {
    score_cell_exists: initialState.scoreCellExists,
    panel_exists: initialState.panelExists,
    panel_preview: initialState.panelText.slice(0, 240),
  };

  ensure(initialState.scoreCellExists, "no_clickable_score_cell", initialState);

  const beforeClickRequestIndex = networkEvents.length;
  if (initialState.scoreCellExists) {
    await page.locator("td[data-clickable=1][data-cell-id]").first().click();
  }

  const afterClickState = await waitForEvalPanelToSettle(page, config.settleTimeoutMs);
  report.steps.after_click = {
    panel_has_table: afterClickState.panelHasTable,
    panel_has_loading: afterClickState.panelHasLoading,
    panel_has_empty_state: afterClickState.panelHasEmptyState,
    row_count: afterClickState.rowCount,
    panel_preview: afterClickState.panelText.slice(0, 260),
    action_request_count: countActionRequests(beforeClickRequestIndex),
  };

  ensure(
    !afterClickState.panelHasLoading,
    "eval_panel_stuck_loading_after_click",
    report.steps.after_click,
  );

  ensure(
    countActionRequests(beforeClickRequestIndex) > 0,
    "click_did_not_trigger_backend_request",
    report.steps.after_click,
  );

  if (afterClickState.panelHasTable) {
    ensure(
      afterClickState.rowCount <= config.preloadRows,
      "first_batch_row_count_exceeds_preload_limit",
      report.steps.after_click,
    );

    const sortCheck = validateSortOrder(afterClickState.rows);
    ensure(sortCheck.ok, "eval_rows_not_sorted", sortCheck);
  }

  if (afterClickState.contextButtonCount > 0) {
    const samplePreview = afterClickState.contextPreviewTexts[0] || "";
    ensure(
      samplePreview.length <= 23,
      "context_preview_too_long",
      {
        preview_length: samplePreview.length,
        preview: samplePreview,
      },
    );

    const beforeContextRequestIndex = networkEvents.length;
    await page.locator("#space-eval-records-panel button.space-context-open").first().click();

    const modalLoaded = await page
      .waitForFunction(() => {
        const app = document.querySelector("gradio-app");
        const root = app && app.shadowRoot ? app.shadowRoot : document;
        const modal = root.querySelector("#space-context-modal");
        const body = root.querySelector("#space-context-modal-body");
        if (!modal || !body || !modal.classList.contains("open")) {
          return false;
        }
        const text = (body.textContent || "").trim();
        return text.length > 0 && !text.includes("正在加载完整 context...");
      }, { timeout: config.contextTimeoutMs })
      .then(() => true)
      .catch(() => false);

    const afterContextState = await inspectUiState(page);
    report.steps.context_modal = {
      modal_open: afterContextState.modalOpen,
      modal_text_length: (afterContextState.modalBodyText || "").trim().length,
      action_request_count: countActionRequests(beforeContextRequestIndex),
    };

    ensure(modalLoaded, "context_modal_did_not_finish_loading", report.steps.context_modal);
    ensure(
      countActionRequests(beforeContextRequestIndex) > 0,
      "context_click_did_not_trigger_backend_request",
      report.steps.context_modal,
    );
  } else {
    warnings.push({ name: "context_buttons_missing", details: "首批记录中没有 context 按钮，跳过弹窗校验。" });
  }

  if (afterClickState.panelHasTable && afterClickState.nextPageExists && !afterClickState.nextPageDisabled) {
    const beforeNextPageRequestIndex = networkEvents.length;
    const beforeNextRowCount = afterClickState.rowCount;

    await page.locator("#space-load-next-page-btn button, button#space-load-next-page-btn").first().click();
    const afterNextPageState = await waitForEvalPanelToSettle(page, config.settleTimeoutMs);

    report.steps.next_page = {
      before_rows: beforeNextRowCount,
      after_rows: afterNextPageState.rowCount,
      button_disabled_after: afterNextPageState.nextPageDisabled,
      action_request_count: countActionRequests(beforeNextPageRequestIndex),
    };

    ensure(
      countActionRequests(beforeNextPageRequestIndex) > 0,
      "next_page_did_not_trigger_backend_request",
      report.steps.next_page,
    );

    ensure(
      afterNextPageState.rowCount >= beforeNextRowCount,
      "next_page_row_count_did_not_increase_or_stay_same",
      report.steps.next_page,
    );

    ensure(
      afterNextPageState.rowCount <= beforeNextRowCount + config.fetchRows,
      "next_page_loaded_more_than_one_page",
      report.steps.next_page,
    );
  } else {
    warnings.push({ name: "next_page_not_available", details: "下一页按钮不可用或没有首批数据，跳过翻页校验。" });
  }

  const wrongTogglePresent = (await inspectUiState(page)).wrongOnlyExists;
  if (wrongTogglePresent) {
    const beforeWrongToggleRequestIndex = networkEvents.length;
    await page.locator("#space-wrong-only-toggle input[type=checkbox]").first().check({ force: true });
    const afterWrongState = await waitForEvalPanelToSettle(page, config.settleTimeoutMs);

    report.steps.wrong_only = {
      row_count: afterWrongState.rowCount,
      panel_has_table: afterWrongState.panelHasTable,
      action_request_count: countActionRequests(beforeWrongToggleRequestIndex),
    };

    ensure(
      countActionRequests(beforeWrongToggleRequestIndex) > 0,
      "wrong_only_toggle_did_not_trigger_backend_request",
      report.steps.wrong_only,
    );

    if (afterWrongState.panelHasTable) {
      const invalidRows = afterWrongState.rows.filter((row) => (row[2] || "") !== "✕");
      ensure(
        invalidRows.length === 0,
        "wrong_only_result_contains_passed_rows",
        { invalid_row_count: invalidRows.length, sample_rows: invalidRows.slice(0, 3) },
      );
    }
  }

  report.artifacts.final_screenshot = path.join(config.artifactDir, "space-e2e-final.png");
  await page.screenshot({ path: report.artifacts.final_screenshot, fullPage: true });
  report.network_summary = sampleNetworkSummary();
} catch (error) {
  recordFailure("unexpected_exception", {
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : null,
  });
} finally {
  try {
    if (context) {
      report.artifacts.trace = path.join(config.artifactDir, "space-e2e-trace.zip");
      await context.tracing.stop({ path: report.artifacts.trace });
    }
  } catch {
    // Ignore trace stop failures.
  }

  if (browser) {
    await browser.close();
  }

  report.finished_at = new Date().toISOString();
  report.network_summary = report.network_summary || sampleNetworkSummary();

  const reportPath = path.join(config.artifactDir, "space-e2e-report.json");
  await writeFile(reportPath, JSON.stringify(report, null, 2), "utf-8");

  const headline = {
    status: failures.length > 0 ? "failed" : "passed",
    target_url: config.url,
    failures: failures.length,
    warnings: warnings.length,
    report_path: reportPath,
    artifact_dir: config.artifactDir,
  };

  console.log(JSON.stringify(headline, null, 2));
  if (failures.length > 0) {
    process.exitCode = 1;
  }
}
