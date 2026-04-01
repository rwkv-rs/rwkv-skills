from __future__ import annotations

"""Gradio space to visualise evaluation scores."""

import json
from pathlib import Path
from typing import Any

import gradio as gr

from .constants import (
    AUTO_MODEL_LABEL,
    TABLE_VIEW_LABELS,
    DEFAULT_TABLE_VIEW,
    DOMAIN_GROUPS,
    EVAL_FETCH_ROWS,
    _normalize_table_view,
)
from .assets import _load_css, _load_vendor_css, _load_vendor_head
from .selection import _prepare_selection, _compute_choices, _initial_payload
from .tables import (
    _render_pivot_html,
    _render_summary,
    _build_domain_tables,
    _export_selection_csv,
)
from .charts import (
    _build_knowledge_bar,
    _build_aime_plot,
    _build_instruction_bar,
    _build_coding_bar,
    _load_coding_example,
)
from .eval_records import (
    _parse_click_payload,
    _build_context_event_payload,
    _render_eval_records_html,
    _start_eval_records_load,
    _continue_eval_records_load,
    _empty_eval_loader_state,
)
from .data import load_scores


def _build_dashboard() -> gr.Blocks:
    base_css, style_warning = _load_css()
    vendor_css, vendor_css_warning = _load_vendor_css()
    css = base_css + ("\n\n" + vendor_css if vendor_css else "")
    head, head_warning = _load_vendor_head()
    entries, selection, load_errors = _initial_payload()
    model_choices = _compute_choices(entries)
    warnings = (
        load_errors
        + ([style_warning] if style_warning else [])
        + ([vendor_css_warning] if vendor_css_warning else [])
        + ([head_warning] if head_warning else [])
    )
    initial_view_mode = DEFAULT_TABLE_VIEW

    summary = _render_summary(
        all_entries=entries,
        visible=selection.entries,
        selection=selection,
        view_mode=initial_view_mode,
        warnings=warnings,
    )
    domain_tables, interaction_meta = _build_domain_tables(
        selection,
        all_entries=entries,
        view_mode=initial_view_mode,
    )
    initial_eval_html = _render_eval_records_html(meta=None, records=[], only_wrong=False)
    initial_eval_rows: list[dict[str, Any]] = []
    initial_eval_loader = _empty_eval_loader_state()
    csv_export_path = _export_selection_csv(selection)

    modal_html = """
<div id="space-context-modal" class="space-modal" aria-hidden="true">
  <div class="space-modal-content">
    <div class="space-modal-topbar">
      <div class="space-modal-title">Context 详情</div>
      <button type="button" class="space-modal-close" data-close-modal="1">关闭</button>
    </div>
    <div class="space-context-layout">
      <div id="space-context-left" class="space-context-left"></div>
      <div id="space-context-right" class="space-context-right"></div>
    </div>
  </div>
</div>
"""

    js_func = """
    () => {
        const app = document.querySelector('gradio-app');
        const root = (app && app.shadowRoot) ? app.shadowRoot : document;
        if (app) {
            app.style.backgroundColor = 'var(--bg-primary)';
        }
        document.body.classList.add('dark');

        if (window.__rwkvSpaceHandlersBound) {
            return;
        }
        window.__rwkvSpaceHandlersBound = true;

        const escapeId = (id) => {
            if (window.CSS && typeof CSS.escape === 'function') {
                return CSS.escape(id);
            }
            return id.replace(/([ #;?%&,.+*~':"!^$\\[\\]()=>|/@])/g, '\\$1');
        };

        const queryById = (id) => root.querySelector(`#${escapeId(id)}`);

        const clearNode = (node) => {
            if (!node) {
                return;
            }
            while (node.firstChild) {
                node.removeChild(node.firstChild);
            }
        };

        const toDisplayString = (value) => {
            if (value === null || value === undefined) {
                return '';
            }
            if (typeof value === 'string') {
                return value;
            }
            if (typeof value === 'number' || typeof value === 'boolean') {
                return String(value);
            }
            try {
                return JSON.stringify(value);
            } catch (error) {
                return String(value);
            }
        };

        const ensureMarkdown = () => {
            if (window.__rwkvSpaceMarkdown) {
                return window.__rwkvSpaceMarkdown;
            }
            if (typeof window.markdownit !== 'function') {
                window.__rwkvSpaceMarkdown = null;
                return null;
            }
            const md = window.markdownit({
                html: false,
                linkify: true,
                breaks: true,
                typographer: true,
            });
            window.__rwkvSpaceMarkdown = md;
            return md;
        };

        const applyHighlight = (container) => {
            if (!container || !window.hljs) {
                return;
            }
            container.querySelectorAll('pre code').forEach((block) => {
                try {
                    window.hljs.highlightElement(block);
                } catch (error) {
                    // ignore highlight failures
                }
            });
        };

        const applyMath = (container) => {
            if (!container || typeof window.renderMathInElement !== 'function') {
                return;
            }
            try {
                window.renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\\\(', right: '\\\\)', display: false },
                        { left: '\\\\[', right: '\\\\]', display: true },
                    ],
                    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                });
            } catch (error) {
                // ignore math failures
            }
        };

        const renderMarkdown = (container, text) => {
            if (!container) {
                return;
            }
            const md = ensureMarkdown();
            if (!md) {
                container.textContent = text || '';
                return;
            }
            container.innerHTML = md.render(text || '');
            applyHighlight(container);
            applyMath(container);
        };

        const closeModal = () => {
            const modal = queryById('space-context-modal');
            if (!modal) {
                return;
            }
            modal.classList.remove('open');
            modal.setAttribute('aria-hidden', 'true');
        };

        const openModalEvent = (evt) => {
            const modal = queryById('space-context-modal');
            const left = queryById('space-context-left');
            const right = queryById('space-context-right');
            if (!modal || !left || !right) {
                return;
            }
            clearNode(left);
            clearNode(right);

            const eventObj = evt && typeof evt === 'object' ? evt : null;
            const view = eventObj && typeof eventObj.view === 'string' ? eventObj.view : 'text';
            const rawText = eventObj && typeof eventObj.raw_text === 'string' ? eventObj.raw_text : '';

            const addCard = (parent, title, bodyBuilder) => {
                const card = document.createElement('div');
                card.className = 'space-context-card';

                const header = document.createElement('div');
                header.className = 'space-context-card-header';

                const h = document.createElement('div');
                h.className = 'space-context-card-title';
                h.textContent = title || '';
                header.appendChild(h);
                card.appendChild(header);

                if (typeof bodyBuilder === 'function') {
                    bodyBuilder(card);
                }
                parent.appendChild(card);
            };

            const addSectionPre = (parent, label, text) => {
                const labelEl = document.createElement('div');
                labelEl.className = 'space-context-label';
                labelEl.textContent = label || '';
                parent.appendChild(labelEl);

                const pre = document.createElement('pre');
                pre.className = 'space-context-pre';
                pre.textContent = text || '';
                parent.appendChild(pre);
            };

            const addSectionMarkdown = (parent, label, text) => {
                const labelEl = document.createElement('div');
                labelEl.className = 'space-context-label';
                labelEl.textContent = label || '';
                parent.appendChild(labelEl);

                const box = document.createElement('div');
                box.className = 'space-context-md';
                renderMarkdown(box, text || '');
                parent.appendChild(box);
            };

            const renderRaw = (text) => {
                addCard(left, 'Raw Context', (card) => {
                    addSectionPre(card, 'raw_text', text || '');
                });
            };

            const renderSampling = (samplingConfig, stopTokens) => {
                if (!samplingConfig || typeof samplingConfig !== 'object') {
                    addCard(right, 'sampling_config', (card) => {
                        addSectionPre(card, 'sampling_config', '—');
                    });
                    return;
                }

                const stageNames = Object.keys(samplingConfig).sort();
                stageNames.forEach((stageName) => {
                    const cfg = samplingConfig[stageName];
                    if (!cfg || typeof cfg !== 'object') {
                        return;
                    }
                    addCard(right, `sampling_config · ${stageName}`, (card) => {
                        const kv = document.createElement('div');
                        kv.className = 'space-kv-list';

                        Object.keys(cfg)
                            .sort()
                            .forEach((key) => {
                                if (key === 'stop_tokens') {
                                    return;
                                }
                                const row = document.createElement('div');
                                row.className = 'space-kv';
                                const k = document.createElement('div');
                                k.className = 'space-kv-key';
                                k.textContent = key;
                                const v = document.createElement('div');
                                v.className = 'space-kv-value';
                                v.textContent = toDisplayString(cfg[key]);
                                row.appendChild(k);
                                row.appendChild(v);
                                kv.appendChild(row);
                            });

                        card.appendChild(kv);

                        const tokens = stopTokens && typeof stopTokens === 'object' ? stopTokens[stageName] : null;
                        const rows = Array.isArray(tokens) ? tokens : [];
                        const label = document.createElement('div');
                        label.className = 'space-context-label';
                        label.textContent = 'stop_tokens';
                        card.appendChild(label);

                        if (!rows.length) {
                            const empty = document.createElement('div');
                            empty.className = 'space-stop-empty';
                            empty.textContent = '—';
                            card.appendChild(empty);
                            return;
                        }

                        const list = document.createElement('div');
                        list.className = 'space-stop-list';
                        rows.forEach((item) => {
                            const line = document.createElement('div');
                            line.className = 'space-stop-token';
                            const id = document.createElement('div');
                            id.className = 'space-stop-id';
                            id.textContent = String(item && typeof item.id === 'number' ? item.id : item && item.id ? item.id : '');
                            const tok = document.createElement('div');
                            tok.className = 'space-stop-text';
                            tok.textContent = item && typeof item.token === 'string' ? item.token : '';
                            line.appendChild(id);
                            line.appendChild(tok);
                            list.appendChild(line);
                        });
                        card.appendChild(list);
                    });
                });
            };

            const errors = eventObj && Array.isArray(eventObj.errors) ? eventObj.errors : [];
            if (errors.length) {
                addCard(right, 'errors', (card) => {
                    addSectionPre(card, 'errors', errors.map((e) => String(e)).join('\\n'));
                });
            }

            if (
                view === 'structured' &&
                eventObj &&
                eventObj.context &&
                typeof eventObj.context === 'object' &&
                eventObj.context !== null
            ) {
                const ctx = eventObj.context;
                const stages = Array.isArray(ctx.stages) ? ctx.stages : [];
                const samplingConfig = ctx.sampling_config && typeof ctx.sampling_config === 'object' ? ctx.sampling_config : null;
                const stopTokens = eventObj.stop_tokens && typeof eventObj.stop_tokens === 'object' ? eventObj.stop_tokens : {};

                if (stages.length) {
                    stages.forEach((stage, idx) => {
                        const title = `Stage ${idx + 1}`;
                        const stopReason =
                            stage && typeof stage.stop_reason === 'string' && stage.stop_reason ? stage.stop_reason : '';
                        addCard(left, stopReason ? `${title} · stop_reason: ${stopReason}` : title, (card) => {
                            const prompt = stage && typeof stage.prompt === 'string' ? stage.prompt : '';
                            const completion = stage && typeof stage.completion === 'string' ? stage.completion : '';
                            addSectionPre(card, 'prompt', prompt);
                            addSectionMarkdown(card, 'completion', completion);
                        });
                    });
                } else {
                    renderRaw(rawText);
                }

                renderSampling(samplingConfig, stopTokens);
            } else {
                renderRaw(rawText);
            }

            modal.classList.add('open');
            modal.setAttribute('aria-hidden', 'false');
        };

        const getTextboxInput = (elemId) => {
            const rootNode = queryById(elemId);
            if (!rootNode) {
                return null;
            }
            return rootNode.matches('textarea, input')
                ? rootNode
                : rootNode.querySelector('textarea, input');
        };

        const setTextboxValue = (elemId, value) => {
            const input = getTextboxInput(elemId);
            if (!input) {
                return;
            }
            if (input.value === value) {
                return;
            }
            input.value = value;
            input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
            input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
        };

        const parseContextEvent = (rawValue) => {
            if (!rawValue) {
                return { view: 'text', raw_text: '' };
            }
            try {
                const parsed = JSON.parse(rawValue);
                if (parsed && typeof parsed === 'object') {
                    if (typeof parsed.raw_text !== 'string' && typeof parsed.text === 'string') {
                        return { ...parsed, raw_text: parsed.text };
                    }
                    return parsed;
                }
            } catch (error) {
                // Keep raw string payload as fallback.
            }
            return { view: 'text', raw_text: String(rawValue) };
        };

        let contextWatchToken = 0;

        const watchContextResult = (previousValue) => {
            const currentToken = ++contextWatchToken;
            const deadline = Date.now() + 30000;

            const poll = () => {
                if (currentToken !== contextWatchToken) {
                    return;
                }

                const input = getTextboxInput('space-context-result');
                const rawValue = input ? (input.value || '') : '';
                if (rawValue && rawValue !== previousValue) {
                    openModalEvent(parseContextEvent(rawValue));
                    return;
                }

                if (Date.now() >= deadline) {
                    if (rawValue) {
                        openModalEvent(parseContextEvent(rawValue));
                    } else {
                        openModalEvent({ view: 'text', raw_text: '读取完整 context 超时，请重试。' });
                    }
                    return;
                }

                window.setTimeout(poll, 120);
            };

            poll();
        };
        window.rwkvSpaceOpenContext = (event) => {
            openModalEvent(event || { view: 'text', raw_text: '' });
        };

        window.rwkvSpaceRequestContext = (cellId, sampleIndex, repeatIndex, passIndex) => {
            if (!cellId) {
                return;
            }
            const contextResultInput = getTextboxInput('space-context-result');
            const previousValue = contextResultInput ? contextResultInput.value : '';
            openModalEvent({ view: 'text', raw_text: '正在加载完整 context...' });
            setTextboxValue(
                'space-context-payload',
                JSON.stringify({
                    cell_id: cellId,
                    sample_index: Number(sampleIndex),
                    repeat_index: Number(repeatIndex),
                    pass_index: Number(passIndex),
                    ts: Date.now(),
                }),
            );
            watchContextResult(previousValue);
        };

        window.rwkvSpaceCellClick = (cellId) => {
            if (!cellId) {
                return;
            }
            setTextboxValue(
                'space-click-payload',
                JSON.stringify({ cell_id: cellId, ts: Date.now() }),
            );
        };


        let tooltip = document.getElementById('space-hover-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'space-hover-tooltip';
            tooltip.className = 'space-hover-tooltip';
            document.body.appendChild(tooltip);
        }

        const hideTooltip = () => {
            if (!tooltip) {
                return;
            }
            tooltip.classList.remove('open');
            tooltip.innerHTML = '';
        };

        const showTooltip = (text, x, y) => {
            if (!tooltip || !text) {
                hideTooltip();
                return;
            }
            tooltip.innerHTML = String(text).replace(/\\n/g, '<br>');
            tooltip.style.left = `${x + 14}px`;
            tooltip.style.top = `${y + 14}px`;
            tooltip.classList.add('open');
        };

        const firstElementFromEvent = (event) => {
            if (typeof event.composedPath === 'function') {
                const path = event.composedPath();
                for (const node of path) {
                    if (node instanceof Element) {
                        return node;
                    }
                }
            }
            return event.target instanceof Element ? event.target : null;
        };

        root.addEventListener('click', (event) => {
            const target = firstElementFromEvent(event);
            if (!target) {
                return;
            }

            const contextButton = target.closest('button.space-context-open[data-context-cell-id]');
            if (contextButton) {
                event.preventDefault();
                const cellId = contextButton.getAttribute('data-context-cell-id');
                const sampleIndex = contextButton.getAttribute('data-sample-index');
                const repeatIndex = contextButton.getAttribute('data-repeat-index');
                const passIndex = contextButton.getAttribute('data-pass-index');
                if (cellId) {
                    window.rwkvSpaceRequestContext(cellId, sampleIndex, repeatIndex, passIndex);
                }
                return;
            }

            const scoreCell = target.closest('td[data-clickable="1"][data-cell-id]');
            if (scoreCell) {
                event.preventDefault();
                const cellId = scoreCell.getAttribute('data-cell-id');
                if (cellId) {
                    window.rwkvSpaceCellClick(cellId);
                }
                return;
            }

            if (target.id === 'space-context-modal' || target.getAttribute('data-close-modal') === '1') {
                closeModal();
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeModal();
            }
        });

        root.addEventListener('mousemove', (event) => {
            const target = firstElementFromEvent(event);
            if (!target) {
                hideTooltip();
                return;
            }
            const cell = target.closest('td[data-tooltip]');
            if (!cell) {
                hideTooltip();
                return;
            }
            const tooltipText = cell.getAttribute('data-tooltip');
            if (!tooltipText) {
                hideTooltip();
                return;
            }
            showTooltip(tooltipText, event.clientX, event.clientY);
        });

        root.addEventListener('scroll', hideTooltip, true);
        window.addEventListener('scroll', hideTooltip, true);
    }
    """


    with gr.Blocks(css=css, head=head, theme=gr.themes.Base(), js=js_func) as demo:
        with gr.Column(elem_classes="space-root"):

            gr.HTML(
                """
<div class="space-header">
    <h1>RWKV Skills · Space</h1>
    <div class="subtitle">以最新分数为基准，快速浏览各评测领域。</div>
</div>
"""
            )

            gr.HTML('<div class="divider accent"></div>')

            with gr.Group(elem_classes="space-controls-card"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="模型选择",
                        info="默认项会对每个架构 + 参数量组合选取 data_version（G0→…→G1d）最新的模型；手动选择时展示单个模型的最新数据库分数。",
                        choices=model_choices,
                        value=AUTO_MODEL_LABEL,
                        scale=3,
                        elem_classes="space-dropdown",
                    )
                    refresh_btn = gr.Button("刷新分数", variant="primary", scale=1)
                    download_btn = gr.DownloadButton("导出为 CSV", scale=1, value=csv_export_path)

                table_view = gr.Radio(
                    label="表格视图",
                    choices=[(label, key) for key, label in TABLE_VIEW_LABELS.items()],
                    value=initial_view_mode,
                    info="在旧表位置切换：最新明细 / 最新均分 / 与上一代对比",
                )

                summary_md = gr.Markdown(summary, elem_classes="space-info-card")

            interaction_state = gr.State(interaction_meta)
            selected_cell_state = gr.State(None)
            eval_records_state = gr.State(initial_eval_rows)
            eval_loader_state = gr.State(initial_eval_loader)
            tables: dict[str, gr.HTML] = {}

            with gr.Tabs(elem_classes="tabs"):
                for group in DOMAIN_GROUPS:
                    with gr.Tab(group["label"]):
                        gr.HTML('<div class="space-spacer"></div>')
                        tables[group["key"]] = gr.HTML(
                            domain_tables.get(group["key"], _render_pivot_html([], [])),
                        )

            with gr.Row(elem_classes="space-eval-toggle-row"):
                wrong_only_toggle = gr.Checkbox(
                    label="仅展示错题",
                    value=False,
                    elem_id="space-wrong-only-toggle",
                    container=False,
                )

            eval_records_html = gr.HTML(initial_eval_html, elem_id="space-eval-records-panel")

            with gr.Row(elem_classes="space-eval-pagination-row"):
                load_next_page_btn = gr.Button(
                    f"下一页（每次 {EVAL_FETCH_ROWS} 条）",
                    variant="secondary",
                    interactive=False,
                    elem_id="space-load-next-page-btn",
                )

            gr.HTML('<div class="divider accent"></div>')
            gr.HTML('<div class="space-footer">RWKV Skills · Space</div>')

            gr.HTML(modal_html)

            click_payload = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-click-payload",
                elem_classes="space-hidden-input",
            )
            context_payload = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-context-payload",
                elem_classes="space-hidden-input",
            )
            context_result = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-context-result",
                elem_classes="space-hidden-input",
            )

            def update_dashboard(
                selected_model: str,
                selected_view_mode: str,
                only_wrong: bool,
                selected_cell: str | None,
            ):
                load_errors: list[str] = []
                entries = load_scores(errors=load_errors)
                model_choices = _compute_choices(entries)
                dropdown_value = selected_model if selected_model in model_choices else AUTO_MODEL_LABEL
                view_mode = _normalize_table_view(selected_view_mode)

                selection_state = _prepare_selection(entries, dropdown_value)
                warnings = (
                    load_errors
                    + ([style_warning] if style_warning else [])
                    + ([vendor_css_warning] if vendor_css_warning else [])
                    + ([head_warning] if head_warning else [])
                )
                csv_path = _export_selection_csv(selection_state)

                summary_value = _render_summary(
                    all_entries=entries,
                    visible=selection_state.entries,
                    selection=selection_state,
                    view_mode=view_mode,
                    warnings=warnings,
                )
                domain_table_values, interaction_map = _build_domain_tables(
                    selection_state,
                    all_entries=entries,
                    view_mode=view_mode,
                )

                selected_cell_id = selected_cell if isinstance(selected_cell, str) else None
                if selected_cell_id and selected_cell_id not in interaction_map:
                    selected_cell_id = None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=selected_cell_id,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map,
                )

                outputs: list[Any] = [
                    gr.update(choices=model_choices, value=selection_state.dropdown_value),
                    gr.update(value=csv_path),
                    gr.update(value=summary_value),
                ]
                for group in DOMAIN_GROUPS:
                    outputs.append(gr.update(value=domain_table_values.get(group["key"])))
                outputs.extend(
                    [
                        interaction_map,
                        resolved_cell,
                        gr.update(value=eval_html),
                        eval_rows,
                        eval_loader,
                        gr.update(interactive=should_poll),
                    ]
                )
                return outputs

            dashboard_outputs: list[Any] = [
                model_dropdown,
                download_btn,
                summary_md,
                tables["knowledge"],
                tables["math"],
                tables["coding"],
                tables["instruction_following"],
                tables["function_call"],
                interaction_state,
                selected_cell_state,
                eval_records_html,
                eval_records_state,
                eval_loader_state,
                load_next_page_btn,
            ]

            model_dropdown.change(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )
            refresh_btn.click(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )
            table_view.change(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )

            def on_cell_click(
                payload: str,
                only_wrong: bool,
                interaction_map: dict[str, dict[str, Any]],
                selected_cell: str | None,
            ):
                parsed_cell = _parse_click_payload(payload)
                if parsed_cell is None:
                    parsed_cell = selected_cell if isinstance(selected_cell, str) else None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=parsed_cell,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return (
                    gr.update(value=eval_html),
                    resolved_cell,
                    eval_rows,
                    eval_loader,
                    gr.update(interactive=should_poll),
                )

            click_payload.input(
                on_cell_click,
                inputs=[click_payload, wrong_only_toggle, interaction_state, selected_cell_state],
                outputs=[
                    eval_records_html,
                    selected_cell_state,
                    eval_records_state,
                    eval_loader_state,
                    load_next_page_btn,
                ],
                queue=False,
            )

            def on_context_click(payload: str, interaction_map: dict[str, dict[str, Any]]):
                context_event = _build_context_event_payload(
                    payload=payload,
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return gr.update(value=json.dumps(context_event, ensure_ascii=False))

            context_payload.input(
                on_context_click,
                inputs=[context_payload, interaction_state],
                outputs=[context_result],
                queue=False,
            )

            def on_wrong_toggle(
                only_wrong: bool,
                interaction_map: dict[str, dict[str, Any]],
                selected_cell: str | None,
            ):
                cell_id = selected_cell if isinstance(selected_cell, str) else None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=cell_id,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return (
                    gr.update(value=eval_html),
                    resolved_cell,
                    eval_rows,
                    eval_loader,
                    gr.update(interactive=should_poll),
                )

            wrong_only_toggle.change(
                on_wrong_toggle,
                inputs=[wrong_only_toggle, interaction_state, selected_cell_state],
                outputs=[
                    eval_records_html,
                    selected_cell_state,
                    eval_records_state,
                    eval_loader_state,
                    load_next_page_btn,
                ],
                queue=False,
            )

            def on_load_next_page(
                loader_state: dict[str, Any],
                current_rows: list[dict[str, Any]],
            ):
                eval_html, merged_rows, next_loader, has_more = _continue_eval_records_load(
                    loader_state=loader_state if isinstance(loader_state, dict) else _empty_eval_loader_state(),
                    records=current_rows if isinstance(current_rows, list) else [],
                )
                return (
                    gr.update(value=eval_html),
                    merged_rows,
                    next_loader,
                    gr.update(interactive=has_more),
                )

            load_next_page_btn.click(
                on_load_next_page,
                inputs=[eval_loader_state, eval_records_state],
                outputs=[eval_records_html, eval_records_state, eval_loader_state, load_next_page_btn],
                queue=False,
            )

            def export_csv(selected_model: str):
                entries = load_scores()
                selection_state = _prepare_selection(entries, selected_model)
                return _export_selection_csv(selection_state)

            download_btn.click(
                export_csv,
                inputs=[model_dropdown],
                outputs=download_btn,
                queue=False,
            )

    return demo


def main() -> None:
    demo = _build_dashboard()
    # Allow KaTeX font files to be served via Gradio's /file= endpoint.
    fonts_dir = Path(__file__).parent / "assets" / "vendor" / "fonts"
    allowed_paths = [str(fonts_dir)] if fonts_dir.exists() else None
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=allowed_paths)


if __name__ == "__main__":  # pragma: no cover
    main()
