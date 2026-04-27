"""
OpenTeddy Report Tool
Render a markdown document (with ``` ```chart ``` ``` JSON blocks) into a
standalone, self-contained HTML file.

Why this exists:
  Analytic-mode tasks produce analysis JSON + want to emit a pretty
  report. Without this tool, the agent's only path is "write a Python
  script that hand-rolls an HTML template with Chart.js" — which small
  local models routinely mangle (missing script tags, broken JSON
  escaping, wrong Chart.js version).

  This tool gives them a single call:
      render_chart_report(markdown="# Q1 Sales\n\n```chart\n{...}\n```",
                          output_path="reports/q1.html")
  and returns the saved file's path + size + chart count.

The rendered HTML:
  - Embeds Chart.js v4 from jsDelivr CDN (same version the UI uses).
  - Uses the same ``` ```chart ``` ``` JSON convention the agent already knows.
  - Minimal CSS — looks decent out of the box, readable on mobile.
  - No external dependencies at runtime beyond Chart.js itself.

If you open the file offline, charts won't render (CDN unreachable).
For fully-offline reports, vendor Chart.js locally — out of scope for
Slice 1; the common case is "share a link / email an attachment that
opens in a browser with internet".
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Markdown → HTML renderer ──────────────────────────────────────────────────
# Mirrors the client-side renderer in static/index.html's renderMarkdown().
# Kept deliberately small — headings, lists, paragraphs, bold / inline
# code, hr, ``` blocks, ```chart blocks. No images / tables / nested
# lists. Agents tend to produce simple markdown; anything fancier they
# can paste raw HTML into.

_CODE_FENCE_RE = re.compile(r"```([\w-]*)?\n?([\s\S]*?)```")
_INLINE_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_HR_RE          = re.compile(r"^---+\s*$")
_H_RE           = re.compile(r"^(#{1,4})\s+(.+)$")
_LIST_RE        = re.compile(r"^[ \t]*[\*\-][ \t]+(.+)$")


def _inline(text: str) -> str:
    """HTML-escape + bold/inline-code transform. Operates on text that's
    already been HTML-escaped at the block level (to keep raw HTML in
    content from breaking out)."""
    text = _INLINE_BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _INLINE_CODE_RE.sub(
        lambda m: f'<code class="md-inline-code">{m.group(1)}</code>',
        text,
    )
    return text


def _markdown_to_html(src: str) -> Tuple[str, int]:
    """Convert markdown to HTML fragment. Returns (html, chart_count).

    Chart blocks become:
      <div class="chart-wrap"><canvas data-chart-spec="<b64>"></canvas></div>
    An inline <script> at the bottom of the page walks those canvases
    and calls Chart.js.
    """
    src = src.replace("\r\n", "\n")

    # 1. Extract fenced code blocks so their contents survive verbatim.
    code_blocks: List[Dict[str, str]] = []
    def _capture(m: re.Match) -> str:
        code_blocks.append({
            "lang": (m.group(1) or "").strip().lower(),
            "code": m.group(2).rstrip("\n"),
        })
        return f"\n\n§§CB{len(code_blocks) - 1}§§\n\n"
    src = _CODE_FENCE_RE.sub(_capture, src)

    # 2. HTML-escape everything else.
    src = html.escape(src)

    chart_count = 0
    out: List[str] = []

    # 3. Walk top-level blocks split on blank lines.
    for raw in re.split(r"\n{2,}", src):
        block = raw.strip("\n")
        if not block:
            continue

        # Code-block placeholder
        cb_m = re.match(r"^§§CB(\d+)§§$", block)
        if cb_m:
            entry = code_blocks[int(cb_m.group(1))]
            if entry["lang"] == "chart":
                # Try to validate JSON up-front so bad agent output
                # doesn't turn into "silent blank box" at render time.
                try:
                    json.loads(entry["code"])
                    import base64
                    b64 = base64.b64encode(
                        entry["code"].encode("utf-8")
                    ).decode("ascii")
                    out.append(
                        f'<div class="chart-wrap">'
                        f'<canvas data-chart-spec="{b64}"></canvas></div>'
                    )
                    chart_count += 1
                except json.JSONDecodeError as exc:
                    out.append(
                        f'<div class="chart-error">Invalid chart JSON: '
                        f'{html.escape(str(exc))}<br><pre>'
                        f'{html.escape(entry["code"])}</pre></div>'
                    )
            else:
                out.append(f'<pre class="code-block">{html.escape(entry["code"])}</pre>')
            continue

        # Horizontal rule (block is the whole HR)
        if _HR_RE.match(block):
            out.append("<hr>")
            continue

        # GFM-style pipe table:
        #   | col1 | col2 |
        #   |---|---|         ← separator row identifies the block as a table
        #   | a  | 1 |
        # Without this branch the table lines fall through to the
        # paragraph/<br> path and end up rendered as raw `|---|---|`
        # text in the agent's HTML reports.
        tlines = [ln for ln in block.split("\n") if ln.strip()]
        if (
            len(tlines) >= 2
            and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", tlines[1])
            and "-" in tlines[1]
        ):
            def _split_row(row: str) -> List[str]:
                s = row.strip()
                if s.startswith("|"): s = s[1:]
                if s.endswith("|"):   s = s[:-1]
                return [c.strip() for c in s.split("|")]

            aligns: List[Optional[str]] = []
            for c in _split_row(tlines[1]):
                left, right = c.startswith(":"), c.endswith(":")
                if   left and right: aligns.append("center")
                elif right:          aligns.append("right")
                elif left:           aligns.append("left")
                else:                aligns.append(None)

            def _style(i: int) -> str:
                return f' style="text-align:{aligns[i]}"' if i < len(aligns) and aligns[i] else ""

            head_cells = _split_row(tlines[0])
            body_rows  = [_split_row(r) for r in tlines[2:]]
            head_html = "<thead><tr>" + "".join(
                f"<th{_style(i)}>{_inline(c)}</th>" for i, c in enumerate(head_cells)
            ) + "</tr></thead>"
            body_html = ""
            if body_rows:
                body_html = "<tbody>" + "".join(
                    "<tr>" + "".join(
                        f"<td{_style(i)}>{_inline(c)}</td>" for i, c in enumerate(r)
                    ) + "</tr>" for r in body_rows
                ) + "</tbody>"
            out.append(
                f'<div class="md-table-wrap"><table class="md-table">'
                f'{head_html}{body_html}</table></div>'
            )
            continue

        # Walk each line and classify as heading / list / paragraph.
        # Done line-by-line rather than block-first because AI-generated
        # markdown frequently omits the blank line between a heading
        # and its body ("## Title\nfirst paragraph..."), lumping them
        # into one block. Block-level heading detection misses that.
        lines = block.split("\n")
        list_buf: List[str] = []
        text_buf: List[str] = []
        def flush_list() -> None:
            if list_buf:
                items = "".join(f"<li>{_inline(x)}</li>" for x in list_buf)
                out.append(f"<ul>{items}</ul>")
                list_buf.clear()
        def flush_text() -> None:
            if text_buf:
                out.append(f"<p>{'<br>'.join(_inline(x) for x in text_buf)}</p>")
                text_buf.clear()
        for line in lines:
            hm = _H_RE.match(line)
            lm = _LIST_RE.match(line)
            if hm:
                flush_list(); flush_text()
                level = len(hm.group(1))
                out.append(f"<h{level}>{_inline(hm.group(2))}</h{level}>")
            elif lm:
                flush_text()
                list_buf.append(lm.group(1))
            else:
                flush_list()
                text_buf.append(line)
        flush_list()
        flush_text()

    return "\n".join(out), chart_count


# ── HTML document template ────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<!-- Datalabels plugin paints values / percentages directly onto bars and
     pie slices so the user reads numbers off the chart at a glance. -->
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
  :root {
    --bg: #faf9f7; --surface: #fff; --border: #e5e0d8;
    --text: #1a1a1a; --muted: #6b6b6b; --accent: #d97757;
    --purple: #7c5cbf;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
  }
  main {
    max-width: 880px; margin: 0 auto;
    background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 2.5rem 2.75rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
  }
  h1 { font-size: 1.75rem; margin: 0 0 1.2rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }
  h2 { font-size: 1.25rem; margin: 2rem 0 0.6rem; }
  h3 { font-size: 1.05rem; margin: 1.5rem 0 0.4rem; color: var(--muted); }
  h4 { font-size: 0.95rem; margin: 1.2rem 0 0.3rem; color: var(--muted); font-weight: 600; }
  p { margin: 0 0 0.9rem; }
  ul { margin: 0.3rem 0 1rem 1.4rem; padding: 0; }
  li { margin: 0.2rem 0; }
  strong { font-weight: 600; }
  hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
  code.md-inline-code {
    font-family: 'Courier New', monospace;
    font-size: 0.88em;
    background: #f3f1ed;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 5px; color: var(--purple);
  }
  pre.code-block {
    background: #f3f1ed; border: 1px solid var(--border);
    border-radius: 6px; padding: 0.7rem 0.9rem;
    overflow-x: auto; font-size: 0.82rem;
    font-family: 'Courier New', monospace;
  }
  .chart-wrap {
    background: #fafafa; border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; margin: 1rem 0;
    height: 320px; position: relative;
  }
  .chart-wrap canvas { max-width: 100%; height: 290px !important; }
  .chart-error {
    color: #cc3333; background: #fff0f0; border: 1px solid #fca5a5;
    border-radius: 6px; padding: 0.6rem 0.8rem;
    font-family: 'Courier New', monospace; font-size: 0.82rem;
    margin: 1rem 0;
  }
  .chart-error pre { background: transparent; border: none; padding: 0.4rem 0; }
  .md-table-wrap {
    margin: 0.6rem 0 0.4rem;
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
  }
  .md-table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    font-size: 0.88rem;
    line-height: 1.45;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .md-table thead th {
    background: #f3f1ed;
    color: var(--muted);
    font-weight: 600;
    text-align: left;
    padding: 9px 13px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    letter-spacing: 0.01em;
  }
  .md-table tbody td {
    padding: 8px 13px;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    vertical-align: top;
  }
  .md-table tbody tr:last-child td { border-bottom: none; }
  .md-table tbody tr:nth-child(even) td { background: rgba(0,0,0,0.015); }
  .md-table tbody tr td:first-child strong { color: var(--text); }
  .md-table tbody tr:has(td:first-child strong) td {
    background: #f3f1ed;
    font-weight: 600;
  }
  footer {
    max-width: 880px; margin: 1rem auto 0;
    font-size: 0.75rem; color: var(--muted); text-align: center;
  }
  @media (max-width: 640px) {
    main { padding: 1.5rem 1.2rem; border-radius: 8px; }
    body { padding: 1rem 0.5rem; }
  }
</style>
</head>
<body>
<main>
__BODY__
</main>
<footer>Generated by OpenTeddy · __TIMESTAMP__</footer>
<script>
// Register datalabels plugin once so all Chart() instances inherit
// it. Specs that explicitly opt out via plugins.datalabels = false
// still skip painting.
if (typeof ChartDataLabels !== 'undefined') Chart.register(ChartDataLabels);

function _fmtVal(v) {
  const n = Number(v);
  if (!isFinite(n)) return String(v);
  const a = Math.abs(n);
  if (a >= 1_000_000) return (n / 1_000_000).toFixed(a >= 10_000_000 ? 0 : 1) + 'M';
  if (a >= 10_000)    return (n / 1_000).toFixed(0) + 'K';
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(a < 10 ? 1 : 0);
}

function _datalabelsFor(type) {
  const ring = type === 'pie' || type === 'doughnut' || type === 'polarArea';
  if (ring) {
    return {
      color: '#fff',
      font: { weight: '600', size: 12 },
      textShadowColor: 'rgba(0,0,0,0.55)',
      textShadowBlur: 4,
      // Hide tiny slices (<4%) so labels don't pile on top of each other.
      display: ctx => {
        const v = Number(ctx.dataset.data[ctx.dataIndex] || 0);
        const total = ctx.dataset.data.reduce((a, b) => a + Number(b || 0), 0) || 1;
        return v / total >= 0.04;
      },
      formatter: (value, ctx) => {
        const total = ctx.dataset.data.reduce((a, b) => a + Number(b || 0), 0) || 1;
        const pct = (Number(value) / total) * 100;
        return _fmtVal(value) + '\\n(' + pct.toFixed(1) + '%)';
      },
    };
  }
  // Bar / line: anchor depends on sign so labels always sit OUTSIDE
  // the bar, on the side away from zero. Without this, negative bars
  // hide their labels below the visible chart area.
  return {
    anchor: ctx => (Number(ctx.dataset.data[ctx.dataIndex] ?? 0) >= 0 ? 'end' : 'start'),
    align:  'top',
    offset: 4,
    clamp:  true,
    color:  '#1a1a1a',
    font:   { weight: '600', size: 11 },
    formatter: v => {
      const n = Number(v);
      if (!isFinite(n) || n === 0) return '';
      return _fmtVal(n);
    },
  };
}

function _mergeDatalabels(spec) {
  const type = (spec.type || 'bar').toLowerCase();
  spec.options = spec.options || {};
  spec.options.plugins = spec.options.plugins || {};
  const userOpts = spec.options.plugins.datalabels;
  if (userOpts === false || (userOpts && userOpts.display === false)) return;
  const defaults = _datalabelsFor(type);
  spec.options.plugins.datalabels = Object.assign({}, defaults, userOpts || {});
  // Horizontal bars (indexAxis: 'y') read better with align:right.
  if (type === 'bar' && spec.options.indexAxis === 'y') {
    spec.options.plugins.datalabels.anchor = 'end';
    spec.options.plugins.datalabels.align  = 'right';
    spec.options.plugins.datalabels.offset = 6;
  }
  // Padding so labels at the chart edge aren't clipped.
  spec.options.layout = spec.options.layout || {};
  const pad = (type === 'bar' && spec.options.indexAxis !== 'y') ? 22 : 8;
  if (spec.options.layout.padding == null) spec.options.layout.padding = pad;
}

document.querySelectorAll('canvas[data-chart-spec]').forEach(canvas => {
  const payload = canvas.getAttribute('data-chart-spec') || '';
  try {
    const json = decodeURIComponent(escape(atob(payload)));
    const spec = JSON.parse(json);
    spec.options = spec.options || {};
    spec.options.responsive = spec.options.responsive !== false;
    spec.options.maintainAspectRatio = false;
    _mergeDatalabels(spec);
    new Chart(canvas.getContext('2d'), spec);
  } catch (e) {
    canvas.parentElement.innerHTML =
      '<div class="chart-error">Render failed: ' + e.message + '</div>';
  }
});
</script>
</body>
</html>
"""


# ── Tool implementation ───────────────────────────────────────────────────────

def _resolve_output_path(path: str) -> str:
    """Absolute path for the HTML output, anchored on the session's
    effective workspace so the agent's relative paths work sanely."""
    from config import effective_workspace_dir
    if os.path.isabs(path):
        return os.path.abspath(path)
    ws = effective_workspace_dir()
    return os.path.abspath(os.path.join(ws, path))


async def render_chart_report(
    markdown: str,
    output_path: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a markdown document (with ``` ```chart ``` ``` JSON blocks)
    into a standalone HTML file.

    Args:
      markdown:    The report body. ``` ```chart ``` ``` blocks will become
                   interactive Chart.js figures. Everything else renders
                   as standard markdown (headings, bullets, paragraphs,
                   inline ``code``, **bold**).
      output_path: Where to save the file. Relative paths resolve against
                   the session workspace; absolute paths used verbatim.
                   Parent directories are created on demand.
      title:      <title> for the HTML page. Defaults to the first H1
                   in the markdown, or "Report".

    Returns:
      {path, relative_path, size_bytes, chart_count}
    """
    start = time.monotonic()
    if not markdown or not markdown.strip():
        return make_result(
            False, error="markdown is empty — nothing to render.",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    if not output_path:
        return make_result(
            False, error="output_path is required.",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # Resolve + ensure parent dir exists.
    abs_path = _resolve_output_path(output_path)
    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False, error=f"Cannot create parent dir: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # Build HTML.
    body_html, chart_count = _markdown_to_html(markdown)

    # Pick title: explicit arg → first H1 in markdown → fallback
    if not title:
        h1 = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        title = h1.group(1).strip() if h1 else "Report"

    from datetime import datetime
    doc = (_HTML_TEMPLATE
        .replace("__TITLE__", html.escape(title))
        .replace("__BODY__", body_html)
        .replace("__TIMESTAMP__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    )

    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(doc)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False, error=f"Write failed: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    size_bytes = len(doc.encode("utf-8"))
    # Compute path relative to the workspace for a friendly display.
    try:
        from config import effective_workspace_dir
        rel = os.path.relpath(abs_path, effective_workspace_dir())
    except Exception:  # noqa: BLE001
        rel = abs_path

    logger.info(
        "render_chart_report: wrote %d bytes (%d charts) to %s",
        size_bytes, chart_count, abs_path,
    )
    return make_result(
        True,
        result={
            "path":          abs_path,
            "relative_path": rel,
            "size_bytes":    size_bytes,
            "chart_count":   chart_count,
            "title":         title,
            "hint": (
                f"Report saved. Open {rel} in a browser to view "
                f"(internet needed for Chart.js CDN). "
                + (f"{chart_count} chart(s) embedded." if chart_count
                   else "No chart blocks were provided — add "
                        "```chart JSON blocks to the markdown if you "
                        "wanted figures.")
            ),
        },
        duration_ms=int((time.monotonic() - start) * 1000),
    )


_SCHEMA_RENDER_REPORT: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "render_chart_report",
        "description": (
            "Save a markdown analysis report (with ``` ```chart ``` ``` JSON "
            "blocks for Chart.js figures) as a standalone HTML file. Use "
            "this as the FINAL step of an Analytic-mode task when the user "
            "wants a shareable artifact. The chart JSON format is the same "
            "one the chat UI renders: `{\"type\":\"bar\",\"data\":{...},"
            "\"options\":{...}}`. Do NOT hand-roll HTML templates — this "
            "tool produces a self-contained file with Chart.js v4 already "
            "wired up. Low risk — only writes the one output file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": (
                        "Full report body in markdown. Use headings (# ##), "
                        "bullets (-), **bold**, `inline code`, and one or "
                        "more ``` ```chart ``` ``` JSON blocks for interactive "
                        "Chart.js figures. Each chart block must contain "
                        "valid Chart.js v4 JSON ({type, data, options})."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Where to save the HTML. Relative paths resolve "
                        "against the session workspace; absolute used as-is. "
                        "Example: 'reports/q1_sales.html'."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "HTML <title>. If omitted, uses the first H1 in the "
                        "markdown, or 'Report'."
                    ),
                },
            },
            "required": ["markdown", "output_path"],
        },
    },
}


REPORT_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (render_chart_report, _SCHEMA_RENDER_REPORT, "low"),
]
