"""Render an experiment snapshot as a browsable HTML report."""

from __future__ import annotations

import base64
import logging
import mimetypes
from html import escape
from typing import TYPE_CHECKING, Any

from src.snapshot import FewShotEntry, SampleRecord, load_experiment_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from src.tools import ToolCallRecord

logger = logging.getLogger(__name__)


def _image_to_data_uri(image_path: Path) -> str:
    """Encode an image file as a base64 data URI for embedding in HTML."""
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _status_class(sample: SampleRecord) -> str:
    if sample.status == "failed":
        return "failed"
    if sample.is_correct is None:
        return "unknown"
    return "correct" if sample.is_correct else "incorrect"


def _status_badge(sample: SampleRecord) -> str:
    if sample.status == "failed":
        return '<span class="badge failed">FALLBACK</span>'
    if sample.is_correct is None:
        return '<span class="badge unknown">?</span>'
    if sample.is_correct:
        return '<span class="badge correct">CORRECT</span>'
    return '<span class="badge incorrect">WRONG</span>'


def _render_few_shot_section(
    few_shot_examples: list[FewShotEntry],
    resolve_images_from: Path | None,
) -> str:
    if not few_shot_examples:
        return ""
    rows = ""
    for ex in few_shot_examples:
        label_text = "Yes (1)" if ex.label == 1 else "No (0)"

        # Embed image if possible
        image_html = f'<span class="fs-path">{escape(ex.image_path)}</span>'
        if resolve_images_from:
            img_path = resolve_images_from / ex.image_path
            if img_path.exists():
                data_uri = _image_to_data_uri(img_path)
                image_html = f'<img src="{data_uri}" class="fs-img" loading="lazy" />'

        # Similarity badge
        sim_html = ""
        if ex.similarity is not None:
            sim_html = f'<span class="badge sim">sim: {ex.similarity:.3f}</span>'

        rows += f"""
        <tr>
            <td>{image_html}</td>
            <td>{escape(ex.prompt[:120])}{sim_html}</td>
            <td><strong>{label_text}</strong></td>
        </tr>"""
    return f"""
    <details class="section">
        <summary>Few-Shot Examples ({len(few_shot_examples)})</summary>
        <table class="fs-table">
            <thead><tr><th>Image</th><th>Prompt</th><th>Label</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </details>"""


def _render_tool_calls_section(
    tool_calls: list[ToolCallRecord] | None,
    exp_dir: Path | None,
) -> str:
    """Render tool call log with step images for a sample card."""
    if not tool_calls:
        return ""
    rows = ""
    for tc in tool_calls:
        args_str = ", ".join(f"{k}={v}" for k, v in tc.args.items())
        # Embed the tool result image if available
        img_html = ""
        if tc.image_path and exp_dir:
            img_file = exp_dir / tc.image_path
            if img_file.exists():
                data_uri = _image_to_data_uri(img_file)
                img_html = f'<img src="{data_uri}" class="tool-img" loading="lazy" />'
        rows += f"""
        <li>
            <code>{escape(tc.tool)}</code>({escape(args_str)})
            {img_html}
        </li>"""
    return f"""
    <details class="section">
        <summary>Tool Calls ({len(tool_calls)})</summary>
        <ol class="tool-calls-list">{rows}</ol>
    </details>"""


def _render_conversation_section(
    conversation: list[dict[str, Any]] | None,
    exp_dir: Path | None,
) -> str:
    """Render the full agentic conversation as a chat-style timeline."""
    if not conversation:
        return ""

    step = 0
    items = ""
    for msg in conversation:
        msg.get("kind", "")
        parts = msg.get("parts", [])

        for part in parts:
            pk = part.get("part_kind", "")

            if pk == "system-prompt":
                # Skip — already shown in its own section
                continue

            if pk == "user-prompt":
                content = part.get("content", "")
                # Extract text from mixed content (image binary + text)
                if isinstance(content, list):
                    text_parts = [c for c in content if isinstance(c, str)]
                    text = "\n".join(text_parts)
                else:
                    text = str(content)
                # Skip image-only user prompts (tool result images injected by pydantic-ai)
                if not text.strip():
                    continue
                items += f"""
                    <div class="conv-msg conv-user">
                        <div class="conv-label">User</div>
                        <pre class="conv-text">{escape(text[:2000])}</pre>
                    </div>"""

            elif pk == "thinking":
                content = part.get("content", "")
                if content:
                    step += 1
                    items += f"""
                    <div class="conv-msg conv-thinking">
                        <div class="conv-label">Thinking (step {step})</div>
                        <pre class="conv-text">{escape(str(content))}</pre>
                    </div>"""

            elif pk == "tool-call":
                tool_name = part.get("tool_name", "?")
                args = part.get("args", "")
                if tool_name == "final_result":
                    # Show the final structured output
                    items += f"""
                    <div class="conv-msg conv-final">
                        <div class="conv-label">Final Answer</div>
                        <pre class="conv-text">{escape(str(args))}</pre>
                    </div>"""
                else:
                    items += f"""
                    <div class="conv-msg conv-tool-call">
                        <div class="conv-label">Tool Call: {escape(tool_name)}</div>
                        <pre class="conv-text">{escape(str(args))}</pre>
                    </div>"""

            elif pk == "tool-return":
                tool_name = part.get("tool_name", "?")
                content = part.get("content", "")
                if tool_name == "final_result":
                    continue
                # Show text content of tool return
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts = [c for c in content if isinstance(c, str)]
                    text = "\n".join(text_parts)
                else:
                    text = str(content)

                # Try to find matching tool image from exp_dir
                img_html = ""
                if exp_dir:
                    tool_images_dir = exp_dir / "tool_images"
                    if tool_images_dir.exists():
                        # Find images matching this step
                        # Steps are 1-indexed in filenames
                        step_imgs = sorted(tool_images_dir.glob(f"*_step_{step}_*.png"))
                        if step_imgs:
                            data_uri = _image_to_data_uri(step_imgs[0])
                            img_html = f'<img src="{data_uri}" class="conv-tool-img" loading="lazy" />'

                items += f"""
                    <div class="conv-msg conv-tool-return">
                        <div class="conv-label">Tool Result: {escape(tool_name)}</div>
                        <pre class="conv-text">{escape(text[:1000])}</pre>
                        {img_html}
                    </div>"""

            elif pk == "text":
                content = part.get("content", "")
                if content:
                    items += f"""
                    <div class="conv-msg conv-assistant">
                        <div class="conv-label">Assistant</div>
                        <pre class="conv-text">{escape(str(content))}</pre>
                    </div>"""

    if not items:
        return ""

    return f"""
    <details class="section">
        <summary>Conversation ({step} thinking steps)</summary>
        <div class="conv-timeline">{items}</div>
    </details>"""


def _render_card(
    sample: SampleRecord,
    system_prompt: str,
    few_shot_examples: list[FewShotEntry],
    image_data_uri: str | None,
    resolve_images_from: Path | None,
    exp_dir: Path | None = None,
) -> str:
    gt_text = "N/A"
    if sample.ground_truth is not None:
        gt_text = str(sample.ground_truth)

    image_html = ""
    if image_data_uri:
        image_html = f'<img src="{image_data_uri}" class="sample-img" loading="lazy" />'
    else:
        image_html = f'<p class="no-img">Image: {escape(sample.image_path)}</p>'

    # Use per-sample few-shot if available, otherwise experiment-level
    card_few_shot = sample.few_shot_examples if sample.few_shot_examples is not None else few_shot_examples

    # Cost display
    cost_html = ""
    if sample.usage.cost_usd > 0:
        cost_html = f'<span class="card-cost">${sample.usage.cost_usd:.4f}</span>'

    return f"""
    <div class="card {_status_class(sample)}" data-status="{_status_class(sample)}">
        <div class="card-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="card-index">#{sample.index}</span>
            {_status_badge(sample)}
            <span class="card-answer">Answer: <strong>{sample.model_answer}</strong></span>
            <span class="card-gt">GT: <strong>{gt_text}</strong></span>
            {cost_html}
            <span class="card-prompt-preview">{escape(sample.user_prompt[:80])}</span>
            <span class="expand-icon">&#9660;</span>
        </div>
        <div class="card-body">
            <div class="card-columns">
                <div class="card-left">
                    {image_html}
                </div>
                <div class="card-right">
                    <div class="section">
                        <h4>User Prompt</h4>
                        <pre class="prompt-text">{escape(sample.user_prompt)}</pre>
                    </div>
                    <details class="section">
                        <summary>System Prompt</summary>
                        <pre class="prompt-text">{escape(system_prompt)}</pre>
                    </details>
                    {_render_few_shot_section(card_few_shot, resolve_images_from)}
                    {_render_tool_calls_section(sample.tool_calls, exp_dir)}
                    {_render_conversation_section(sample.conversation, exp_dir)}
                    <div class="section">
                        <h4>Model Reasoning</h4>
                        <pre class="reasoning-text">{escape(sample.model_reasoning)}</pre>
                    </div>
                    <div class="section">
                        <h4>Raw Model Response</h4>
                        <details>
                            <summary>Show raw response</summary>
                            <pre class="raw-text">{escape(sample.model_raw_response)}</pre>
                        </details>
                    </div>
                </div>
            </div>
        </div>
    </div>"""


_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #333; padding: 20px; }
.header { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.header h1 { font-size: 1.4em; margin-bottom: 10px; }
.stats { display: flex; gap: 20px; flex-wrap: wrap; }
.stat { background: #f0f0f0; padding: 8px 16px; border-radius: 4px; }
.stat strong { color: #1a73e8; }
.filters { margin: 16px 0; display: flex; gap: 8px; }
.filter-btn { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
              cursor: pointer; background: #fff; font-size: 0.9em; }
.filter-btn.active { background: #1a73e8; color: #fff; border-color: #1a73e8; }
.card { background: #fff; border-radius: 8px; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #ccc; overflow: hidden; }
.card.correct { border-left-color: #34a853; }
.card.incorrect { border-left-color: #ea4335; }
.card.unknown { border-left-color: #999; }
.card.failed { border-left-color: #f29900; }
.card-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
               cursor: pointer; user-select: none; }
.card-header:hover { background: #fafafa; }
.card-index { font-weight: 700; font-size: 1.1em; min-width: 40px; }
.badge { padding: 2px 8px; border-radius: 3px; font-size: 0.75em; font-weight: 600; }
.badge.correct { background: #e6f4ea; color: #137333; }
.badge.incorrect { background: #fce8e6; color: #c5221f; }
.badge.unknown { background: #e8e8e8; color: #666; }
.badge.failed { background: #fff3e0; color: #b06000; }
.badge.sim { background: #e8f0fe; color: #1a73e8; margin-left: 6px; }
.card-answer, .card-gt { font-size: 0.9em; }
.card-cost { font-size: 0.8em; color: #666; font-family: monospace; }
.card-prompt-preview { font-size: 0.85em; color: #666; flex: 1;
                       overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.expand-icon { font-size: 0.7em; color: #999; transition: transform 0.2s; }
.card.expanded .expand-icon { transform: rotate(180deg); }
.card-body { display: none; padding: 16px; border-top: 1px solid #eee; }
.card.expanded .card-body { display: block; }
.card-columns { display: flex; gap: 20px; }
.card-left { flex: 0 0 300px; }
.card-right { flex: 1; min-width: 0; }
.sample-img { max-width: 300px; max-height: 300px; border-radius: 4px; border: 1px solid #ddd;
              cursor: pointer; transition: opacity 0.2s; }
.sample-img:hover { opacity: 0.85; }
.no-img { color: #999; font-style: italic; }
.section { margin-bottom: 12px; }
.section h4 { font-size: 0.9em; color: #555; margin-bottom: 4px; }
.section summary { font-weight: 600; font-size: 0.9em; cursor: pointer; color: #555; }
pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em;
      background: #f8f8f8; padding: 8px; border-radius: 4px; max-height: 400px; overflow-y: auto; }
.prompt-text { border-left: 3px solid #1a73e8; }
.reasoning-text { border-left: 3px solid #34a853; }
.raw-text { border-left: 3px solid #999; }
.fs-table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 8px; }
.fs-table th, .fs-table td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }
.fs-table td:first-child { width: 120px; }
.fs-img { max-width: 100px; max-height: 100px; border-radius: 4px; border: 1px solid #ddd; }
.fs-path { font-family: monospace; font-size: 0.8em; color: #666; }
.tool-calls-list { padding-left: 20px; }
.tool-calls-list li { margin-bottom: 10px; }
.tool-calls-list code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
.tool-img { max-width: 300px; max-height: 200px; border-radius: 4px; border: 1px solid #ddd;
            display: block; margin-top: 6px; cursor: pointer; transition: opacity 0.2s; }
.tool-img:hover { opacity: 0.85; }
.conv-timeline { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.conv-msg { padding: 8px 12px; border-radius: 6px; border-left: 3px solid #ccc; }
.conv-label { font-size: 0.75em; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
.conv-text { font-size: 0.83em; background: transparent; padding: 4px 0; margin: 0;
             max-height: 300px; overflow-y: auto; }
.conv-user { background: #e8f0fe; border-left-color: #1a73e8; }
.conv-user .conv-label { color: #1a73e8; }
.conv-thinking { background: #fef7e0; border-left-color: #f9ab00; }
.conv-thinking .conv-label { color: #e37400; }
.conv-tool-call { background: #f3e8fd; border-left-color: #8e24aa; }
.conv-tool-call .conv-label { color: #8e24aa; }
.conv-tool-return { background: #e8f5e9; border-left-color: #34a853; }
.conv-tool-return .conv-label { color: #2e7d32; }
.conv-tool-img { max-width: 400px; max-height: 300px; border-radius: 4px; border: 1px solid #ddd;
                 display: block; margin-top: 6px; cursor: pointer; transition: opacity 0.2s; }
.conv-tool-img:hover { opacity: 0.85; }
/* Lightbox overlay for expanded images */
.lightbox-overlay { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.92); z-index: 9999; cursor: pointer;
                    justify-content: center; align-items: center; padding: 0; }
.lightbox-overlay.active { display: flex; }
.lightbox-overlay img { width: 100vw; height: 100vh; object-fit: contain;
                        background: rgba(0,0,0,0.92); }
.lightbox-close { position: fixed; top: 12px; right: 20px; color: #fff; font-size: 2.5em;
                  cursor: pointer; z-index: 10000; font-weight: 700; line-height: 1;
                  text-shadow: 0 2px 6px rgba(0,0,0,0.7); }
.conv-assistant { background: #f5f5f5; border-left-color: #666; }
.conv-assistant .conv-label { color: #555; }
.conv-final { background: #e6f4ea; border-left-color: #137333; }
.conv-final .conv-label { color: #137333; }
"""

_JS = """
function filterCards(status) {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.card').forEach(card => {
        if (status === 'all' || card.dataset.status === status) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}
/* Lightbox: click any image to expand */
(function() {
    var overlay = document.getElementById('lightbox-overlay');
    var lbImg = document.getElementById('lightbox-img');
    document.body.addEventListener('click', function(e) {
        var img = e.target.closest('img.sample-img, img.tool-img, img.conv-tool-img, img.fs-img');
        if (img) {
            e.stopPropagation();
            lbImg.src = img.src;
            overlay.classList.add('active');
        }
    });
    overlay.addEventListener('click', function() {
        overlay.classList.remove('active');
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') overlay.classList.remove('active');
    });
})();
"""


def generate_report(
    exp_dir: Path,
    output_path: Path,
    resolve_images_from: Path | None = None,
) -> Path:
    """Generate a self-contained HTML report from an experiment snapshot.

    Args:
        exp_dir: Experiment directory containing snapshot.json.
        output_path: Path to write the HTML report.
        resolve_images_from: Base directory for resolving relative image paths.
            If None, images are referenced by path text only (no embedding).

    Returns:
        Path to the generated HTML file.
    """
    experiment = load_experiment_snapshot(exp_dir)
    samples = experiment.samples

    if not samples:
        logger.warning("No samples in snapshot at %s", exp_dir)
        return output_path

    # Compute stats
    total = len(samples)
    with_gt = [s for s in samples if s.ground_truth is not None]
    correct = sum(1 for s in with_gt if s.is_correct)
    accuracy = f"{correct / len(with_gt) * 100:.1f}%" if with_gt else "N/A"
    u = experiment.total_usage
    tokens_str = f"{u.total_tokens:,} (in: {u.input_tokens:,} / out: {u.output_tokens:,})"
    cache_str = f"R:{u.cache_read_tokens:,} W:{u.cache_write_tokens:,}"
    cost_str = f"${u.cost_usd:.4f}"

    # Render cards
    cards_html = ""
    for sample in samples:
        image_data_uri: str | None = None
        if resolve_images_from:
            img_path = resolve_images_from / sample.image_path
            if img_path.exists():
                image_data_uri = _image_to_data_uri(img_path)

        cards_html += _render_card(
            sample,
            system_prompt=experiment.system_prompt,
            few_shot_examples=experiment.few_shot_examples,
            image_data_uri=image_data_uri,
            resolve_images_from=resolve_images_from,
            exp_dir=exp_dir,
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DataCV Snapshot Report — {escape(experiment.exp_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
    <h1>DataCV Snapshot Report</h1>
    <div class="stats">
        <div class="stat">Experiment: <strong>{escape(experiment.exp_name)}</strong></div>
        <div class="stat">Task: <strong>{escape(experiment.task)}</strong></div>
        <div class="stat">Model: <strong>{escape(experiment.model_name)}</strong></div>
        <div class="stat">Samples: <strong>{total}</strong></div>
        <div class="stat">Accuracy: <strong>{accuracy}</strong> ({correct}/{len(with_gt)})</div>
        <div class="stat">Tokens: <strong>{tokens_str}</strong></div>
        <div class="stat">Cache: <strong>{cache_str}</strong></div>
        <div class="stat">Cost: <strong>{cost_str}</strong></div>
    </div>
    <div class="filters">
        <button class="filter-btn active" onclick="filterCards('all')">All</button>
        <button class="filter-btn" onclick="filterCards('failed')">Failed</button>
        <button class="filter-btn" onclick="filterCards('correct')">Correct</button>
        <button class="filter-btn" onclick="filterCards('incorrect')">Incorrect</button>
        <button class="filter-btn" onclick="filterCards('unknown')">Unknown</button>
    </div>
</div>
{cards_html}
<div id="lightbox-overlay" class="lightbox-overlay">
    <span class="lightbox-close">&times;</span>
    <img id="lightbox-img" src="" alt="Expanded image" />
</div>
<script>{_JS}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Generated report with %d samples → %s", total, output_path)
    return output_path
