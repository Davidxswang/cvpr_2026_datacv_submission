"""Agentic drawing tools for breaking visual illusions.

General-purpose drawing primitives (line, rectangle, circle, crop) that the
model can compose to annotate images and verify its visual perception.

Each tool operates on a named image resource identified by ``image_id``.
Drawing tools create a **new** image resource (leaving the source unchanged)
so the agent can explore different annotation paths.  The original image
always has the fixed ID ``"original"``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel
from pydantic_ai import BinaryContent, RunContext, ToolReturn

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum size (longest side) for auto-resizing single crops
CROP_DISPLAY_SIZE = 512

# Longest-side target for the final side-by-side composite
COMPARE_COMPOSITE_LONG_SIDE = 1024


class ToolCallRecord(BaseModel):
    """Record of a single tool invocation for experiment snapshots."""

    tool: str
    args: dict[str, object]
    image_path: str | None = None  # Relative path to saved result image


@dataclass
class ToolDeps:
    """Mutable state carried across tool calls for one sample.

    Attributes:
        images: Registry of named image resources.  ``"original"`` is always
            present and never mutated.
        original_width: Width of the original image in pixels.
        original_height: Height of the original image in pixels.
        tool_calls: Log of tool calls for experiment snapshot.
        exp_dir: Experiment directory for saving intermediate images.
        sample_index: Sample index for naming saved images.
    """

    images: dict[str, Image.Image]
    original_width: int
    original_height: int
    _next_id: int = field(default=1, repr=False)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    exp_dir: Path | None = None
    sample_index: int = 0

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    def add_image(self, img: Image.Image) -> str:
        """Store *img* under a new sequential ID and return that ID."""
        img_id = f"img_{self._next_id:03d}"
        self._next_id += 1
        self.images[img_id] = img
        return img_id

    def get_image(self, image_id: str) -> Image.Image | None:
        """Return the image for *image_id*, or ``None`` if unknown."""
        return self.images.get(image_id)

    def available_ids(self) -> list[str]:
        """Return a sorted list of all available image IDs."""
        return sorted(self.images.keys())


def create_tool_deps(
    image_path: Path,
    exp_dir: Path | None = None,
    sample_index: int = 0,
) -> ToolDeps:
    """Load an image and create ToolDeps with the original registered.

    Args:
        image_path: Path to the image file.
        exp_dir: Experiment directory for saving intermediate images.
        sample_index: Sample index for naming saved images.

    Returns:
        ToolDeps with ``"original"`` in the image registry.
    """
    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    return ToolDeps(
        images={"original": original},
        original_width=w,
        original_height=h,
        exp_dir=exp_dir,
        sample_index=sample_index,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _image_to_png_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _clamp(val: int, lo: int, hi: int) -> int:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(val, hi))


def _save_tool_image(deps: ToolDeps, img: Image.Image, tool_name: str) -> str | None:
    """Save a tool result image to disk and return its relative path."""
    if deps.exp_dir is None:
        return None
    tool_images_dir = deps.exp_dir / "tool_images"
    tool_images_dir.mkdir(parents=True, exist_ok=True)
    step = len(deps.tool_calls) + 1
    filename = f"sample_{deps.sample_index}_step_{step}_{tool_name}.png"
    img.save(tool_images_dir / filename)
    return f"tool_images/{filename}"


def _resolve_image(deps: ToolDeps, image_id: str) -> tuple[Image.Image, int, int] | str:
    """Look up *image_id* and return ``(image, width, height)`` or an error string."""
    img = deps.get_image(image_id)
    if img is None:
        return f"Unknown image_id '{image_id}'. Available IDs: {deps.available_ids()}"
    w, h = img.size
    return img, w, h


# ------------------------------------------------------------------
# Drawing tools
# ------------------------------------------------------------------


async def draw_line(
    ctx: RunContext[ToolDeps],
    image_id: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    width: int = 3,
) -> ToolReturn:
    """Draw a straight line on an image, creating a NEW image resource.

    The source image is NOT modified.  A copy is made, the line is drawn
    on the copy, and the copy is stored under a new resource ID.

    Args:
        image_id: ID of the source image (e.g. "original" or a previous tool output).
        x1: Start X coordinate (0 = left edge).
        y1: Start Y coordinate (0 = top edge).
        x2: End X coordinate.
        y2: End Y coordinate.
        color: Line color name (e.g. "red", "blue", "green", "yellow", "white").
        width: Line width in pixels (default 3). Use thicker lines for visibility.
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    x2 = _clamp(x2, 0, w - 1)
    y2 = _clamp(y2, 0, h - 1)

    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    new_id = deps.add_image(canvas)
    image_path = _save_tool_image(deps, canvas, "draw_line")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="draw_line",
            args={"image_id": image_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "width": width},
            image_path=image_path,
        )
    )
    logger.debug("draw_line(%s, %d,%d->%d,%d, %s, w=%d) -> %s", image_id, x1, y1, x2, y2, color, width, new_id)

    png_bytes = _image_to_png_bytes(canvas)
    return ToolReturn(
        return_value=(
            f"Drew {color} line from ({x1},{y1}) to ({x2},{y2}) with width={width} "
            f"on '{image_id}'. Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def draw_rectangle(
    ctx: RunContext[ToolDeps],
    image_id: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    width: int = 3,
) -> ToolReturn:
    """Draw a rectangle outline on an image, creating a NEW image resource.

    The source image is NOT modified.

    Args:
        image_id: ID of the source image.
        x1: Left edge X coordinate.
        y1: Top edge Y coordinate.
        x2: Right edge X coordinate.
        y2: Bottom edge Y coordinate.
        color: Outline color name.
        width: Outline width in pixels (default 3).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    x1 = _clamp(min(x1, x2), 0, w - 1)
    y1 = _clamp(min(y1, y2), 0, h - 1)
    x2 = _clamp(max(x1, x2), 0, w - 1)
    y2 = _clamp(max(y1, y2), 0, h - 1)

    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)

    new_id = deps.add_image(canvas)
    image_path = _save_tool_image(deps, canvas, "draw_rectangle")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="draw_rectangle",
            args={"image_id": image_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, "width": width},
            image_path=image_path,
        )
    )
    logger.debug("draw_rectangle(%s, %d,%d,%d,%d, %s) -> %s", image_id, x1, y1, x2, y2, color, new_id)

    png_bytes = _image_to_png_bytes(canvas)
    return ToolReturn(
        return_value=(
            f"Drew {color} rectangle ({x1},{y1})-({x2},{y2}) with width={width} "
            f"on '{image_id}'. Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def draw_circle(
    ctx: RunContext[ToolDeps],
    image_id: str,
    cx: int,
    cy: int,
    radius: int,
    color: str = "red",
    width: int = 3,
) -> ToolReturn:
    """Draw a circle outline on an image, creating a NEW image resource.

    The source image is NOT modified.

    Args:
        image_id: ID of the source image.
        cx: Center X coordinate.
        cy: Center Y coordinate.
        radius: Circle radius in pixels.
        color: Outline color name.
        width: Outline width in pixels (default 3).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    cx = _clamp(cx, 0, w - 1)
    cy = _clamp(cy, 0, h - 1)
    radius = max(1, radius)

    bx1 = _clamp(cx - radius, 0, w - 1)
    by1 = _clamp(cy - radius, 0, h - 1)
    bx2 = _clamp(cx + radius, 0, w - 1)
    by2 = _clamp(cy + radius, 0, h - 1)

    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)
    draw.ellipse([(bx1, by1), (bx2, by2)], outline=color, width=width)

    new_id = deps.add_image(canvas)
    image_path = _save_tool_image(deps, canvas, "draw_circle")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="draw_circle",
            args={"image_id": image_id, "cx": cx, "cy": cy, "radius": radius, "color": color, "width": width},
            image_path=image_path,
        )
    )
    logger.debug("draw_circle(%s, %d,%d, r=%d, %s) -> %s", image_id, cx, cy, radius, color, new_id)

    png_bytes = _image_to_png_bytes(canvas)
    return ToolReturn(
        return_value=(
            f"Drew {color} circle at ({cx},{cy}) radius={radius} with width={width} "
            f"on '{image_id}'. Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


# ------------------------------------------------------------------
# Inspection tools
# ------------------------------------------------------------------


async def crop(
    ctx: RunContext[ToolDeps],
    image_id: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> ToolReturn:
    """Crop a rectangular region from an image and enlarge it for inspection.

    The cropped region is upscaled so the longest side is at least 512 px.
    The result is stored as a new image resource.

    Args:
        image_id: ID of the source image (e.g. "original" or any annotated version).
        x1: Left edge of crop box.
        y1: Top edge of crop box.
        x2: Right edge of crop box.
        y2: Bottom edge of crop box.
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    x1 = _clamp(min(x1, x2), 0, w - 1)
    y1 = _clamp(min(y1, y2), 0, h - 1)
    x2 = _clamp(max(x1, x2), 0, w - 1)
    y2 = _clamp(max(y1, y2), 0, h - 1)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return ToolReturn(
            return_value="Crop region too small (< 10px). Please specify a larger region.",
        )

    cropped = source.crop((x1, y1, x2, y2))
    cw, ch = cropped.size

    # Upscale so the model can see details
    scale = max(1.0, CROP_DISPLAY_SIZE / max(cw, ch))
    if scale > 1:
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_id = deps.add_image(cropped)
    image_path = _save_tool_image(deps, cropped, "crop")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="crop",
            args={"image_id": image_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
            image_path=image_path,
        )
    )
    logger.debug("crop(%s, %d,%d,%d,%d) -> %s (%s)", image_id, x1, y1, x2, y2, new_id, cropped.size)

    png_bytes = _image_to_png_bytes(cropped)
    return ToolReturn(
        return_value=(
            f"Cropped region ({x1},{y1})-({x2},{y2}) from '{image_id}', "
            f"original crop size {cw}x{ch}, displayed at {cropped.size[0]}x{cropped.size[1]}. "
            f"Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def compare_crops(
    ctx: RunContext[ToolDeps],
    image_id: str,
    x1a: int,
    y1a: int,
    x2a: int,
    y2a: int,
    x1b: int,
    y1b: int,
    x2b: int,
    y2b: int,
    label_a: str = "A",
    label_b: str = "B",
) -> ToolReturn:
    """Crop two regions and display them SIDE BY SIDE for direct comparison.

    Both crops are placed at NATIVE resolution (preserving true size
    relationships), then the composite is scaled so the longest side is
    1024 px.  If the crops differ in height, the shorter one gets white
    padding below.  Horizontal grid lines are drawn for visual reference.

    IMPORTANT: Use the SAME crop-box dimensions for both regions when
    comparing sizes — this ensures a fair comparison.

    The result is stored as a new image resource.

    Args:
        image_id: ID of the source image.
        x1a: Left edge of crop region A.
        y1a: Top edge of crop region A.
        x2a: Right edge of crop region A.
        y2a: Bottom edge of crop region A.
        x1b: Left edge of crop region B.
        y1b: Top edge of crop region B.
        x2b: Right edge of crop region B.
        y2b: Bottom edge of crop region B.
        label_a: Label for the left panel (default "A").
        label_b: Label for the right panel (default "B").
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    # Clamp coordinates
    x1a = _clamp(min(x1a, x2a), 0, w - 1)
    y1a = _clamp(min(y1a, y2a), 0, h - 1)
    x2a = _clamp(max(x1a, x2a), 0, w - 1)
    y2a = _clamp(max(y1a, y2a), 0, h - 1)

    x1b = _clamp(min(x1b, x2b), 0, w - 1)
    y1b = _clamp(min(y1b, y2b), 0, h - 1)
    x2b = _clamp(max(x1b, x2b), 0, w - 1)
    y2b = _clamp(max(y1b, y2b), 0, h - 1)

    if x2a - x1a < 10 or y2a - y1a < 10 or x2b - x1b < 10 or y2b - y1b < 10:
        return ToolReturn(
            return_value="One or both crop regions too small (< 10px). Please specify larger regions.",
        )

    crop_a = source.crop((x1a, y1a, x2a, y2a))
    crop_b = source.crop((x1b, y1b, x2b, y2b))

    # Place side-by-side at native resolution (no independent resizing).
    gap = 4
    canvas_h = max(crop_a.height, crop_b.height)
    total_w = crop_a.width + gap + crop_b.width
    composite = Image.new("RGB", (total_w, canvas_h), color=(255, 255, 255))
    composite.paste(crop_a, (0, 0))
    composite.paste(crop_b, (crop_a.width + gap, 0))

    # Resize the final composite so longest side = COMPARE_COMPOSITE_LONG_SIDE
    long_side = max(composite.size)
    scale = COMPARE_COMPOSITE_LONG_SIDE / long_side
    if scale != 1.0:
        new_w = max(1, int(composite.width * scale))
        new_h = max(1, int(composite.height * scale))
        composite = composite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Draw divider line and grid references
    comp_draw = ImageDraw.Draw(composite)
    div_x = int(crop_a.width * scale) + int(gap * scale) // 2
    comp_draw.line([(div_x, 0), (div_x, composite.height - 1)], fill=(180, 180, 180), width=2)

    grid_color = (220, 220, 220)
    grid_step = composite.height // 6
    if grid_step > 0:
        for gy in range(grid_step, composite.height, grid_step):
            comp_draw.line([(0, gy), (composite.width - 1, gy)], fill=grid_color, width=1)

    new_id = deps.add_image(composite)
    image_path = _save_tool_image(deps, composite, "compare_crops")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="compare_crops",
            args={
                "image_id": image_id,
                "x1a": x1a,
                "y1a": y1a,
                "x2a": x2a,
                "y2a": y2a,
                "x1b": x1b,
                "y1b": y1b,
                "x2b": x2b,
                "y2b": y2b,
            },
            image_path=image_path,
        )
    )
    logger.debug(
        "compare_crops(%s, A=%d,%d,%d,%d B=%d,%d,%d,%d) -> %s",
        image_id,
        x1a,
        y1a,
        x2a,
        y2a,
        x1b,
        y1b,
        x2b,
        y2b,
        new_id,
    )

    size_a = f"{crop_a.width}x{crop_a.height}"
    size_b = f"{crop_b.width}x{crop_b.height}"
    png_bytes = _image_to_png_bytes(composite)
    return ToolReturn(
        return_value=(
            f"Side-by-side comparison from '{image_id}': "
            f"{label_a}=({x1a},{y1a})-({x2a},{y2a}) [{size_a}] | "
            f"{label_b}=({x1b},{y1b})-({x2b},{y2b}) [{size_b}]. "
            f"Crops placed at native resolution, composite scaled to longest side "
            f"= {COMPARE_COMPOSITE_LONG_SIDE}px. "
            f"{label_a} is on the LEFT, {label_b} is on the RIGHT. "
            f"Size differences reflect TRUE pixel-size differences. "
            f"Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


# ------------------------------------------------------------------
# Grid / channel / color tools (Task 2)
# ------------------------------------------------------------------


def _get_label_font(cell_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a font sized appropriately for grid cell labels.

    Tries to load a TrueType font; falls back to the PIL default bitmap font.
    """
    target = max(10, min(cell_size // 3, 24))
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", target)
    except OSError:
        return ImageFont.load_default()


async def overlay_grid(
    ctx: RunContext[ToolDeps],
    image_id: str,
    rows: int = 3,
    cols: int = 3,
    color: str = "red",
    width: int = 2,
) -> ToolReturn:
    """Overlay a labeled row/col grid on an image, creating a NEW image resource.

    Each cell is labeled ``(r,c)`` (1-indexed) so you can refer to specific
    grid cells in your reasoning.  Useful for odd-one-out and counting tasks.

    Args:
        image_id: ID of the source image.
        rows: Number of rows in the grid (clamped to [1, 20]).
        cols: Number of columns in the grid (clamped to [1, 20]).
        color: Grid line color name (e.g. "red", "cyan", "yellow").
        width: Grid line width in pixels (default 2).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    rows = _clamp(rows, 1, 20)
    cols = _clamp(cols, 1, 20)

    canvas = source.copy()
    draw = ImageDraw.Draw(canvas)

    cell_w = w / cols
    cell_h = h / rows
    font = _get_label_font(int(min(cell_w, cell_h)))

    # Draw horizontal lines
    for r in range(1, rows):
        y = int(r * cell_h)
        draw.line([(0, y), (w - 1, y)], fill=color, width=width)

    # Draw vertical lines
    for c in range(1, cols):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, h - 1)], fill=color, width=width)

    # Label each cell with (r,c)
    for r in range(rows):
        for c in range(cols):
            label = f"({r + 1},{c + 1})"
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), label, fill=color, font=font)

    new_id = deps.add_image(canvas)
    image_path = _save_tool_image(deps, canvas, "overlay_grid")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="overlay_grid",
            args={"image_id": image_id, "rows": rows, "cols": cols, "color": color, "width": width},
            image_path=image_path,
        )
    )
    logger.debug("overlay_grid(%s, %dx%d, %s) -> %s", image_id, rows, cols, color, new_id)

    png_bytes = _image_to_png_bytes(canvas)
    return ToolReturn(
        return_value=(
            f"Overlaid {rows}x{cols} grid on '{image_id}' with {color} lines. "
            f"Cells are labeled (row,col) from (1,1) at top-left to ({rows},{cols}) at bottom-right. "
            f"Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def extract_channel(
    ctx: RunContext[ToolDeps],
    image_id: str,
    channel: str = "red",
) -> ToolReturn:
    """Extract a single color channel from an image, creating a NEW image resource.

    Converts the selected channel to a grayscale-as-RGB image so the model
    can view it.  Useful for reading hidden content in Ishihara-style images.

    Args:
        image_id: ID of the source image.
        channel: One of "red", "green", "blue", or "grayscale".
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    channel = channel.lower().strip()
    valid_channels = {"red", "green", "blue", "grayscale"}
    if channel not in valid_channels:
        return ToolReturn(return_value=f"Invalid channel '{channel}'. Must be one of: {sorted(valid_channels)}")

    if channel == "grayscale":
        gray = source.convert("L")
        result = gray.convert("RGB")
    else:
        r, g, b = source.split()
        channel_map = {"red": r, "green": g, "blue": b}
        single = channel_map[channel]
        result = single.convert("RGB")

    new_id = deps.add_image(result)
    image_path = _save_tool_image(deps, result, f"extract_{channel}")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="extract_channel",
            args={"image_id": image_id, "channel": channel},
            image_path=image_path,
        )
    )
    logger.debug("extract_channel(%s, %s) -> %s", image_id, channel, new_id)

    png_bytes = _image_to_png_bytes(result)
    return ToolReturn(
        return_value=(f"Extracted '{channel}' channel from '{image_id}' ({w}x{h}). Result saved as '{new_id}'."),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def sample_color(
    ctx: RunContext[ToolDeps],
    image_id: str,
    x: int,
    y: int,
    radius: int = 5,
) -> ToolReturn:
    """Sample the average color of a small patch around a pixel coordinate.

    Returns text only (no image) with the RGB values.  Useful for color
    comparison questions where you need precise color measurements.

    Args:
        image_id: ID of the source image.
        x: X coordinate of the sample center.
        y: Y coordinate of the sample center.
        radius: Half-size of the sampling square (default 5 → 11x11 patch).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    x = _clamp(x, 0, w - 1)
    y = _clamp(y, 0, h - 1)
    radius = max(1, radius)

    x1 = _clamp(x - radius, 0, w - 1)
    y1 = _clamp(y - radius, 0, h - 1)
    x2 = _clamp(x + radius, 0, w - 1)
    y2 = _clamp(y + radius, 0, h - 1)

    arr = np.array(source)
    patch = arr[y1 : y2 + 1, x1 : x2 + 1]  # shape: (patch_h, patch_w, 3)
    mean_rgb = patch.mean(axis=(0, 1))
    r_val, g_val, b_val = int(round(mean_rgb[0])), int(round(mean_rgb[1])), int(round(mean_rgb[2]))

    deps.tool_calls.append(
        ToolCallRecord(
            tool="sample_color",
            args={"image_id": image_id, "x": x, "y": y, "radius": radius},
            image_path=None,
        )
    )
    logger.debug("sample_color(%s, %d,%d, r=%d) -> R=%d G=%d B=%d", image_id, x, y, radius, r_val, g_val, b_val)

    return ToolReturn(
        return_value=(
            f"Color at ({x},{y}) in '{image_id}' "
            f"(averaged over {x2 - x1 + 1}x{y2 - y1 + 1} patch): "
            f"R={r_val}, G={g_val}, B={b_val}"
        ),
    )


async def isolate_color(
    ctx: RunContext[ToolDeps],
    image_id: str,
    target_color: str = "brown",
) -> ToolReturn:
    """Isolate pixels matching a named color family, showing them as black on white.

    Pixels matching the target color become **black**; everything else becomes
    **white**.  This is the BEST tool for reading hidden content in Ishihara
    color-blindness test images — isolate the dot color that forms the hidden
    number or word.

    Args:
        image_id: ID of the source image.
        target_color: Color family to isolate. One of:
            "brown" — warm brown/orange dots (R high, G/B low).
            "red" — red-dominant pixels.
            "green" — green-dominant pixels.
            "blue" — blue-dominant pixels.
            "orange" — orange/amber pixels.
            "purple" — purple/violet pixels.
            "gray" — neutral gray pixels (low saturation).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    arr = np.array(source, dtype=np.float32)
    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    target_color = target_color.lower().strip()

    # Define masks for each color family
    masks: dict[str, np.ndarray] = {
        "brown": (r_ch > 120) & (g_ch < 140) & (b_ch < 110) & (r_ch > g_ch),
        "red": (r_ch > 140) & (r_ch > g_ch * 1.5) & (r_ch > b_ch * 1.5),
        "green": (g_ch > 120) & (g_ch > r_ch * 1.2) & (g_ch > b_ch * 1.2),
        "blue": (b_ch > 120) & (b_ch > r_ch * 1.2) & (b_ch > g_ch * 1.2),
        "orange": (r_ch > 160) & (g_ch > 80) & (g_ch < 180) & (b_ch < 100),
        "purple": (r_ch > 80) & (b_ch > 80) & (g_ch < r_ch * 0.7) & (b_ch > g_ch),
        "gray": (np.abs(r_ch - g_ch) < 30) & (np.abs(g_ch - b_ch) < 30) & (r_ch > 60) & (r_ch < 200),
    }

    if target_color not in masks:
        return ToolReturn(return_value=f"Unknown target_color '{target_color}'. Must be one of: {sorted(masks.keys())}")

    mask = masks[target_color]
    result_arr = np.full((h, w, 3), 255, dtype=np.uint8)
    result_arr[mask] = [0, 0, 0]
    result = Image.fromarray(result_arr)

    matched_pct = float(mask.sum()) / (w * h) * 100

    new_id = deps.add_image(result)
    image_path = _save_tool_image(deps, result, f"isolate_{target_color}")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="isolate_color",
            args={"image_id": image_id, "target_color": target_color},
            image_path=image_path,
        )
    )
    logger.debug(
        "isolate_color(%s, %s) -> %s (%.1f%% matched)",
        image_id,
        target_color,
        new_id,
        matched_pct,
    )

    png_bytes = _image_to_png_bytes(result)
    return ToolReturn(
        return_value=(
            f"Isolated '{target_color}' pixels from '{image_id}' ({w}x{h}). "
            f"{matched_pct:.1f}% of pixels matched. "
            f"Matching pixels are BLACK, rest is WHITE. "
            f"Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )


async def blur(
    ctx: RunContext[ToolDeps],
    image_id: str,
    radius: int = 30,
    contrast_stretch: bool = True,
) -> ToolReturn:
    """Apply heavy Gaussian blur to reveal hidden large-scale patterns.

    Fine details (lines, dots, textures) merge into a smooth density map,
    revealing hidden portraits, faces, or shapes encoded in the pattern
    density.  With ``contrast_stretch=True`` (default), the result is
    contrast-stretched to maximize visibility.

    This is the BEST tool for finding hidden people/faces in patterned images.

    Args:
        image_id: ID of the source image.
        radius: Gaussian blur radius in pixels (default 30). Larger values
            reveal coarser hidden patterns. Try 20-60.
        contrast_stretch: If True, stretch output to full black-white range
            for maximum visibility (default True).
    """
    deps = ctx.deps
    resolved = _resolve_image(deps, image_id)
    if isinstance(resolved, str):
        return ToolReturn(return_value=resolved)
    source, w, h = resolved

    radius = max(1, min(radius, 200))

    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))

    if contrast_stretch:
        gray = blurred.convert("L")
        arr = np.array(gray, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 1e-3:
            arr = (arr - lo) / (hi - lo) * 255.0
        result = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        result = blurred

    new_id = deps.add_image(result)
    image_path = _save_tool_image(deps, result, "blur")
    deps.tool_calls.append(
        ToolCallRecord(
            tool="blur",
            args={
                "image_id": image_id,
                "radius": radius,
                "contrast_stretch": contrast_stretch,
            },
            image_path=image_path,
        )
    )
    logger.debug("blur(%s, r=%d, stretch=%s) -> %s", image_id, radius, contrast_stretch, new_id)

    stretch_note = " with contrast stretch" if contrast_stretch else ""
    png_bytes = _image_to_png_bytes(result)
    return ToolReturn(
        return_value=(
            f"Applied Gaussian blur (radius={radius}){stretch_note} "
            f"to '{image_id}' ({w}x{h}). "
            f"Fine details are smoothed — look for large-scale hidden patterns. "
            f"Result saved as '{new_id}'."
        ),
        content=[BinaryContent(data=png_bytes, media_type="image/png")],
    )
