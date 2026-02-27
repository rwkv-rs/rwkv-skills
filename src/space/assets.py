"""Static asset loading — CSS, vendor CSS, vendor JS head tags."""

from __future__ import annotations

import re
from pathlib import Path


def _load_css() -> tuple[str, str | None]:
    style_path = Path(__file__).parent / "styles" / "space.css"
    if not style_path.exists():
        warning = f"未找到样式文件：{style_path}"
        print(f"[space] {warning}")
        return "", warning
    try:
        return style_path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载样式：读取 {style_path.name} 失败 ({exc})"
        print(f"[space] {warning}")
        return "", warning


def _load_vendor_css() -> tuple[str, str | None]:
    """Load vendor CSS for context modal rendering (markdown, code, math).

    Gradio content lives inside a Shadow DOM — CSS must be injected via the
    ``Blocks(css=...)`` parameter to apply inside that tree.
    """
    vendor_dir = Path(__file__).parent / "assets" / "vendor"
    if not vendor_dir.exists():
        warning = f"未找到前端依赖目录：{vendor_dir}"
        print(f"[space] {warning}")
        return "", warning

    fonts_dir = vendor_dir / "fonts"
    fonts_url_prefix = f"/file={fonts_dir.as_posix()}/"

    def read_text(name: str) -> str:
        return (vendor_dir / name).read_text(encoding="utf-8")

    try:
        katex_css = read_text("katex.min.css")
        katex_css = re.sub(
            r',url\(fonts/[^)]+?\.woff\) format\("woff"\),url\(fonts/[^)]+?\.ttf\) format\("truetype"\)',
            "",
            katex_css,
        )
        katex_css = katex_css.replace("url(fonts/", f"url({fonts_url_prefix}")
        hljs_css = read_text("github-dark.min.css")

        warning = None
        if not fonts_dir.exists():
            warning = f"未找到 KaTeX 字体目录：{fonts_dir}（数学公式可能无法正常显示）"
            print(f"[space] {warning}")

        css = "\n".join(
            [
                "/* Vendor: highlight.js theme */",
                hljs_css,
                "/* Vendor: KaTeX */",
                katex_css,
            ]
        )
        return css, warning
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载前端渲染依赖：{exc}"
        print(f"[space] {warning}")
        return "", warning


def _load_vendor_head() -> tuple[str, str | None]:
    """Load vendor JS for context modal rendering (markdown, code, math)."""
    vendor_dir = Path(__file__).parent / "assets" / "vendor"
    if not vendor_dir.exists():
        warning = f"未找到前端依赖目录：{vendor_dir}"
        print(f"[space] {warning}")
        return "", warning

    def read_js(name: str) -> str:
        return (vendor_dir / name).read_text(encoding="utf-8")

    try:
        head_parts = [
            f"<script id=\"space-vendor-katex-js\">{read_js('katex.min.js')}</script>",
            f"<script id=\"space-vendor-katex-auto-render\">{read_js('auto-render.min.js')}</script>",
            f"<script id=\"space-vendor-markdown-it\">{read_js('markdown-it.min.js')}</script>",
            f"<script id=\"space-vendor-highlight-js\">{read_js('highlight.min.js')}</script>",
        ]
        return "\n".join(head_parts), None
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载前端渲染依赖：{exc}"
        print(f"[space] {warning}")
        return "", warning
