#!/usr/bin/env python3
"""Recreate the cerebras.pytorch source tree from public docs viewcode pages."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_START_URL = (
    "https://training-api.cerebras.ai/en/latest/wsc/api/cerebras_pytorch/index.html"
)
DEFAULT_API_PREFIX = "https://training-api.cerebras.ai/en/latest/wsc/api/cerebras_pytorch/"
DEFAULT_MODULE_INDEX_URL = "https://training-api.cerebras.ai/en/latest/_modules/index.html"
DEFAULT_OUTPUT_ROOT = Path(".")

HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)
PRE_BLOCK_RE = re.compile(
    r"""<div class="highlight"><pre>(.*?)</pre></div>""",
    re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")


def fetch_text(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "cerebras-pytorch-src-recreator/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310
            encoding = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(encoding, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc}") from exc


def extract_links(page_text: str) -> list[str]:
    return HREF_RE.findall(page_text)


def normalize_url(base: str, href: str) -> str:
    absolute = urljoin(base, href)
    normalized, _fragment = urldefrag(absolute)
    return normalized


def discover_api_pages(start_url: str, api_prefix: str) -> list[str]:
    queue: list[str] = [start_url]
    seen: set[str] = set()

    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        page = fetch_text(url)
        for href in extract_links(page):
            candidate = normalize_url(url, href)
            if not candidate.endswith(".html"):
                continue
            if candidate.startswith(api_prefix) and candidate not in seen:
                queue.append(candidate)

    return sorted(seen)


def discover_module_urls(api_pages: list[str], module_index_url: str) -> list[str]:
    module_urls: set[str] = set()

    source_pages = list(api_pages)
    if module_index_url not in source_pages:
        source_pages.append(module_index_url)

    for page_url in source_pages:
        page = fetch_text(page_url)
        for href in extract_links(page):
            candidate = normalize_url(page_url, href)
            if (
                "/_modules/cerebras/pytorch/" in candidate
                and candidate.endswith(".html")
            ):
                module_urls.add(candidate)

    return sorted(module_urls)


def extract_python_source(module_page_html: str, module_url: str) -> str:
    match = PRE_BLOCK_RE.search(module_page_html)
    if not match:
        raise RuntimeError(f"Could not find source <pre> block in {module_url}")

    highlighted = match.group(1)
    stripped = TAG_RE.sub("", highlighted)
    code = html.unescape(stripped)
    cleaned_lines = []
    for line in code.splitlines():
        if line.startswith("[docs]"):
            line = line[len("[docs]") :]
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).rstrip() + "\n"
    return result


def module_url_to_stem(module_url: str) -> str:
    path = urlparse(module_url).path
    marker = "/_modules/"
    if marker not in path:
        raise RuntimeError(f"Unexpected module URL: {module_url}")
    rel = path.split(marker, maxsplit=1)[1]
    return str(Path(rel).with_suffix(""))


def stem_to_relative_path(stem: str, package_stems: set[str]) -> Path:
    if stem in package_stems:
        return Path(stem) / "__init__.py"
    return Path(f"{stem}.py")


def write_module(output_root: Path, relative_path: Path, source: str) -> Path:
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    return relative_path


def ensure_package_markers(output_root: Path, module_paths: list[Path]) -> int:
    package_dirs: set[Path] = set()
    for module_path in module_paths:
        parent = module_path.parent
        while parent != Path("."):
            package_dirs.add(parent)
            parent = parent.parent

    created = 0
    for package_dir in sorted(package_dirs):
        init_file = output_root / package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            created += 1
    return created


def cleanup_package_file_conflicts(output_root: Path, package_stems: set[str]) -> int:
    removed = 0
    for stem in sorted(package_stems):
        legacy_module_file = output_root / f"{stem}.py"
        if legacy_module_file.exists():
            legacy_module_file.unlink()
            removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the recreated package tree should be written.",
    )
    parser.add_argument(
        "--start-url",
        default=DEFAULT_START_URL,
        help="Starting Cerebras PyTorch API docs page.",
    )
    parser.add_argument(
        "--api-prefix",
        default=DEFAULT_API_PREFIX,
        help="Prefix for API pages to crawl recursively.",
    )
    parser.add_argument(
        "--module-index-url",
        default=DEFAULT_MODULE_INDEX_URL,
        help="Sphinx module index URL for exhaustive module link discovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()

    api_pages = discover_api_pages(args.start_url, args.api_prefix)
    module_urls = discover_module_urls(api_pages, args.module_index_url)

    if not module_urls:
        raise RuntimeError("No cerebras.pytorch module URLs were discovered.")

    stems_by_url = {module_url: module_url_to_stem(module_url) for module_url in module_urls}
    stems = set(stems_by_url.values())
    package_stems = {
        stem for stem in stems if any(other.startswith(f"{stem}/") for other in stems)
    }

    written_paths: list[Path] = []
    for module_url in module_urls:
        module_page = fetch_text(module_url)
        source = extract_python_source(module_page, module_url)
        relative_path = stem_to_relative_path(stems_by_url[module_url], package_stems)
        relative_path = write_module(output_root, relative_path, source)
        written_paths.append(relative_path)

    removed_conflicts = cleanup_package_file_conflicts(output_root, package_stems)
    created_inits = ensure_package_markers(output_root, written_paths)

    print(f"Discovered API pages: {len(api_pages)}")
    print(f"Discovered module pages: {len(module_urls)}")
    print(f"Wrote module files: {len(written_paths)}")
    print(f"Removed stale module files: {removed_conflicts}")
    print(f"Created __init__.py files: {created_inits}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
