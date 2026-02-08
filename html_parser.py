#!/usr/bin/env python3
"""
HTML-парсер для папки output/ru (сгенерированная документация Yandex Cloud).
Обходит все HTML-файлы, извлекает title, canonical, alternate и контент из
diplodoc-state/data/html (в формате Markdown). Результат — один JSON-файл на страницу.
Работает в многопоточном режиме (по умолчанию 3 потока).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from html import unescape
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Установите beautifulsoup4: pip install beautifulsoup4")

try:
    import html2text
except ImportError:
    raise ImportError("Установите html2text: pip install html2text")


DEFAULT_ROOT = Path(__file__).resolve().parent / "output" / "ru"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output_ru_parsed"
DEFAULT_WORKERS = 3
SKIP_DIRS = {"_assets", "_templates", "_bundle"}


def _extract_diplodoc_state(soup: BeautifulSoup) -> dict | None:
    """Извлекает JSON из script#diplodoc-state."""
    script = soup.find("script", id="diplodoc-state", type="application/json")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


def _html_to_md(html: str) -> str:
    """Конвертирует HTML-строку в Markdown."""
    if not html or not html.strip():
        return ""
    h2t = html2text.HTML2Text()
    h2t.ignore_links = False
    h2t.ignore_images = False
    h2t.body_width = 0
    return h2t.handle(html).strip()


def parse_one_html(file_path: Path, root: Path) -> dict[str, Any] | None:
    """
    Парсит один HTML-файл. Возвращает словарь для JSON:
    { "title": str, "canonical": str, "alternate": str, "text": str }
    """
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    soup = BeautifulSoup(raw, "html.parser")

    # title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # canonical
    canonical_tag = soup.find("link", rel="canonical")
    canonical = (canonical_tag.get("href") or "").strip() if canonical_tag else ""

    # alternate (первая ссылка rel="alternate")
    alternate_tag = soup.find("link", rel="alternate")
    alternate = (alternate_tag.get("href") or "").strip() if alternate_tag else ""

    # data/html из diplodoc-state → Markdown
    text = ""
    state = _extract_diplodoc_state(soup)
    if state:
        data = state.get("data") or {}
        raw_html = data.get("html") or ""
        if raw_html:
            unescaped = unescape(raw_html)
            text = _html_to_md(unescaped)

    return {
        "title": title,
        "canonical": canonical,
        "alternate": alternate,
        "text": text,
    }


def walk_html(root: Path):
    """Рекурсивно обходит все .html в root, пропуская SKIP_DIRS."""
    root = Path(root).resolve()
    for path in root.rglob("*.html"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def process_file(
    html_path: Path,
    root: Path,
    out_dir: Path,
) -> tuple[Path, bool, str | None]:
    """
    Обрабатывает один HTML-файл и записывает JSON в out_dir.
    Возвращает (html_path, success, error_message).
    """
    try:
        data = parse_one_html(html_path, root)
        if data is None:
            return (html_path, False, "parse_one_html returned None")

        rel = html_path.relative_to(root)
        out_path = out_dir / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (html_path, True, None)
    except Exception as e:
        return (html_path, False, str(e))


def run(
    root: Path = DEFAULT_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = DEFAULT_WORKERS,
) -> tuple[int, int]:
    """
    Обходит все HTML в root, в многопоточном режиме пишет JSON в out_dir.
    Возвращает (успешно, ошибок).
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(walk_html(root))
    ok, err = 0, 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, p, root, out_dir): p for p in files
        }
        for future in as_completed(futures):
            path, success, msg = future.result()
            if success:
                ok += 1
            else:
                err += 1
                if msg:
                    print(f"Ошибка {path}: {msg}", file=__import__("sys").stderr)

    return ok, err


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Парсер HTML из output/ru: title, canonical, alternate, text (MD). Один JSON на страницу."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help="Корневая папка с HTML (по умолчанию: output/ru)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUT_DIR),
        help="Папка для JSON-файлов (по умолчанию: output_ru_parsed)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Число потоков (по умолчанию: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    ok, err = run(root=Path(args.root), out_dir=Path(args.output), workers=args.jobs)
    print(f"Готово: {ok} записано, {err} ошибок", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
