#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html as html_lib
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


DEFAULT_URL = "https://www.povertystoplight.org/es/faq"
DEFAULT_OUTPUT = Path("data/faq_povertystoplight_es.md")


def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.text


def extract_main_html(page_html: str) -> str:
    soup = BeautifulSoup(page_html, "html.parser")

    collapsibles = soup.select("div.Collapsible")
    if collapsibles:
        parts: list[str] = []
        for block in collapsibles:
            trigger = block.select_one(".Collapsible__trigger")
            content = block.select_one(".collapsible__content")
            if not content:
                continue

            if trigger:
                title = " ".join(trigger.stripped_strings)
                if title:
                    parts.append(f"<h2>{html_lib.escape(title)}</h2>")

            parts.append(str(content))

        if parts:
            return "<div>" + "\n".join(parts) + "</div>"

    candidates = [
        "main",
        "article",
        "section",
        "div.faq",
        "div[class*='faq']",
        "div[id*='faq']",
    ]

    for selector in candidates:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            return str(node)

    return str(soup.body or soup)


def html_to_markdown(html_fragment: str) -> str:
    markdown = md(
        html_fragment,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "noscript"],
    )
    markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
    return markdown.strip() + "\n"


def save_markdown(content: str, output_file: Path, source_url: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = f"# FAQ - Poverty Stoplight (ES)\n\nFuente: {source_url}\n\n"
    output_file.write_text(header + content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrapea una página y la guarda en formato Markdown."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL objetivo (por defecto: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Archivo de salida Markdown (por defecto: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    html = fetch_html(args.url)
    main_html = extract_main_html(html)
    markdown = html_to_markdown(main_html)
    save_markdown(markdown, output_path, args.url)

    print(f"Markdown guardado en: {output_path}")


if __name__ == "__main__":
    main()