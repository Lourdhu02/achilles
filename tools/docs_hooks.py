"""MkDocs hooks for the Achilles docs site (loaded from mkdocs.yml).

GitHub and Python-Markdown build heading anchors differently ("Lab 06 — Kernels" is
#lab-06--kernels on GitHub and #lab-06-kernels in Python-Markdown). The repo's links are
written for GitHub, so the site uses GitHub's rule: the same one tools/check_links.py uses.
"""

from __future__ import annotations

import re


def gfm_slugify(value: str, separator: str = "-") -> str:
    text = re.sub(r"[^\w\- ]", "", value.strip().lower())
    return text.replace(" ", separator)


def on_config(config, **kwargs):
    config.mdx_configs.setdefault("toc", {})["slugify"] = gfm_slugify
    return config
