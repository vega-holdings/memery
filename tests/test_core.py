"""End-to-end tests for the index/query pipeline.

Marked ``integration`` because they download CLIP weights on first run
(~338MB) and run the model. Skipped in default test runs; opt in with
``pytest -m integration``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memery.core import Memery

pytestmark = pytest.mark.integration


def test_index_flow_writes_archives(tiny_image_dir: Path):
    m = Memery(root=str(tiny_image_dir))
    db_path, tree_path = m.index_flow(str(tiny_image_dir))
    assert Path(db_path).exists()
    assert Path(tree_path).exists()


def test_query_flow_returns_ranked_paths(tiny_image_dir: Path):
    m = Memery(root=str(tiny_image_dir))
    ranked = m.query_flow(str(tiny_image_dir), query="dog")
    assert isinstance(ranked, list)
    assert len(ranked) >= 1
    # the cupcake-dog fixture should rank above the wholesome-meme texts
    assert "cute-dog-with-cupcake-P9E2YL5-min.jpg" in ranked[0]


def test_query_flow_no_query_returns_empty(tiny_image_dir: Path):
    m = Memery(root=str(tiny_image_dir))
    out = m.query_flow(str(tiny_image_dir))
    # current behaviour: returns empty string when there's nothing to search for
    assert out == "" or out == []


def test_clean_removes_index_files(tiny_image_dir: Path):
    m = Memery(root=str(tiny_image_dir))
    m.index_flow(str(tiny_image_dir))
    assert (tiny_image_dir / "memery.ann").exists()
    m.clean(str(tiny_image_dir))
    assert not (tiny_image_dir / "memery.ann").exists()
    assert not (tiny_image_dir / "memery.pt").exists()
