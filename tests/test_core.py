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


def test_index_flow_keeps_paths_aligned_when_decode_fails_mid_batch(
    tiny_image_dir: Path, monkeypatch
):
    """Files that pass verify_image but fail at decode time used to corrupt the
    db: safe_collate would drop the failed item from a batch, but the encoder
    didn't tell the indexer which item was dropped, so subsequent files got
    tagged with the wrong embedding (and the last file would IndexError).

    Simulate that case by monkey-patching pil_loader to return None for a
    specific file. The resulting db must contain only successfully-decoded
    files, with paths correctly correlated to their embeddings.
    """
    import torch

    from memery import crafter

    real_loader = crafter.pil_loader

    def flaky_loader(path: str):
        if "Wholesome-Meme-1.jpg" in path:
            return None
        return real_loader(path)

    monkeypatch.setattr(crafter, "pil_loader", flaky_loader)

    m = Memery(root=str(tiny_image_dir))
    db_path, _ = m.index_flow(str(tiny_image_dir))

    db = torch.load(db_path, map_location="cpu", weights_only=False)
    paths = [entry["fpath"] for entry in db.values()]

    # The deliberately-flaky file must NOT appear in the db
    assert not any("Wholesome-Meme-1.jpg" in p for p in paths)
    # The other fixtures must still be there
    assert any("Wholesome-Meme-68.jpg" in p for p in paths)
    assert any("cute-dog-with-cupcake" in p for p in paths)
    # And every entry must have a 512-dim embedding (i.e. nothing got
    # IndexError'd into a half-built state)
    for entry in db.values():
        assert entry["embed"].shape == (512,)
