"""Unit tests for memery.loader — no CLIP, no torch model loading."""
from __future__ import annotations

from pathlib import Path

from memery import loader


def test_hash_path_combines_stem_and_mtime(tmp_path: Path):
    f = tmp_path / "hello.jpg"
    f.write_bytes(b"\x00")
    h = loader.hash_path(f)
    assert h.startswith("hello_")
    # mtime portion is everything after the underscore and is an integer-ish string
    _, mtime = h.split("_", 1)
    assert mtime.isdigit()


def test_get_image_files_filters_extensions(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"\x00")
    (tmp_path / "b.PNG").write_bytes(b"\x00")  # uppercase suffix is NOT matched
    (tmp_path / "notes.txt").write_text("nope")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.png").write_bytes(b"\x00")

    paths = loader.get_image_files(tmp_path)
    names = sorted(p.name for p, _ in paths)
    # current behaviour: extension matching is case-sensitive on the suffix set
    assert "a.jpg" in names
    assert "c.png" in names
    assert "notes.txt" not in names


def test_get_valid_images_skips_corrupt(tiny_image_dir: Path):
    valid = loader.get_valid_images(tiny_image_dir)
    names = {Path(p).name for p, _ in valid}
    # corrupt fixture must be excluded
    assert "corrupted-file.jpeg" not in names
    # the three real fixtures must all be there
    assert "Wholesome-Meme-68.jpg" in names
    assert "Wholesome-Meme-1.jpg" in names
    assert "cute-dog-with-cupcake-P9E2YL5-min.jpg" in names


def test_treemap_loader_returns_none_on_missing(tmp_path: Path):
    assert loader.treemap_loader(tmp_path / "nope.ann") is None


def test_db_loader_returns_empty_dict_on_missing(tmp_path: Path):
    import torch

    out = loader.db_loader(tmp_path / "nope.pt", torch.device("cpu"))
    assert out == {}
