"""Tests for crafter — pil_loader fast path and safe_collate."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from memery import crafter


def test_pil_loader_returns_rgb_image(bundled_images_dir: Path):
    f = bundled_images_dir / "memes" / "Wholesome-Meme-1.jpg"
    img = crafter.pil_loader(str(f))
    assert img is not None
    assert img.mode == "RGB"


def test_pil_loader_returns_none_on_corrupt(bundled_images_dir: Path):
    f = bundled_images_dir / "memes" / "corrupted-file.jpeg"
    if not f.exists():
        pytest.skip("corrupt fixture missing")
    out = crafter.pil_loader(str(f))
    assert out is None


def test_safe_collate_drops_none_items():
    """A batch with mixed None and real items collates only the real ones."""
    a = (torch.zeros(3, 8, 8), 0)
    b = (torch.zeros(3, 8, 8), 1)
    out = crafter.safe_collate([a, None, b])
    images, labels = out
    assert images.shape == (2, 3, 8, 8)
    assert labels.tolist() == [0, 1]


def test_safe_collate_returns_none_on_all_failed():
    """A batch where every item failed yields None — encoder skips it."""
    assert crafter.safe_collate([None, None]) is None
    assert crafter.safe_collate([]) is None
