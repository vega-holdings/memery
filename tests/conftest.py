"""Shared fixtures for the memery test suite.

The bundled ``images/`` folder is the canonical fixture corpus, but tests
shouldn't touch it directly — running ``index_flow`` writes ``memery.ann``
and ``memery.pt`` into whatever folder you point it at, and we don't want
those artifacts ending up in the working tree. So most fixtures copy a
subset into a tmp dir and let pytest clean it up.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_IMAGES = REPO_ROOT / "images"


@pytest.fixture(scope="session")
def bundled_images_dir() -> Path:
    """Read-only path to the bundled images/ folder."""
    assert BUNDLED_IMAGES.is_dir(), f"missing fixture corpus: {BUNDLED_IMAGES}"
    return BUNDLED_IMAGES


@pytest.fixture
def tiny_image_dir(tmp_path: Path, bundled_images_dir: Path) -> Path:
    """A tmp dir with ~3 valid images plus the corrupt fixture.

    Fast enough to encode end-to-end with real CLIP under the integration
    marker without the test suite taking forever.
    """
    dst = tmp_path / "tiny"
    dst.mkdir()
    valid = [
        "memes/Wholesome-Meme-68.jpg",  # the "dad joke" winner from earlier
        "memes/Wholesome-Meme-1.jpg",
        "memes/cute-dog-with-cupcake-P9E2YL5-min.jpg",
    ]
    for rel in valid:
        src = bundled_images_dir / rel
        assert src.exists(), f"fixture moved or missing: {src}"
        shutil.copy(src, dst / src.name)
    # include the deliberately corrupt file so loader tests can exercise it
    corrupt = bundled_images_dir / "memes" / "corrupted-file.jpeg"
    if corrupt.exists():
        shutil.copy(corrupt, dst / corrupt.name)
    return dst


@pytest.fixture
def empty_image_dir(tmp_path: Path) -> Path:
    d = tmp_path / "empty"
    d.mkdir()
    return d
