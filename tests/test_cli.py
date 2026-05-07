"""CLI smoke tests — exercises typer wiring without touching CLIP."""
from __future__ import annotations

from typer.testing import CliRunner

from memery.cli import app

runner = CliRunner()


def test_cli_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("recall", "build", "serve", "purge"):
        assert cmd in result.stdout


def test_recall_help_exposes_negative_flag():
    result = runner.invoke(app, ["recall", "--help"])
    assert result.exit_code == 0
    assert "--negative-text" in result.stdout or "-nt" in result.stdout


def test_purge_is_idempotent_on_empty_dir(empty_image_dir):
    """purge on a folder with no index files should succeed quietly."""
    result = runner.invoke(app, ["purge", str(empty_image_dir)])
    assert result.exit_code == 0
    assert "Purged" in result.stdout
