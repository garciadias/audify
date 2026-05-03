"""Tests for container-aware CLI path helpers."""

import logging
from pathlib import Path

import audify.cli as cli


def _patch_container_paths(monkeypatch, container_data_root: Path) -> None:
    """Patch CLI container path constants for isolated tests."""
    monkeypatch.setattr(cli, "_CONTAINER_DATA_ROOT", container_data_root)
    monkeypatch.setattr(cli, "_CONTAINER_INPUT_ROOT", container_data_root / "input")
    monkeypatch.setattr(cli, "_CONTAINER_OUTPUT_ROOT", container_data_root / "output")
    monkeypatch.setattr(cli, "_is_container_runtime", lambda: True)


def test_resolve_input_path_maps_host_style_data_path(tmp_path, monkeypatch):
    """Host-style /.../data/... path should resolve to mounted container data path."""
    container_data = tmp_path / "container_data"
    (container_data / "ebooks").mkdir(parents=True)
    mapped_input = container_data / "ebooks" / "book.pdf"
    mapped_input.write_bytes(b"pdf-bytes")

    _patch_container_paths(monkeypatch, container_data)

    resolved = cli._resolve_input_path_for_runtime(
        "/home/user/workspace/data/ebooks/book.pdf",
        logging.getLogger("test"),
    )

    assert resolved == str(mapped_input)
