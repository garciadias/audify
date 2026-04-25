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


def test_stage_input_to_host_data_copies_non_shared_input(tmp_path, monkeypatch):
    """Inputs outside mounted data path are copied to /app/data/input equivalent."""
    container_data = tmp_path / "container_data"
    container_data.mkdir(parents=True)
    external_input = tmp_path / "outside.pdf"
    external_input.write_bytes(b"outside-bytes")

    _patch_container_paths(monkeypatch, container_data)

    staged = cli._stage_input_to_host_data(external_input, logging.getLogger("test"))

    assert staged == container_data / "input" / "outside.pdf"
    assert staged.exists()
    assert staged.read_bytes() == b"outside-bytes"


def test_ensure_output_synced_to_host_data_copies_external_output(
    tmp_path,
    monkeypatch,
):
    """Outputs created outside mounted data path are copied into /app/data/output."""
    container_data = tmp_path / "container_data"
    container_data.mkdir(parents=True)
    external_output = tmp_path / "external_output"
    external_output.mkdir()
    (external_output / "episode_001.mp3").write_bytes(b"mp3-bytes")

    _patch_container_paths(monkeypatch, container_data)

    synced = cli._ensure_output_synced_to_host_data(
        external_output,
        logging.getLogger("test"),
    )

    assert synced == container_data / "output" / "external_output"
    assert (synced / "episode_001.mp3").exists()
    assert (synced / "episode_001.mp3").read_bytes() == b"mp3-bytes"
