"""Crash-safe checkpoint helpers (CPU only)."""
import pytest
import torch

from malaria_ssl.utils import run
from malaria_ssl.utils.run import load_checkpoint, prev_path, save_checkpoint


def test_keep_prev_rotates_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "ck" / "last.pt"
    save_checkpoint(p, keep_prev=True, epoch=1, w=torch.ones(3))
    assert not prev_path(p).exists()
    save_checkpoint(p, keep_prev=True, epoch=2, w=torch.zeros(3))
    save_checkpoint(p, keep_prev=True, epoch=3, w=torch.zeros(3))
    assert prev_path(p).name == "last.prev.pt"
    assert load_checkpoint(p)["epoch"] == 3
    assert torch.load(prev_path(p), weights_only=False)["epoch"] == 2
    assert sorted(f.name for f in p.parent.iterdir()) == ["last.prev.pt", "last.pt"]


def test_without_keep_prev_no_prev_file(tmp_path):
    p = tmp_path / "best.pt"
    save_checkpoint(p, epoch=1)
    save_checkpoint(p, epoch=2)
    assert load_checkpoint(p)["epoch"] == 2 and not prev_path(p).exists()


def test_failed_save_keeps_old_file(tmp_path, monkeypatch):
    p = tmp_path / "last.pt"
    save_checkpoint(p, keep_prev=True, epoch=1)
    save_checkpoint(p, keep_prev=True, epoch=2)

    def broken_save(obj, f):
        f.write(b"partial")
        raise RuntimeError("simulated crash during torch.save")
    monkeypatch.setattr(run.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(p, keep_prev=True, epoch=3)
    monkeypatch.undo()
    assert load_checkpoint(p)["epoch"] == 2
    assert torch.load(prev_path(p), weights_only=False)["epoch"] == 1


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_load_falls_back_to_prev(tmp_path, damage):
    p = tmp_path / "last.pt"
    save_checkpoint(p, keep_prev=True, epoch=1, w=torch.randn(1000))
    save_checkpoint(p, keep_prev=True, epoch=2, w=torch.randn(1000))
    if damage == "truncate":                      # the lc02 failure: "failed finding central directory"
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 2])
    else:
        p.unlink()
    assert load_checkpoint(p)["epoch"] == 1


def test_load_raises_when_nothing_readable(tmp_path):
    p = tmp_path / "last.pt"
    with pytest.raises(FileNotFoundError):
        load_checkpoint(p)
    p.write_bytes(b"garbage")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(p)
