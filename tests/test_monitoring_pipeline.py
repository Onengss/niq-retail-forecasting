"""Tests for the NIQ monitoring pipeline."""

import subprocess

import pytest

from src.monitoring_pipeline import run_step


def test_run_step_success(monkeypatch):
    """Verify a monitoring step runs successfully."""
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_step("src.monitor_model")

    assert len(calls) == 1
    assert calls[0][0][-2:] == ["-m", "src.monitor_model"]
    assert calls[0][1] is True


def test_run_step_failure(monkeypatch):
    """Verify a failed monitoring step raises an error."""
    def fake_run(command, check):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_step("src.monitor_drift")
