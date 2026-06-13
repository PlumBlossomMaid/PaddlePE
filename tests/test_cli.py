"""Tests for CLI."""

from __future__ import annotations

import subprocess
import sys


def _run_paddlepe(*args: str) -> subprocess.CompletedProcess:
    """Run paddlepe CLI and return result."""
    cmd = [sys.executable, "-m", "paddlepe.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestCLI:
    """Test CLI commands."""

    def test_list_models(self):
        """-l lists available models."""
        result = _run_paddlepe("-l")
        assert result.returncode == 0
        assert "fcpe" in result.stdout
        assert "rmvpe" in result.stdout

    def test_help(self):
        """--help displays usage."""
        result = _run_paddlepe("--help")
        assert result.returncode == 0
        assert "paddlePE" in result.stdout

    def test_no_input_shows_help(self):
        """Running without input shows help."""
        result = _run_paddlepe()
        assert result.returncode != 0

    def test_nonexistent_file(self, tmp_dir):
        """Running with nonexistent file shows error."""
        result = _run_paddlepe("/nonexistent/file.wav", "-o", str(tmp_dir / "out.f0"))
        assert result.returncode != 0
        assert "not found" in result.stderr or "not found" in result.stdout.lower()

    def test_f0_to_csv_convert(self, tmp_dir, sample_f0):
        """Convert .f0 to .csv."""
        from paddlepe.io import write_f0

        f0_path = tmp_dir / "test.f0"
        csv_path = tmp_dir / "test.csv"
        write_f0(f0_path, sample_f0, sample_rate=16000, hop_length=160)

        result = _run_paddlepe(str(f0_path), "-o", str(csv_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "time" in content
        assert "f0_hz" in content

    def test_csv_to_f0_convert(self, tmp_dir, sample_f0):
        """Convert .csv to .f0."""
        from paddlepe.io import write_csv

        csv_path = tmp_dir / "test.csv"
        f0_path = tmp_dir / "test.f0"
        write_csv(csv_path, sample_f0, sample_rate=16000, hop_length=160)

        result = _run_paddlepe(str(csv_path), "-o", str(f0_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert f0_path.exists()
        assert f0_path.stat().st_size > 44  # header size
