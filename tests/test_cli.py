"""Tests for tiktoken CLI."""

import subprocess
import sys
import tempfile
import os

import pytest


def run_cli(*args: str, input: str | None = None) -> subprocess.CompletedProcess:
    """Run the tiktoken CLI with the given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "tiktoken.cli", *args],
        capture_output=True,
        text=True,
        input=input,
    )


def test_cli_text_argument():
    """Test passing text as a positional argument."""
    result = run_cli("hello world")
    assert result.returncode == 0
    # "hello world" with o200k_base should produce 2 tokens
    assert result.stdout.strip() == "2"


def test_cli_file_input():
    """Test reading text from a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        f.flush()
        temp_path = f.name
    
    try:
        result = run_cli("-f", temp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == "2"
    finally:
        os.unlink(temp_path)


def test_cli_stdin_input():
    """Test reading text from stdin."""
    result = run_cli(input="hello world")
    assert result.returncode == 0
    assert result.stdout.strip() == "2"


def test_cli_encoding_option():
    """Test specifying encoding."""
    result = run_cli("-e", "cl100k_base", "hello world")
    assert result.returncode == 0
    assert result.stdout.strip() == "2"


def test_cli_model_option():
    """Test specifying model."""
    result = run_cli("-m", "gpt-4o", "hello world")
    assert result.returncode == 0
    assert result.stdout.strip() == "2"


def test_cli_list_encodings():
    """Test listing available encodings."""
    result = run_cli("--list-encodings")
    assert result.returncode == 0
    assert "o200k_base" in result.stdout
    assert "cl100k_base" in result.stdout


def test_cli_version():
    """Test version output."""
    result = run_cli("--version")
    assert result.returncode == 0
    assert "tiktoken" in result.stdout


def test_cli_no_input():
    """Test that CLI shows help when no input is provided."""
    # Run with a fake stdin that appears as a tty
    result = subprocess.run(
        [sys.executable, "-m", "tiktoken.cli"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 1


def test_cli_invalid_encoding():
    """Test error handling for invalid encoding."""
    result = run_cli("-e", "nonexistent_encoding", "hello")
    assert result.returncode == 1
    assert "Unknown encoding" in result.stderr


def test_cli_invalid_model():
    """Test error handling for invalid model."""
    result = run_cli("-m", "nonexistent_model", "hello")
    assert result.returncode == 1
    assert "Could not automatically map" in result.stderr


def test_cli_mutually_exclusive_encoding_model():
    """Test that -e and -m are mutually exclusive."""
    result = run_cli("-e", "o200k_base", "-m", "gpt-4o", "hello")
    assert result.returncode != 0
