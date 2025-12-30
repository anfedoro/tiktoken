"""Command-line interface for tiktoken token counter."""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

import tiktoken


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count the number of tokens in the text."""
    return len(encoding.encode(text))


def read_input(
    text: str | None = None,
    file: str | None = None,
    stdin: TextIO | None = None,
) -> tuple[str | None, bool]:
    """Read input from text argument, file, or stdin.
    
    Returns a tuple of (content, input_provided) where content is the text
    content (may be empty string if file/stdin is empty) and input_provided
    is True if an input source was specified.
    """
    if text is not None:
        return text, True
    
    if file is not None:
        with open(file, "r", encoding="utf-8") as f:
            return f.read(), True
    
    if stdin is not None and not stdin.isatty():
        return stdin.read(), True
    
    return None, False


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="tiktoken",
        description="Count tokens in text using OpenAI's tiktoken tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  tiktoken "Hello, world!"
  tiktoken -f document.txt
  echo "Hello, world!" | tiktoken
  tiktoken -e cl100k_base "Hello, world!"
  tiktoken -m gpt-4o "Hello, world!"
""",
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize (can also be provided via stdin or -f)",
    )
    
    parser.add_argument(
        "-f", "--file",
        metavar="FILE",
        help="Read text from a file",
    )
    
    encoding_group = parser.add_mutually_exclusive_group()
    encoding_group.add_argument(
        "-e", "--encoding",
        default="o200k_base",
        help="Encoding to use (default: o200k_base). Use --list-encodings to see available encodings.",
    )
    encoding_group.add_argument(
        "-m", "--model",
        help="Model name to determine encoding (e.g., gpt-4o, gpt-3.5-turbo)",
    )
    
    parser.add_argument(
        "--list-encodings",
        action="store_true",
        help="List available encodings and exit",
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"tiktoken {tiktoken.__version__}",
    )
    
    args = parser.parse_args()
    
    # List encodings if requested
    if args.list_encodings:
        print("Available encodings:")
        for name in sorted(tiktoken.list_encoding_names()):
            print(f"  {name}")
        return 0
    
    # Read input
    try:
        input_text, input_provided = read_input(
            text=args.text,
            file=args.file,
            stdin=sys.stdin,
        )
    except OSError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1
    
    if not input_provided:
        parser.print_help()
        return 1
    
    # Get encoding
    try:
        if args.model:
            enc = tiktoken.encoding_for_model(args.model)
        else:
            enc = tiktoken.get_encoding(args.encoding)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Count tokens (input_text is a string when input_provided is True)
    token_count = count_tokens(input_text, enc)  # type: ignore[arg-type]
    print(token_count)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
