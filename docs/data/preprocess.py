#!/usr/bin/env python3
"""Preprocess the PyPercent files to improve the conversion to ipynb.

- Convert admonitions to work in Jupyter Lab.
"""

import argparse
import re

MAPPING = {
    "note": "✏ **Note**",
    "warning": "⚠ **Warning**",
}


def replacer(match):
    kind = match.group(1)
    content = match.group(2).strip()
    new = [f"# > {MAPPING[kind]}", *(f"# > {line[1:].strip()}" for line in content.splitlines())]
    return "\n".join(new)


def main():
    args = parse_args()
    with open(args.inp) as f:
        original = f.read()
    converted = re.sub(r"# :::{(\w+)}\n(.*?)# :::", replacer, original, flags=re.DOTALL)
    with open(args.out, "w") as f:
        f.write(converted)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MyST admonitions in Python files to HTML."
    )
    parser.add_argument("inp", type=str, help="Input Python file with MyST admonitions")
    parser.add_argument("out", type=str, help="Output Python file with converted admonitions")
    return parser.parse_args()


if __name__ == "__main__":
    main()
