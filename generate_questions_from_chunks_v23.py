#!/usr/bin/env python3
"""
Generate questions from pre-extracted chunks.
Thin wrapper that skips chunking and calls make_v23 processing directly.
Uses \boxed{} format for concise, gradable answers.

Usage:
    python generate_questions_from_chunks_v23.py <chunks_dir> [make_v23 args...]

Example:
    python generate_questions_from_chunks_v23.py plos_genetics_chunks_qa --type qa --output plos_qa.json --enhanced
"""

import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_questions_from_chunks_v23.py <chunks_dir> [make_v23 args...]")
        sys.exit(1)

    chunks_dir = sys.argv[1]

    if not os.path.isdir(chunks_dir):
        print(f"Error: {chunks_dir} is not a directory")
        sys.exit(1)

    # Build chunk list from directory
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.txt')]
    chunk_ids = [f.replace('.txt', '') for f in chunk_files]

    print(f"Found {len(chunk_ids)} chunks in {chunks_dir}")

    # Import make_v23 and set pre-loaded chunks
    import make_v23
    make_v23._preloaded_chunks = chunk_ids
    make_v23._preloaded_chunks_dir = os.path.abspath(chunks_dir)

    # Build argv for make_v23 - use chunks_dir as dummy input, pass remaining args
    sys.argv = ['make_v23.py', chunks_dir] + sys.argv[2:]

    # Run make_v23
    make_v23.main()

if __name__ == '__main__':
    main()
