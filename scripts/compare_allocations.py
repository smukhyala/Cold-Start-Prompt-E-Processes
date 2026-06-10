#!/usr/bin/env python
"""Thin shim around the `cold-start-compare` CLI for users who prefer running
a script directly. See `src/cold_start/cli/compare.py` for the real entrypoint.
"""

from cold_start.cli.compare import main

if __name__ == "__main__":
    main()
