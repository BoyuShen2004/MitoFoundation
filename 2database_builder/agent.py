"""
Compatibility entrypoint for database_builder (stage 2).

The master dispatcher lives in ``2database_builder/master/agent.py``.
"""

from __future__ import annotations

from master.agent import main


if __name__ == "__main__":
    main()
