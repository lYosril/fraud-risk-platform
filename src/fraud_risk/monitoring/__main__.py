"""Run the monitoring CLI with ``python -m fraud_risk.monitoring``."""

from __future__ import annotations

import sys

from fraud_risk.monitoring.cli import main

if __name__ == "__main__":
    sys.exit(main())
