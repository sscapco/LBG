#!/usr/bin/env python3
"""
Launcher script that ensures proper Python path setup.
"""
import sys
import os
from pathlib import Path

# Add the governance_pipeline directory to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Now import and run
from run import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
