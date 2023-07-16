# Copyright (c) Microsoft Corporation. All rights reserved.
from __future__ import annotations

from pathlib import Path

import versioningit

PROJECT_DIR = Path(__file__).parent.parent
__version__ = versioningit.get_version(project_dir=PROJECT_DIR)
