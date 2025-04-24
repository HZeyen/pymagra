# -*- coding: utf-8 -*-
# Author:
#     Hermann Zeyen <hermann.zeyen@gmail.com>
#     Université Paris-Saclay, France
#
# License: BSD 3 clause
"""Splinart package."""

from .io import get_files, read_geography_file, Data
from .dialog import dialog
from .geometrics import Geometrics

print(f"Invoking __init__.py for {__name__}")
