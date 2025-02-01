# -*- coding: utf-8 -*-
# Author:
#     Hermann Zeyen <hermann.zeyen@gmail.com>
#     Université Paris-Saclay, France
#
# License: BSD 3 clause

# from .io import get_files, read_geography_file, Data
# from .dialog import dialog
# from .geometrics import Geometrics

# print(f"Invoking __init__.py for {__name__}")

from .io import get_files, read_lineaments, read_synthetic_model, read_geography_file, get_mag_field, read_geometrics, write_geometrics
from .communication import get_geometry, get_time_correction, get_justify_indices, clip_parameters, get_spector1D, get_spector2D
from .dialog import Dialog
from .earth import Earth_mag
__all__ = ["get_files", "read_lineaments", "read_synthetic_model", "read_geography_file", "get_mag_field", "read_geometrics", "write_geometrics", "get_geometry", "get_time_correction", "get_justify_indices", "clip_parameters", "get_spector1D", "get_spector2D", "Dialog", "Earth_mag"]
