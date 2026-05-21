"""Compatibility alias for SMART's minimal PyMesh replacement.

The official package keeps this top-level module so legacy SMART code that does
``import pymesh`` continues to run. New SMART code should import
``smart.pymesh_compat`` directly.
"""

from smart.pymesh_compat import *  # noqa: F401,F403

