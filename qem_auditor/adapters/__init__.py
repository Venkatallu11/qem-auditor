"""Backend adapters: the components that let the auditor run a control
itself rather than take the claimant's word for it.

Importing this package does not import Qiskit. The adapters are optional;
the core auditor stays dependency-free.
"""
from .base import ControlMeasurement, MeasurementError

__all__ = ["ControlMeasurement", "MeasurementError"]
