"""Interpretation of a trained segmentation model.

Four analyses, each answering a different question about a finished checkpoint:

``ambiguity``
    At what confidence gap is the model no longer really choosing between its
    top two classes, and what does it cost to resolve those near-ties by class
    priority instead? Produces the threshold inference consumes.
``feature_importance``
    How much accuracy is lost when one input band is destroyed?
``shap``
    How much of the evidence for each class was built from each input band?
``grad_cam``
    Which part of the tile did the decision come from?

None of these feed back into training or into any later pipeline stage; they
describe a model that is already trained and published. That is why they live
in their own package, run from one analyst-invoked entry point
(``scripts/06_interpret_model.py``), and are each independently switchable.

The two band-attribution analyses reuse
:mod:`glacier_fsnow_unet.training.shap_importance` rather than reimplementing
their estimators, so the numbers an analyst gets here and the ones a training
run exports are produced by the same code.
"""

from .ambiguity import (
    AMBIGUITY_FILENAME,
    AmbiguityCalibration,
    apply_priority_rule,
    calibrate_threshold,
    collect_confidence_gaps,
    read_calibration,
    write_calibration,
)
from .runner import ANALYSIS_NAMES, InterpretationOutcome, run_interpretation

__all__ = [
    "AMBIGUITY_FILENAME",
    "ANALYSIS_NAMES",
    "AmbiguityCalibration",
    "InterpretationOutcome",
    "apply_priority_rule",
    "calibrate_threshold",
    "collect_confidence_gaps",
    "read_calibration",
    "run_interpretation",
    "write_calibration",
]
