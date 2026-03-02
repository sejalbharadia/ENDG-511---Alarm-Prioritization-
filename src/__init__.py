"""ENDG-511 Alarm Prioritization Project package exports.

This file exposes common symbols for easier imports from notebooks and scripts:

	from src import AlarmClassifier, train_supervised, ssl_train, prune_model_l1

Keeping imports lightweight to avoid heavy imports on package import.
"""

__version__ = "0.1.0"

# Expose commonly used symbols for convenience
from .model import AlarmClassifier, TinyBackbone, ClassifierHead, EarlyExitHead  # noqa: E402,F401
from .train import train_supervised  # noqa: E402,F401
from .ssl_train import SSLWrapper, ssl_train  # noqa: E402,F401
from .prune import prune_model_l1  # noqa: E402,F401
from .quantize import quantize_fx_model  # noqa: E402,F401
from .evaluate import evaluate, early_exit_inference  # noqa: E402,F401
from .utils import count_parameters, get_example_input  # noqa: E402,F401

__all__ = [
	"AlarmClassifier",
	"TinyBackbone",
	"ClassifierHead",
	"EarlyExitHead",
	"train_supervised",
	"SSLWrapper",
	"ssl_train",
	"prune_model_l1",
	"quantize_fx_model",
	"evaluate",
	"early_exit_inference",
	"count_parameters",
	"get_example_input",
]
