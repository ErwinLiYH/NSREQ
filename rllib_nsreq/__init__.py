from .nsr_eq import NSREQConfig, NSREQ
from .nsr_eq_torch_policy import NSREQTorchPolicy

from ray.tune.registry import register_trainable

__all__ = [
    "NSREQ",
    "NSREQConfig",
    "NSREQTorchPolicy",
]

register_trainable("rllib-contrib-simple-dqn", NSREQ)
