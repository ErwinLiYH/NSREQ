from rllib_simple_dqn.simple_q import SimpleQ, SimpleQConfig
from rllib_simple_dqn.simple_q_torch_policy import SimpleQTorchPolicy

from ray.tune.registry import register_trainable

__all__ = [
    "SimpleQ",
    "SimpleQConfig",
    "SimpleQTorchPolicy",
]

register_trainable("rllib-contrib-simple-dqn", SimpleQ)
