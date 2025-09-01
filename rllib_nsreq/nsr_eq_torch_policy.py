"""PyTorch policy class used for Simple Q-Learning"""

import logging
from typing import Any, Dict, List, Tuple, Type, Union

from .utils import make_model

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors, l2_loss
from ray.rllib.utils.typing import TensorStructType, TensorType

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional
logger = logging.getLogger(__name__)

import pickle
import torch
import copy


class NSREQTorchPolicy(
    LearningRateSchedule,
    TorchPolicyV2,
):
    """PyTorch policy class used with SimpleQTrainer."""

    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def make_model(self) -> ModelV2:
        """Builds q_model for Simple Q learning."""
        return make_model(self)

    @override(TorchPolicyV2)
    def compute_actions(
        self,
        *,
        input_dict,
        explore=True,
        timestep=None,
        episodes=None,
        is_training=False,
        **kwargs
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorStructType]]:
        if timestep is None:
            timestep = self.global_timestep
        # Compute the Q-values for each possible action, using our Q-value network.
        q_vals = self._compute_q_values(
            self.model, input_dict[SampleBatch.OBS], is_training=is_training
        )
        # Use a Categorical distribution for the exploration component.
        # This way, it may either sample storchastically (e.g. when using SoftQ)
        # or deterministically/greedily (e.g. when using EpsilonGreedy).
        distribution = TorchCategorical(q_vals, self.model)
        # Call the exploration component's `get_exploration_action` method to
        # explore, if necessary.
        actions, logp = self.exploration.get_exploration_action(
            action_distribution=distribution, timestep=timestep, explore=explore
        )
        # Return (exploration) actions, state_outs (empty list), and extra outs.
        return (
            actions,
            [],
            {
                "q_values": q_vals,
                SampleBatch.ACTION_LOGP: logp,
                SampleBatch.ACTION_PROB: torch.exp(logp),
                SampleBatch.ACTION_DIST_INPUTS: q_vals,
            },
        )

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for NSREQ.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The NSREQ loss tensor given the input batch.
        """

        # q network evaluation
        q_t = self._compute_q_values(
            model,
            train_batch[SampleBatch.CUR_OBS],
            is_training=True
        )

        # q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), self.action_space.n
        )
        q_t_selected = torch.sum(q_t * one_hot_selection, 1)

        # compute KEEPs
        if isinstance(train_batch[SampleBatch.INFOS][0], dict):
            k = [i["keep_return"] for i in train_batch[SampleBatch.INFOS]]
        else:
            k = [0] * len(train_batch)
        k = torch.tensor(k, dtype=torch.float32).to(self.device)

        # compute RHS of bellman equation
        q_t_selected_target = (
            train_batch[SampleBatch.REWARDS] + k
        )

        # Compute the error (Square/Huber).
        td_error = q_t_selected - q_t_selected_target.detach()
        # loss = torch.mean(huber_loss(td_error))
        loss = torch.mean(l2_loss(td_error))

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["loss"] = loss
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        model.tower_stats["td_error"] = td_error

        return loss

    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        # Auto-add empty learner stats dict if needed.
        return dict({LEARNER_STATS_KEY: {}}, **fetches)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "loss": torch.mean(torch.stack(self.get_tower_stats("loss"))),
                "cur_lr": self.cur_lr,
            }
        )

    def _compute_q_values(
        self, model: ModelV2, obs_batch: TensorType, is_training=None
    ) -> TensorType:
        _is_training = is_training if is_training is not None else False
        input_dict = SampleBatch(obs=obs_batch, _is_training=_is_training)
        # Make sure, everything is PyTorch tensors.
        model_out, _ = model(input_dict, [], None)
        return model_out

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch,
        other_agent_batches = None,
        episode = None,
    ):
        raw_b = copy.deepcopy(sample_batch)
        try:
            if (
                isinstance(sample_batch[SampleBatch.INFOS][0], dict) and   # ray will use fack datas to check API before formal training, infos will be 0
                SampleBatch.ACTIONS in sample_batch.keys()                 # when evaluation, there is no action in sample_batch
                                                                           # and also we dont need to compute keep_times when evaluation
            ):
                n_step = self.config["keep_n_steps"]
                # compute action keep times to info
                for i in range(len(sample_batch)):
                    k = 0
                    n = 1
                    next_bound = min(i + 1 + n_step, len(sample_batch))
                    for j in range(i+1, next_bound):
                        to_continue = (
                            # sample_batch[SampleBatch.ACTIONS][i] == sample_batch[SampleBatch.ACTIONS][j] and  # same action
                            sample_batch[SampleBatch.DONES][i] == False                                   # not done
                        )
                        if to_continue:
                            k += (self.config["gamma"] ** n) * sample_batch[SampleBatch.INFOS][j]["RB_variation"]
                        n += 1
                    sample_batch[SampleBatch.INFOS][i]["keep_return"] = k
        except Exception as e:                                                    # store the raw batch and batch for debug
            with open("raw_batch.pkl", "wb") as f:
                pickle.dump(raw_b, f)
            with open("batch.pkl", "wb") as f:
                pickle.dump(sample_batch, f)
            raise Exception(f"Error in postprocess_trajectory: \n{e}")
        return sample_batch
