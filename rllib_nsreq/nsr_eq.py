"""
Simple Q-Learning
=================

This module provides a basic implementation of the DQN algorithm without any
optimizations.

This file defines the distributed Algorithm class for the Simple Q algorithm.
See `simple_q_[tf|torch]_policy.py` for the definition of the policy loss.
"""

import logging
from typing import List, Optional, Type, Union

from .nsr_eq_torch_policy import NSREQTorchPolicy

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TARGET_NET_UPDATE_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import (
    update_priorities_in_replay_buffer,
    validate_buffer_config,
)
from ray.rllib.utils.typing import ResultDict

logger = logging.getLogger(__name__)


class NSREQConfig(AlgorithmConfig):

    def __init__(self, algo_class=None):
        """Initializes a NSREQConfig instance."""
        super().__init__(algo_class=algo_class or NSREQ)

        # Simple Q specific
        # fmt: off
        # __sphinx_doc_begin__
        self.target_network_update_freq = 500
        self.replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": 50000,
            # The number of contiguous environment steps to replay at once. This
            # may be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
        }
        self.num_steps_sampled_before_learning_starts = 1000
        self.store_buffer_in_checkpoints = False
        self.lr_schedule = None
        self.adam_epsilon = 1e-8

        self.grad_clip = 40.0
        # Note: Only when using _enable_learner_api=True can the clipping mode be
        # configured by the user. On the old API stack, RLlib will always clip by
        # global_norm, no matter the value of `grad_clip_by`.
        self.grad_clip_by = "global_norm"

        self.tau = 1.0
        # __sphinx_doc_end__
        # fmt: on

        # Overrides of AlgorithmConfig defaults
        # `rollouts()`
        self.rollout_fragment_length = 4

        # `training()`
        self.lr = 5e-4
        self.train_batch_size = 32
        self.collect_size = 1500
        self.train_times_per_step = 1
        self.keep_n_steps = 10

        # `exploration()`
        self.exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,
        }

        # `evaluation()`
        self.evaluation(evaluation_config=AlgorithmConfig.overrides(explore=False))

        # `reporting()`
        self.min_time_s_per_iteration = None
        self.min_sample_timesteps_per_iteration = 1000

        # Deprecated.
        self.buffer_size = DEPRECATED_VALUE
        self.prioritized_replay = DEPRECATED_VALUE
        self.learning_starts = DEPRECATED_VALUE
        self.replay_batch_size = DEPRECATED_VALUE
        # Can not use DEPRECATED_VALUE here because -1 is a common config value
        self.replay_sequence_length = None
        self.prioritized_replay_alpha = DEPRECATED_VALUE
        self.prioritized_replay_beta = DEPRECATED_VALUE
        self.prioritized_replay_eps = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        target_network_update_freq: Optional[int] = NotProvided,
        replay_buffer_config: Optional[dict] = NotProvided,
        store_buffer_in_checkpoints: Optional[bool] = NotProvided,
        lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        adam_epsilon: Optional[float] = NotProvided,
        grad_clip: Optional[int] = NotProvided,
        num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
        tau: Optional[float] = NotProvided,
        collect_size: Optional[float] = NotProvided,
        train_times_per_step: Optional[float] = NotProvided,
        keep_n_steps: Optional[int] = NotProvided,
        **kwargs,
    ) -> "NSREQConfig":
        """Sets the training related configuration.

        Args:
            target_network_update_freq: Update the target network every
                `target_network_update_freq` sample steps.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            store_buffer_in_checkpoints: Set this to True, if you want the contents of
                your buffer(s) to be stored in any saved checkpoints as well.
                Warnings will be created if:
                - This is True AND restoring from a checkpoint that contains no buffer
                data.
                - This is False AND restoring from a checkpoint that does contain
                buffer data.
            lr_schedule: Learning rate schedule. In the format of [[timestep, value],
                [timestep, value], ...]. A schedule should normally start from
                timestep 0.
            adam_epsilon: Adam optimizer's epsilon hyper parameter.
            grad_clip: If not None, clip gradients during optimization at this value.
            num_steps_sampled_before_learning_starts: Number of timesteps to collect
                from rollout workers before we start sampling from replay buffers for
                learning. Whether we count this in agent steps  or environment steps
                depends on config.multi_agent(count_steps_by=..).
            tau: Update the target by \tau * policy + (1-\tau) * target_policy.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if target_network_update_freq is not NotProvided:
            self.target_network_update_freq = target_network_update_freq
        if replay_buffer_config is not NotProvided:
            # Override entire `replay_buffer_config` if `type` key changes.
            # Update, if `type` key remains the same or is not specified.
            new_replay_buffer_config = deep_update(
                {"replay_buffer_config": self.replay_buffer_config},
                {"replay_buffer_config": replay_buffer_config},
                False,
                ["replay_buffer_config"],
                ["replay_buffer_config"],
            )
            self.replay_buffer_config = new_replay_buffer_config["replay_buffer_config"]
        if store_buffer_in_checkpoints is not NotProvided:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if adam_epsilon is not NotProvided:
            self.adam_epsilon = adam_epsilon
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if collect_size is not NotProvided:
            self.collect_size = collect_size
        if train_times_per_step is not NotProvided:
            self.train_times_per_step = train_times_per_step
        if keep_n_steps is not NotProvided:
            self.keep_n_steps = keep_n_steps
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if tau is not NotProvided:
            self.tau = tau
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        if self.exploration_config["type"] == "ParameterNoise":
            if self.batch_mode != "complete_episodes":
                raise ValueError(
                    "ParameterNoise Exploration requires `batch_mode` to be "
                    "'complete_episodes'. Try setting `config.rollouts("
                    "batch_mode='complete_episodes')`."
                )

        if not self.in_evaluation:
            validate_buffer_config(self)


class NSREQ(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return NSREQConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return NSREQTorchPolicy
        else:
            raise ValueError("Only PyTorch is supported for Simple DQN.")

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        """NSREQ training iteration function.

        NSREQ consists of the following steps:
        - Sample n MultiAgentBatches from n workers synchronously.
        - Store new samples in the replay buffer.
        - Sample one training MultiAgentBatch from the replay buffer.
        - Learn on the training batch.
        - Update the target network every `target_network_update_freq` sample steps.
        - Return all collected training metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        batch_size = self.config.train_batch_size
        local_worker = self.workers.local_worker()

        # print("begin to collect batches from envs")

        # Sample n MultiAgentBatches from n workers.
        with self._timers[SAMPLE_TIMER]:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers, concat=False,
                max_env_steps=self.config.collect_size,
                sample_timeout_s = self.config.sample_timeout_s
            )

        # print("finish collect batches from envs, store to buffer")
        
        for batch in new_sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Store new samples in the replay buffer
            self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_SAMPLED
            if self.config.count_steps_by == "agent_steps"
            else NUM_ENV_STEPS_SAMPLED
        ]

        # print("begin to train on the buffer")
        
        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(int(self.config.train_times_per_step * self.config.collect_size)):
                # Use deprecated replay() to support old replay buffers for now
                train_batch = self.local_replay_buffer.sample(batch_size)

                # Learn on the training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers (only those policies that were actually trained).
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(
                        policies=list(train_results.keys()),
                        global_vars=global_vars,
                    )
        else:
            train_results = {}

        # Return all collected metrics for the iteration.
        return train_results
