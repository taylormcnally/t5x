# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The model inference libraries for t5x evaluations."""

import abc
import collections
import time
from typing import Any, List, Mapping, Optional, Tuple, Type

import jax
import seqio
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf



class BaseModelInference(abc.ABC):
  """The abstract base class for the model inference framework."""

  def __init__(self, model: models.BaseTransformerModel):
    """The initialization that doesn't require shapes."""
    self._model = model
    self._feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)  # pytype:disable=not-instantiable

  @abc.abstractmethod
  def initialize_with_shapes(self, model_feature_shapes: Mapping[str,
                                                                 Tuple[int,
                                                                       ...]],
                             batch_size: int):
    """The initializations that requires shapes from the Evaluator."""
    pass

  @abc.abstractmethod
  def predict_fn(self, dataset: tf.data.Dataset, state: Any):
    """The functions for prediction.

    A user-defined function, which takes in a tf.data.Dataset and a state, and
    outputs the sequence of predicted tokens. Only called if predict metrics
    exist for the tasks.

    Args:
      dataset: A tf.data.Dataset.
      state: A user defined state that constains attributes step and stub.
    """
    pass

  @abc.abstractmethod
  def score_fn(self, dataset: tf.data.Dataset, state: Any):
    """The functions for scoring.

    A user-defined function, which takes in a tf.data.Dataset and a state, and
    outputs the log likelihood score of the targets. Only called if score
    metrics exist for the task.

    Args:
      dataset: A tf.data.Dataset.
      state: A user defined state that constains attributes step and stub.
    """
    pass

  @abc.abstractmethod
  def state_iterations(self):
    """Iterating over different states, e.g. checkpoints."""
    pass

  @property
  def feature_converter(self):
    """Feature converter for the task."""
    return self._feature_converter


class LocalModelInference(BaseModelInference):
  """The inference with a local model."""

  def __init__(self,
               model: models.BaseTransformerModel,
               restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
               partitioner: partitioning.BasePartitioner,
               fallback_init_rng: Optional[int] = None):
    """The initialization that doesn't require shapes.

    Args:
      model: The model object to use for inference.
      restore_checkpoint_cfg: Specification for the model parameter checkpoint
        to load.
      partitioner: Partitioner for the model parameters and data across devices.
      fallback_init_rng: A random seed used for parameter initialization during
        model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch
        is set to True. If None, parameter initialization is not allowed during
        model loading and having fallback_to_scratch enabled will result in an
        error.
    """
    super().__init__(model)
    self._restore_checkpoint_cfg = restore_checkpoint_cfg
    self._partitioner = partitioner
    self._fallback_init_rng = fallback_init_rng

  def initialize_with_shapes(self, model_feature_shapes: Mapping[str,
                                                                 Tuple[int,
                                                                       ...]],
                             batch_size: int):
    """T5X model loading.

    Args:
      model_feature_shapes: mapping from model feature to its shape in the
        `cached_model_datasets`.
      batch_size: batch size of the datasets.
    """
    # Initialize optimizer from the existing checkpoint.
    input_shapes = {
        k: (batch_size,) + s for k, s in model_feature_shapes.items()
    }
    self._train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=self._model.optimizer_def,
        init_fn=self._model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=self._partitioner)
    self._train_state_axes = self._train_state_initializer.train_state_axes
    if self._fallback_init_rng is not None:
      self._fallback_init_rng = jax.random.PRNGKey(self._fallback_init_rng)

    # Compile the model only once.
    self._predict_fn = utils.get_infer_fn(
        infer_step=self._model.predict_batch,
        batch_size=batch_size,
        train_state_axes=self._train_state_axes,
        partitioner=self._partitioner)

    self._score_fn = utils.get_infer_fn(
        infer_step=self._model.score_batch,
        batch_size=batch_size,
        train_state_axes=self._train_state_axes,
        partitioner=self._partitioner)

  def predict_fn(self, dataset: tf.data.Dataset, state: Any):
    return self._predict_fn(dataset, state)

  def score_fn(self, dataset: tf.data.Dataset, state: Any):
    return self._score_fn(dataset, state)

  def state_iterations(self):
    """Iterates over checkpoints and returns TrainState as state."""
    return self._train_state_initializer.from_checkpoints(
        [self._restore_checkpoint_cfg], init_rng=self._fallback_init_rng)


