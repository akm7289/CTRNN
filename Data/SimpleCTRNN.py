import tensorflow.compat.v2 as tf

import collections
import functools
import warnings

import numpy as np
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.saving.saved_model import layer_serialization
from keras.utils import control_flow_util
from keras.utils import generic_utils
from keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@doc_controls.do_not_generate_docs
class DropoutRNNCellMixin:
  """Object that hold dropout related fields for RNN Cell.
  This class is not a standalone RNN cell. It suppose to be used with a RNN cell
  by multiple inheritance. Any cell that mix with class should have following
  fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
    _random_generator: A backend.RandomGenerator instance, which will be used
      to produce outputs based on the inputs and dropout rate.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

  def __init__(self, *args, **kwargs):
    self._create_non_trackable_mask_cache()
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _create_non_trackable_mask_cache(self):
    """Create the cache for dropout and recurrent dropout mask.
    Note that the following two masks will be used in "graph function" mode,
    e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    tensors will be generated differently than in the "graph function" case,
    and they will be cached.
    Also note that in graph mode, we still cache those masks only because the
    RNN could be created with `unroll=True`. In that case, the `cell.call()`
    function will be invoked multiple times, and we want to ensure same mask
    is used every time.
    Also the caches are created without tracking. Since they are not picklable
    by python when deepcopy, we don't want `layer._obj_reference_counts_dict`
    to track it by default.
    """
    self._dropout_mask_cache = backend.ContextValueCache(
        self._create_dropout_mask)
    self._recurrent_dropout_mask_cache = backend.ContextValueCache(
        self._create_recurrent_dropout_mask)

  def reset_dropout_mask(self):
    """Reset the cached dropout masks if any.
    This is important for the RNN layer to invoke this in it `call()` method so
    that the cached mask is cleared before calling the `cell.call()`. The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._dropout_mask_cache.clear()

  def reset_recurrent_dropout_mask(self):
    """Reset the cached recurrent dropout masks if any.
    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._recurrent_dropout_mask_cache.clear()

  def _create_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask(
        self._random_generator,
        tf.ones_like(inputs),
        self.dropout,
        training=training,
        count=count)

  def _create_recurrent_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask(
        self._random_generator,
        tf.ones_like(inputs),
        self.recurrent_dropout,
        training=training,
        count=count)

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the dropout mask for RNN cell's input.
    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.
    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the recurrent dropout mask for RNN cell.
    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.
    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.recurrent_dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def __getstate__(self):
    # Used for deepcopy. The caching can't be pickled by python, since it will
    # contain tensor and graph.
    state = super(DropoutRNNCellMixin, self).__getstate__()
    state.pop('_dropout_mask_cache', None)
    state.pop('_recurrent_dropout_mask_cache', None)
    return state

  def __setstate__(self, state):
    state['_dropout_mask_cache'] = backend.ContextValueCache(
        self._create_dropout_mask)
    state['_recurrent_dropout_mask_cache'] = backend.ContextValueCache(
        self._create_recurrent_dropout_mask)
    super(DropoutRNNCellMixin, self).__setstate__(state)


@keras_export('keras.layers.SimpleCTRNN')
class SimpleCTRRNN(RNN):
  """Fully-connected RNN where the output is to be fed back to input.
  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.  Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for the linear transformation of the inputs.
      Default: 0.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for the linear transformation of the
      recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state
      in addition to the output. Default: `False`
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
  Call arguments:
    inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False` entry
      indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  Examples:
  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  simple_rnn = tf.keras.layers.SimpleRNN(4)
  output = simple_rnn(inputs)  # The output has shape `[32, 4]`.
  simple_rnn = tf.keras.layers.SimpleRNN(
      4, return_sequences=True, return_state=True)
  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = simple_rnn(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if 'implementation' in kwargs:
      kwargs.pop('implementation')
      logging.warning('The `implementation` argument '
                      'in `SimpleCTRNN` has been deprecated. '
                      'Please remove it from your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = SimpleCTRNNCell(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(SimpleRNN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    return super(SimpleRNN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(SimpleRNN, self).get_config()
    config.update(_config_for_enable_caching_device(self.cell))
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config:
      config.pop('implementation')
    return cls(**config)


def _generate_dropout_mask(generator, ones, rate, training=None, count=1):
  def dropped_inputs():
    return generator.dropout(ones, rate)

  if count > 1:
    return [
        backend.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  return backend.in_train_phase(dropped_inputs, ones, training=training)


def _config_for_enable_caching_device(rnn_cell):
  """Return the dict config for RNN cell wrt to enable_caching_device field.
  Since enable_caching_device is a internal implementation detail for speed up
  the RNN variable read when running on the multi remote worker setting, we
  don't want this config to be serialized constantly in the JSON. We will only
  serialize this field when a none default value is used to create the cell.
  Args:
    rnn_cell: the RNN cell for serialize.
  Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
  """
  default_enable_caching_device = tf.compat.v1.executing_eagerly_outside_functions()
  if rnn_cell._enable_caching_device != default_enable_caching_device:
    return {'enable_caching_device': rnn_cell._enable_caching_device}
  return {}