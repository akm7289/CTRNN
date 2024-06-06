import tensorflow as tf

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
from tensorflow.keras.layers import SimpleRNNCell, TimeDistributed, Dense, TimeDistributed, SimpleRNN, Input,concatenate

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


class CTRNN(SimpleRNN):

    def __init__(self, units, **kwargs):
        #self.units = units
        super(CTRNN, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, output