#################Start########################
##############Shuttle Design##################
##############################################

@keras_export('keras.layers.ShuttleCTRNNCell')
class ShuttleCTRNNCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    """Cell class for ShuttleCTRNN.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.ShuttleCTRNN` processes the whole sequence.

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
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
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  rnn = tf.keras.layers.RNN(tf.keras.layers.ShuttleCTRNNCell(4))

  output = rnn(inputs)  # The output has shape `[32, 4]`.

  rnn = tf.keras.layers.RNN(
      tf.keras.layers.ShuttleCTRNNCell(4),
      return_sequences=True,
      return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = rnn(inputs)
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
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 omega_0=2.83 * (10 ** 4),
                 zeta=1,
                 vb=10,
                 mass=14 * (10 ** -9),
                 d=4.5 * (10 ** -6),  # gap
                 eps=8.854 * (10 ** -12),
                 A=150 * (10 ** -9),  # Excitation Area
                 Am=100 * (10 ** -9),  # tunning area

                 **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for argument `units`, '
                             f'expected a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(ShuttleCTRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units
        self.omega_0 = omega_0,
        self.zeta = zeta,
        self.vb = vb,
        self.mass = mass,
        self.d = d,  # gap
        self.eps = eps,
        self.A = A,  # Excitation Area
        self.Am = Am  # tunning area
        self.two_Eps_A_Vb_multipication_over_d_squared=(2 * self.eps * self.A * self.vb)/(self.d**2)
        #self.two_Eps_A_Vb_multipication_over_d_squared =tf.math.divide(tf.math.multiply(2, self.eps, self.A, self.vb),backend.pow(self.d,2))
        #self.Eps_Aover_d_cubic =tf.math.divide(tf.math.multiply(self.eps, self.A),backend.pow(self.d,3))
        self.eps_A_over_d_cubic =(self.eps* self.A)/(self.d**3)


    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True


    def call(self, inputs, states, training=None):
        print("ShuttleCTRNNCALL")
        prev_output = states[0] if tf.nest.is_nested(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            tf.print("dp_mask is not None")
            h = backend.dot(inputs * dp_mask, self.kernel)
        else:
            tf.print("dp_mask is None")
            h = backend.dot(inputs, self.kernel)
        if self.bias is not None:
            #h = backend.bias_add(h, self.bias)
            pass # no biased

        if rec_dp_mask is not None:
            tf.print("rec_dp_mask is not None")
            prev_output = prev_output * rec_dp_mask




        wv_square=backend.pow(self.recurrent_kernel,2)
        wv_square_zi=tf.math.multiply(wv_square,prev_output)
        first_term=tf.math.multiply(wv_square_zi,self.eps_A_over_d_cubic)

        second_term=tf.math.multiply(self.two_Eps_A_Vb_multipication_over_d_squared,h)
        output=tf.math.add(first_term+second_term)
        #output = h + backend.dot(prev_output, self.recurrent_kernel)
        # if self.activation is not None:
        #     output = self.activation(output)

        new_state = [output] if tf.nest.is_nested(states) else output
        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

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
        config.update(_config_for_enable_caching_device(self))
        base_config = super(ShuttleCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.ShuttleCTRNN')
class ShuttleCTRNN(RNN):
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
  simple_rnn = tf.keras.layers.ShuttleCTRNN(4)

  output = simple_rnn(inputs)  # The output has shape `[32, 4]`.

  simple_rnn = tf.keras.layers.ShuttleCTRNN(
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
                 omega_0=2.83 * (10 ** 4),
                 zeta=1,
                 vb=10,
                 mass=14 * (10 ** -9),
                 d=4.5 * (10 ** -6),  # gap
                 eps=8.854 * (10 ** -12),
                 A=150 * (10 ** -9),  # Excitation Area
                 Am=100 * (10 ** -9),  # tunning area
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `ShuttleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                               kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = ShuttleCTRNNCell(
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
            zeta = zeta,
            vb = vb,
            mass = mass,
            d = d,  # gap
            eps = eps,
            A = A,  # Excitation Area
            Am = Am  # tunning area
            **cell_kwargs)
        super(ShuttleCTRNN, self).__init__(
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
        print("RNN")
        return super(ShuttleCTRNN, self).call(
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
        base_config = super(ShuttleCTRNN, self).get_config()
        config.update(_config_for_enable_caching_device(self.cell))
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


####################End#######################
##############Shuttle Design##################
##############################################