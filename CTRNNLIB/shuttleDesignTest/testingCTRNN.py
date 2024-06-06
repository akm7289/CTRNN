import tensorflow as tf
import tensorflow_probability as tfp


class CTRNNCell(tf.keras.layers.Layer):
    def __init__(self, num_neurons, activation=tf.nn.tanh, **kwargs):
        super(CTRNNCell, self).__init__(**kwargs)
        self.num_neurons = num_neurons
        self.activation = activation
        self.state_size = (num_neurons,)  # Define the state size

    def build(self, input_shape):
        self.tau = self.add_weight(shape=(self.num_neurons,),
                                   initializer='ones',
                                   trainable=True,
                                   name='tau')
        self.weight_matrix = self.add_weight(shape=(self.num_neurons, self.num_neurons),
                                             initializer='random_normal',
                                             trainable=True,
                                             name='weights')
        self.input_weights = self.add_weight(shape=(self.num_neurons,),
                                             initializer='random_normal',
                                             trainable=True,
                                             name='input_weights')
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        super(CTRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        prev_state = states[0]

        def ode_fn(_, state):
            dx_dt = (1.0 / self.tau) * (
                    -state + tf.matmul(self.weight_matrix,
                                       self.activation(state + self.bias)) + inputs * self.input_weights)
            return dx_dt

        ode_solver = tfp.math.ode.DormandPrince(atol=1e-6, rtol=1e-3)
        states, _ = ode_solver.solve(ode_fn, initial_time=0.0, initial_state=prev_state, solution_times=[1.0])

        new_state = states[-1]
        return new_state, [new_state]


class CTRNN(tf.keras.layers.RNN):
    def __init__(self, num_neurons, activation=tf.nn.tanh, return_sequences=True, **kwargs):
        cell = CTRNNCell(num_neurons, activation)
        super(CTRNN, self).__init__(cell, return_sequences=return_sequences, **kwargs)


# Define the initial state, input signal, and time steps
initial_state = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
input_signal = tf.constant(0.5, dtype=tf.float32)
num_steps = 100

# Create an instance of the CTRNN
ctrnn = CTRNN(num_neurons=3)

# Iterate through time steps to simulate the CTRNN
current_state = initial_state
for step in range(num_steps):
    current_state, _ = ctrnn(input_signal, [current_state])
    print(f"Step {step}: {current_state.numpy()}")
