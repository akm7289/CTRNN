import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



my_ode_func=lambda t:(1./2.) * np.sin(t) - (1./2.) * np.cos(t) + \
                    (3./5.) * np.cos(2.*t) + (6./5.) * np.sin(2.*t) - \
                    (1./10.) * np.exp(-t)

class ODECell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, func, t0, t1, init_state, **kwargs):
        super(ODECell, self).__init__(**kwargs)
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.init_state = init_state
        self.ode_solver = tfp.math.ode.DormandPrince()

    @property
    def state_size(self):
        return self.init_state.shape

    @property
    def output_size(self):
        return self.init_state.shape

    def call(self, inputs, states):
        init_state = states[0]
        # Solve the ODE
        solution = self.ode_solver.solve(
            self.func,
            self.t0,
            init_state,
            solution_times=tfp.math.ode.ChosenBySolver(self.t1),
            rtol=1e-6,
            atol=1e-6,
        )
        # Update the state and output
        new_state = solution.states[-1]
        output = new_state
        return output, [new_state]


ode_cell = ODECell(func=my_ode_func, t0=0., t1=1., init_state=tf.constant([1.0, 0.0]))
rnn_layer = tf.keras.layers.RNN(ode_cell)
print("finished")
