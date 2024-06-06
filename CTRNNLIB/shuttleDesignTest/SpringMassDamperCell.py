import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
from keras import layers, regularizers
from keras.backend import concatenate
from matplotlib import pyplot as plt
from numpy.distutils.command.build import build
from tensorflow import keras
from keras.layers import SimpleRNN, Input
from keras.optimizers import SGD, Adam
import pandas as pd

import CTRNNLIB.GenericFunctions
from CTRNNLIB import GenericFunctions
from CTRNNLIB.Recurrent import SimpleMEMSCTRNN
from CTRNNLIB.Recurrent import ShuttleCTRNN
from CTRNNLIB.Recurrent import SimpleRNN
from CTRNNLIB.shuttleDesignTest import Generate_Data_Module
import tensorflow as tf

class ODE_RNN_Cell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [units]
        super(ODE_RNN_Cell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_h = self.add_weight(shape=(self.units, self.units),
                                   initializer='uniform',
                                   name='W_h')
        self.W_x = self.add_weight(shape=(input_dim, self.units),
                                   initializer='uniform',
                                   name='W_x')
        self.b = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b')
        super(ODE_RNN_Cell, self).build(input_shape)

    def call(self, inputs, states):
        h = states[0]
        x = inputs
        tf.print(h)
        def ode(h, x):
            dh_dt = tf.matmul(h, self.W_h) + tf.matmul(x, self.W_x) + self.b
            return dh_dt

        h = h + ode(h, x) * 1.0  # Euler method with step size 1

        return h, [h]

    def get_config(self):
        config = {'units': self.units}
        base_config = super(ODE_RNN_Cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def read_dataset(dataset_path="../../Data/denoising_signal_100.csv", window_size=20,percentage=.95):
    df = pd.read_csv(filepath_or_buffer=dataset_path)
    split = int(len(df)* percentage)
    training = df.iloc[:split, :]
    testing = df.iloc[split:, :]
    windowSize = window_size
    traininX = training.iloc[:, 0:windowSize]
    x_train = np.array(training.iloc[:, 0:windowSize])
    y_train = np.array(training.iloc[:, windowSize:windowSize + 1])

    x_test = np.array(testing.iloc[:, 0:windowSize])
    y_test = np.array(testing.iloc[:, windowSize:windowSize + 1])
    print("***************")
    print("data has been loaded successfully")
    print((x_train.shape))
    print("***************")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def check_output_of_one_cell(windowSize, factor, shift):
    batch_size_ = 1
    window_size = 1
    number_of_features = 1
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=window_size,percentage=.1)
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

    input_layer = Input(batch_shape=(batch_size_, window_size, number_of_features),
                        name="input")
    #input_layer = Input((window_size, number_of_features))

    sin_signal=5*np.append(y_train,y_test)

    y_train=5*y_train
    #plt.plot(y_train, '-')
    #plt.show()

    m = 1.0
    c = 0.1
    k = 1.0
    ode_solver = tfp.math.ode.DormandPrince(atol=1e-6)

    rnn_layer = tf.keras.layers.RNN(ODE_RNN_Cell(units=1), return_sequences=
                                    True,stateful=True)(input_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=rnn_layer)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    epochs_=1
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/test.h5')
    model.summary()
    # layer1 = model.layers[1].get_weights()
    # layer1[1]=layer1[1]*0+1
    # layer1[0]=layer1[0]*0+1
    #
    # model.layers[1].set_weights(layer1)
    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print(weights)
    #     print("*" * 100)
    print("-" * 300)

    output=model.predict(x_train,batch_size=batch_size_)
    output=output[:,0,0]
    plt.plot(output, 'k')
    plt.show()

    print(output)
if __name__ == '__main__':
    print('hi main')

    #train_deniosing_problem_rnn_window_size_not_stateful_stacked(window_size=1, factor=0.3)
    #train_deniosing_problem_rnn_window_size_not_stateful(window_size=1, factor=5, shift=0)
    check_output_of_one_cell(windowSize=1,factor=5, shift=0)
    shift_arg = 5
    # train_deniosing_problem_all_cell_mems_window_size_not_stateful(factor=.1, window_size=5, shift_=shift_arg)
    # train_deniosing_problem_all_NN__window_size_not_stateful(factor=1,window_size=1)
    # read_and_test_model(factor=1,window_size=5,shift_=shift_arg,model_path="../models/model_weights_window_size1_no_pull_in.h5")
    #read_and_test_model_no_seq(factor=1, window_size=5, shift_=shift_arg,
                             #  model_path="../../models/model_weights_window_size1_no_pull_in.h5")
    #train_deniosing_problem_rnn_window_size_stateful(window_size=5,factor=1, shift=0)
    #train_deniosing_problem_all_cell_mems_window_size_mems_statful()