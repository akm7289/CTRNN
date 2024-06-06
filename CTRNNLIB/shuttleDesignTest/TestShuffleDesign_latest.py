import random

import numpy as np
import tensorflow as tf
from keras import layers, regularizers
from keras.backend import concatenate
from matplotlib import pyplot as plt
from numpy.distutils.command.build import build
from tensorflow import keras
from keras.layers import SimpleRNN, Input
#from keras.optimizers import SGD,Adam


import pandas as pd
from keras.constraints import max_norm,MinMaxNorm
from keras.regularizers import l2



import CTRNNLIB.GenericFunctions
from CTRNNLIB import GenericFunctions
from CTRNNLIB.Recurrent import SimpleMEMSCTRNN
from CTRNNLIB.Recurrent import ShuttleCTRNN,RealShuttleCTRNN
from CTRNNLIB.Recurrent import SimpleRNN
from CTRNNLIB.shuttleDesignTest import Generate_Data_Module
from CTRNNLIB.shuttleDesignTest.Generate_Data_Module import generate_training_data

windowSize = 1





def train_deniosing_problem_MEMS_window_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=25,enviroment_end_time=30*3392*10**-6,N_=5)



    x_train = factor * x_train+shift
    y_train = factor * y_train+shift
    x_test = factor * x_train
    y_test = factor * y_train
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_,window_size,number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True

    arg_vb = 10
    arg_input_gain = 10
    #arg_output_gain = 10**7
    arg_output_gain = 5*(10**5)
    min_weight = -100
    max_weight = 100
    reccurrent_min=5
    recurrent_max=15
    #recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.3, stddev=0.05, seed=2)
    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain, kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min,max_value=recurrent_max,rate=1), omega_0=14566.3706144)(input_layer)#f=1k
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+2, max_value=recurrent_max), omega_0=12566.3706144)(input_layer)#f=2k
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain, kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+4, max_value=recurrent_max), omega_0=18849.5559215)(input_layer)#f=3k
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain, kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+5, max_value=recurrent_max), omega_0=25132.741)(input_layer)#4k
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+7, max_value=recurrent_max), omega_0=31415.9265359)(input_layer)#5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu', kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 8
    for i in range(n_epochs):
        print("Epoch:", i + 1)
        for j in range(len(x_train)):
            model.fit(x_train[j], y_train[j], epochs=1, batch_size=batch_size_, shuffle=False)

            model.reset_states()
            print("Reset State")
    model.save_weights('../../models/shuttle_design_latest.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3392*10**-6)
    x_test=x_test.reshape(x_test.shape[1], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1], window_size, number_of_features)

    x_test = factor * x_test + shift
    y_test = factor * y_test + shift

    output=model.predict(x_test,batch_size=batch_size_)
    plt.plot(y_test.reshape(-1),'r',alpha=1)
    plt.plot(x_test.reshape(-1),'g',alpha=.6)
    plt.xlabel("time*10^-6")
    plt.ylabel("voltage")
    plt.title("blue:Predicted Signal, Red: Ground Truth, Green:Noised Signal")


    plt.plot(output.reshape(-1), 'b',alpha=.8)
    plt.show()



if __name__ == '__main__':

    window_size_arg=100
    factor_arg=0.1
    train_deniosing_problem_MEMS_window_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg, shift=0)
