import random

import numpy as np
import tensorflow as tf
from keras import layers, regularizers
from keras.backend import concatenate
from matplotlib import pyplot as plt
from numpy.distutils.command.build import build
from tensorflow import keras
from keras.layers import SimpleRNN, Input
#from keras.optimizers import SGD, Adam
from keras.optimizers import gradient_descent_v2


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


def draw_model_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def train_deniosing_problem_all_cell_mems_window_size_not_stateful(factor=1, window_size=1, shift_=5):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="../../Data/denoising_signal_window_size_5_batch_size_16_generator.csv", window_size=window_size)

    # x_train, y_train, x_test, y_test = read_dataset(
    #     dataset_path="../Data/denoising_signal_100.csv", window_size=window_size)

    x_train = factor * x_train + shift_
    y_train = factor * y_train + shift_
    from random import shuffle

    # ind_list = [i for i in range(len(x_train))]
    # shuffle(ind_list)
    # x_train = x_train[ind_list,]
    # y_train = x_train[ind_list,]

    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 50
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 32
    input_layer = Input((None, 1))
    # input_layer = Input((1, 1))

    alpha_ = 1 * (10 ** 0)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    # z_factor_ = 0.3
    v_factor_ = 1.9146 * (10 ** -20)
    # scale_shift_layer = layers.Dense(1, activation='linear', )(input_layer)
    # rnn_1=SimpleRNN(5, activation='tanh',return_sequences=True)(input_layer)
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, alpha=alpha_,
                            displacement=displacement_,
                            stopper=stopper_,
                            z_factor=z_factor_,
                            v_factor=v_factor_, return_sequences=True, stateful=False,
                            # kernel_regularizer=regularizers.L2(l2=1e-4),
                            # bias_regularizer=regularizers.L2(1e-4),
                            # activity_regularizer=regularizers.L2(1e-6)
                            )(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='sigmoid'))(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs_, batch_size=batch_size_)
    draw_model_history(history)
    model.save_weights('../../models/model_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    get_weights_custom_function(model, [5])
    testMode_modified(model, factor, shift=shift_, window_size=window_size, batch_size=batch_size_)


def train_deniosing_problem_all_NN__window_size_not_stateful(factor=1, window_size=1):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="../../Data/denoising_signal_1_window_size.csv", window_size=window_size)

    # x_train, y_train, x_test, y_test = read_dataset(
    #     dataset_path="../Data/denoising_signal_100.csv", window_size=window_size)

    shift_ = 5
    x_train = factor * x_train + shift_
    y_train = factor * y_train + shift_
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 0
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 32
    input_layer = Input((None, 1))
    # input_layer = Input((1, 1))

    alpha_ = 1 * (10 ** 0)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    # z_factor_ = 0.3
    v_factor_ = 1.9146 * (10 ** -20)

    hidden_1 = layers.Dense(5, activation='sigmoid')(input_layer)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.SGD(learning_rate=0.1),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs_, batch_size=batch_size_)
    draw_model_history(history)
    model.save_weights('../../models/model_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5])
    testMode_modified(model, factor, shift=shift_, window_size=window_size, batch_size=batch_size_)


def test_final(model,window_size,batch_size,factor):
    window_size = 1
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=window_size, percentage=.05)
    x_train=x_train*factor
    y_train=y_train*factor
    output = model.predict(x_train)
    output = output[:, 0, 0]
    plt.plot(y_train)
    plt.plot(output, 'k')
    plt.show()

def train_deniosing_problem_rnn_window_size_not_stateful(factor=1, window_size=1, shift=0):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/1window_size_32generators.csv", window_size=window_size)*factor+shift

    # x_train = factor * x_train+shift
    # y_train = factor * y_train+shift
    # x_test = factor * x_test
    # y_test = factor * y_test
    epochs_ = 1
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 64
    input_layer = Input((window_size, 1))

    #
    # rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True)(input_layer)
    # rnn_2 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True)(input_layer)
    # rnn_3 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True)(input_layer)
    # rnn_4 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True)(input_layer)
    # rnn_5 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True)(input_layer)
    # rnn_1=concatenate([rnn_1, rnn_2, rnn_3, rnn_4, rnn_5])
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 10 ** 0
    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain*10, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)


    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='sigmoid',kernel_initializer=keras.initializers.RandomNormal(mean=10**7, stddev=0.05, seed=None)
))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/shuttle_model_weights_window_size1.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])
    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)
    test_final(model,window_size=window_size, batch_size=batch_size_,factor=factor)


def train_deniosing_problem_rnn_window_size_stateful(factor=1, window_size=1, shift=0):
    number_of_features = 1
    batch_size_=1
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/1window_size_32generators.csv", window_size=window_size,percentage=1)


    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], number_of_features)
    x_train = x_train[0:int(len(x_train) / batch_size_) * batch_size_, :]
    y_train = y_train[0:int(len(y_train) / batch_size_) * batch_size_, :]
    x_train = factor * x_train+shift
    y_train = factor * y_train+shift
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 1
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_,window_size,number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True

    # rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_2 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_3 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_4 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_5 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_1 = concatenate([rnn_1, rnn_2, rnn_3, rnn_4, rnn_5])
    arg_vb = 10
    arg_input_gain = 1
    arg_output_gain = 10**6
    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=1))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=2))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=3))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=4))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=100))(input_layer)


    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='sigmoid'))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.1),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_,shuffle=False)
    model.save_weights('../../models/shuttle_model_weights_window_size1.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])




    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=window_size,percentage=1)

    x_test = factor * x_train + shift
    y_test = factor * y_train + shift







    x_test=x_test.reshape(x_test.shape[0],window_size,number_of_features)
    x_test = x_test[0:int(len(x_test) / batch_size_) * batch_size_, :]
    output=model.predict(x_test,batch_size=batch_size_)
    plt.plot(y_test.reshape(-1),'k',alpha=.5)
    plt.plot(x_test.reshape(-1),'r',alpha=.3)


    plt.plot(output.reshape(-1), 'b',alpha=.8)
    plt.show()

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)
def train_deniosing_problem_rnn_window_size10_stateful(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=32
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/10window_size_32generators.csv", window_size=window_size,percentage=1)


    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], number_of_features)
    x_train = x_train[0:int(len(x_train) / batch_size_) * batch_size_, :]
    y_train = y_train[0:int(len(y_train) / batch_size_) * batch_size_, :]
    x_train = factor * x_train+shift
    y_train = factor * y_train+shift
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_,window_size,number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=False

    # rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_2 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_3 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_4 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_5 = CTRNNLIB.Recurrent.SimpleRNN(1, return_sequences=True, stateful=stateful_arg)(input_layer)
    # rnn_1 = concatenate([rnn_1, rnn_2, rnn_3, rnn_4, rnn_5])
    arg_vb = 10
    arg_input_gain = 1
    arg_output_gain = 10**7
    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=1))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=2))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=3))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=4))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.009, stddev=0.05, seed=100))(input_layer)

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(3, activation='sigmoid'))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear'))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_, shuffle=False)
    # train the network
    # for i in range(3):
    #     model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_,shuffle=False)
    #     model.reset_states()
    model.save_weights('../../models/shuttle_model_weights_window_size1.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])




    x_test, y_test, _, _ = read_dataset(
        dataset_path="./NewTrainingData/testing_10window_size.csv", window_size=window_size,percentage=1)

    x_test = factor * x_test + shift
    y_test = factor * y_test + shift

    x_test=x_test.reshape(x_test.shape[0],window_size,number_of_features)
    x_test = x_test[0:int(len(x_test) / batch_size_) * batch_size_, :]
    output=model.predict(x_test,batch_size=batch_size_)
    plt.plot(y_test.reshape(-1),'k',alpha=.5)
    plt.plot(x_test.reshape(-1),'r',alpha=.3)


    plt.plot(output.reshape(-1), 'b',alpha=.8)
    plt.show()

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)

def train_deniosing_problem_MEMS_window_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=10,enviroment_end_time=20*3392*10**-6,N_=5)



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
                  optimizer= tf.optimizers.Adam(learning_rate=0.0001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 10
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)
def train_deniosing_problem_Shuttle_SameFreq_MEMS_window_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=10,enviroment_end_time=20*3392*10**-6,N_=5)



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
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min,max_value=recurrent_max,rate=1))(input_layer)#f=1k
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+2, max_value=recurrent_max))(input_layer)#f=2k
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain, kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+4, max_value=recurrent_max))(input_layer)#f=3k
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain, kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+5, max_value=recurrent_max))(input_layer)#4k
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+7, max_value=recurrent_max))(input_layer)#5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu', kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.0001),
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
    model.save_weights('../../models/shuttle_approximate_design_sameFreq_latest.h5')
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)

def train_deniosing_problem_Real_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path='../../models/Real_shuttle_design_latest.h5'):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3392*10**-6,N_=5)



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
    min_weight = -1
    max_weight = 1
    reccurrent_min=1
    recurrent_max=7
    #recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.3, stddev=0.05, seed=2)
    cell1 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min,max_value=recurrent_max,rate=1))(input_layer)#f=1k
    cell2 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+2, max_value=recurrent_max))(input_layer)#f=2k
    cell3 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+4, max_value=recurrent_max))(input_layer)#f=3k
    cell4 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+5, max_value=recurrent_max))(input_layer)#4k
    cell5 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+8, max_value=recurrent_max))(input_layer)#5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu', kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 50
    for i in range(n_epochs):
        print("Epoch:", i + 1)
        for j in range(len(x_train)):
            model.fit(x_train[j], y_train[j], epochs=1, batch_size=batch_size_, shuffle=False)

            model.reset_states()
            print("Reset State")
    model.save_weights(model_path)
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=16,enviroment_end_time=5*3392*10**-6)
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)
def train_deniosing_problem_Real_MEMS_Different_frequency_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=25,enviroment_end_time=60*3392*10**-6,N_=5)



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
    min_weight = -1
    max_weight = 1
    reccurrent_min=1
    recurrent_max=5
    #recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.3, stddev=0.05, seed=2)
    cell1 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min,max_value=recurrent_max,rate=1), omega_0=14566.3706144)(input_layer)#f=1k
    cell2 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+2, max_value=recurrent_max), omega_0=12566.3706144)(input_layer)#f=2k
    cell3 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+3, max_value=recurrent_max), omega_0=18849.5559215)(input_layer)#f=3k
    cell4 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+4, max_value=recurrent_max), omega_0=25132.741)(input_layer)#4k
    cell5 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min+5, max_value=recurrent_max), omega_0=31415.9265359)(input_layer)#5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu', kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.0001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 3
    for i in range(n_epochs):
        print("Epoch:", i + 1)
        for j in range(len(x_train)):
            model.fit(x_train[j], y_train[j], epochs=1, batch_size=batch_size_, shuffle=False)

            model.reset_states()
            print("Reset State")
    model.save_weights('../../models/Real_shuttle_design_diff_freq_latest.h5')
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)


def train_deniosing_problem_naive_rnn_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=10,enviroment_end_time=20*3392*10**-6,N_=5)



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

    cell1 = SimpleRNN(1,  bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                          recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.1, stddev=0.05, seed=1))(input_layer)
    cell2 = SimpleRNN(1,  bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.3, stddev=0.05, seed=2))(input_layer)
    cell3 = SimpleRNN(1,  bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.5, stddev=0.05, seed=3))(input_layer)
    cell4 = SimpleRNN(1,  bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.75, stddev=0.05, seed=4))(input_layer)
    cell5 = ShuttleCTRNN(1,  bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight),
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.85, stddev=0.05, seed=100))(input_layer)

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu', kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',kernel_constraint=MinMaxNorm(min_value=min_weight,max_value=max_weight), use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.00001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 10
    for i in range(n_epochs):
        print("Epoch:", i + 1)
        for j in range(len(x_train)):
            model.fit(x_train[j], y_train[j], epochs=1, batch_size=batch_size_, shuffle=False)

            model.reset_states()
            print("Reset State")
    model.save_weights('../../models/rnn_design_latest_jun_22.h5')
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)
def test_deniosing_problem_naive_rnn_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/shuttle_model_working_10windowsize.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_,window_size,number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    min_weight = -100
    max_weight = 100

    cell1 = SimpleRNN(1, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                      kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                      recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.1, stddev=0.05, seed=1))(
        input_layer)
    cell2 = SimpleRNN(1, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                      kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                      recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.3, stddev=0.05, seed=2))(
        input_layer)
    cell3 = SimpleRNN(1, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                      kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                      recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.5, stddev=0.05, seed=3))(
        input_layer)
    cell4 = SimpleRNN(1, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                      kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                      recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.75, stddev=0.05, seed=4))(
        input_layer)
    cell5 = ShuttleCTRNN(1, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=.85, stddev=0.05, seed=100))(
        input_layer)

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()
    # for i in range(1,6):
    #     layer1 = model.layers[i].get_weights()
    #     layer1[1] = layer1[1] * 1  # recurrent weight
    #     if layer1[0][0]>1 or layer1[0][0]<-1:
    #         #layer1[0] = layer1[0] / layer1[0]
    #         #layer1[0]=layer1[0]/abs(layer1[0])
    #         print("change")
    #     model.layers[i].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=4*3393*10**-6)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)


def get_weight_from_array_test_deniosing_problem_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0


    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.05, seed=1))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=2, stddev=0.05, seed=2))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.5, stddev=0.05, seed=3))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.75, stddev=0.05, seed=4))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.1, stddev=0.05, seed=100))(input_layer)

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()
    for i in range(1,6):
        layer1 = model.layers[i].get_weights()
        layer1[1] = layer1[1] * 1  # recurrent weight
        if layer1[0][0]>1 or layer1[0][0]<-1:
            #layer1[0] = layer1[0] / layer1[0]
            #layer1[0]=layer1[0]/abs(layer1[0])
            print("change")
        model.layers[i].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    Weights=[[[0.3231884,-16.],[0.14096689,-17.999998],[0.52925754,20.0],[1.6084939, 23.115145],[1.3228484, 25.999998]],
            [[ 0.6506657 , -0.32637355, -0.34351805, -0.3226587 , -0.73256326],[-0.26785105, -0.6820075 , -0.06991576,  0.595215  , -0.64625233],
             [-0.6258356 ,  0.01996473, -0.41565964,  0.60832924,  0.63705415],[ 0.14840576, -0.27272093,  0.58691484, -0.41415378, -0.43970287],
             [ 0.5574578 ,  0.27873558, -0.64290947, -0.7251691 ,  0.6040413 ]
             ],
            [
	        [ 0.02700649],[-0.14893918],[ 0.9039477 ],[-0.5022317 ],[ 0.70600945]
            ]
            ]

    for i in range(1,6):
        layer1 = model.layers[i].get_weights()
        layer1[1] =layer1[1] -layer1[1]+Weights[0][i-1][1]# recurrent
        layer1[0]=layer1[0]-layer1[0] +Weights[0][i-1][0]#kernal
        model.layers[i].set_weights(layer1)
    layer2=model.layers[7].get_weights()+np.array(Weights[1])-model.layers[7].get_weights()
    model.layers[7].set_weights(layer2)
    layers3=model.layers[8].get_weights()+np.array(Weights[2])-model.layers[8].get_weights()
    model.layers[8].set_weights(layers3)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]

    model.save_weights(model_path)


    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)
def get_weight_from_array_test_deniosing_problem_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0


    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.05, seed=1))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=2, stddev=0.05, seed=2))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.5, stddev=0.05, seed=3))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.75, stddev=0.05, seed=4))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         recurrent_initializer=tf.keras.initializers.RandomNormal(mean=1.1, stddev=0.05, seed=100))(input_layer)

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()
    for i in range(1,6):
        layer1 = model.layers[i].get_weights()
        layer1[1] = layer1[1] * 1  # recurrent weight
        if layer1[0][0]>1 or layer1[0][0]<-1:
            #layer1[0] = layer1[0] / layer1[0]
            #layer1[0]=layer1[0]/abs(layer1[0])
            print("change")
        model.layers[i].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    Weights=[[[0.3231884,-16.],[0.14096689,-17.999998],[0.52925754,20.0],[1.6084939, 23.115145],[1.3228484, 25.999998]],
            [[ 0.6506657 , -0.32637355, -0.34351805, -0.3226587 , -0.73256326],[-0.26785105, -0.6820075 , -0.06991576,  0.595215  , -0.64625233],
             [-0.6258356 ,  0.01996473, -0.41565964,  0.60832924,  0.63705415],[ 0.14840576, -0.27272093,  0.58691484, -0.41415378, -0.43970287],
             [ 0.5574578 ,  0.27873558, -0.64290947, -0.7251691 ,  0.6040413 ]
             ],
            [
	        [ 0.02700649],[-0.14893918],[ 0.9039477 ],[-0.5022317 ],[ 0.70600945]
            ]
            ]

    for i in range(1,6):
        layer1 = model.layers[i].get_weights()
        layer1[1] =layer1[1] -layer1[1]+Weights[0][i-1][1]# recurrent
        layer1[0]=layer1[0]-layer1[0] +Weights[0][i-1][0]#kernal
        model.layers[i].set_weights(layer1)
    layer2=model.layers[7].get_weights()+np.array(Weights[1])-model.layers[7].get_weights()
    model.layers[7].set_weights(layer2)
    layers3=model.layers[8].get_weights()+np.array(Weights[2])-model.layers[8].get_weights()
    model.layers[8].set_weights(layers3)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]

    model.save_weights(model_path)


    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)

def test_deniosing_problem_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0
    reccurrent_min=5
    recurrent_max=15
    min_weight=5
    max_weight=15

    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min, max_value=recurrent_max, rate=1),
                         omega_0=14566.3706144)(input_layer)  # f=1k
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 2, max_value=recurrent_max),
                         omega_0=12566.3706144)(input_layer)  # f=2k
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 4, max_value=recurrent_max),
                         omega_0=18849.5559215)(input_layer)  # f=3k
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 5, max_value=recurrent_max),
                         omega_0=25132.741)(input_layer)  # 4k
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 7, max_value=recurrent_max),
                         omega_0=31415.9265359)(input_layer)  # 5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)

def test_deniosing_problem_Shuttel_sameFreq_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0
    reccurrent_min=5
    recurrent_max=15
    min_weight=5
    max_weight=15

    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min, max_value=recurrent_max, rate=1))(input_layer)  # f=1k
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 2, max_value=recurrent_max))(input_layer)  # f=2k
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 4, max_value=recurrent_max))(input_layer)  # f=3k
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 5, max_value=recurrent_max))(input_layer)  # 4k
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 7, max_value=recurrent_max))(input_layer)  # 5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)


def read_testing_data(path):
    df = pd.read_csv(path)
    y_test = df['Ground Truth']
    x_test=df['Noised Signal']
    return x_test,y_test


def test_deniosing_problem_Real_MEMS_window_size10_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1
    omega_factor=.8
    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0
    reccurrent_min=5
    recurrent_max=15
    min_weight=5
    max_weight=15
    d_arg=4.5 * (10 ** -6)
    dsUp_arg = 4.5 * (10 ** -6)
    omega_0_arg=2.0 * (22 / 7) * 3727.58

    cell1 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),omega_0=omega_0_arg*omega_factor,
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min, max_value=recurrent_max, rate=1))(input_layer)  # f=1k
    cell2 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),omega_0=omega_0_arg*omega_factor,
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 2, max_value=recurrent_max))(input_layer)  # f=2k
    cell3 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),omega_0=omega_0_arg*omega_factor,
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 4, max_value=recurrent_max))(input_layer)  # f=3k
    cell4 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),omega_0=omega_0_arg*omega_factor,
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 5, max_value=recurrent_max))(input_layer)  # 4k
    cell5 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),omega_0=omega_0_arg*omega_factor,
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min +8, max_value=recurrent_max))(input_layer)  # 5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]
    #x_test, y_test=read_testing_data(path='./output/file_perfect_match.csv')
    # x_test=x_test[40000:60000]
    # y_test=y_test[40000:60000]

    x_test, y_test = generate_training_data(window_size=window_size,signal_frequency=300, num_of_generator=1, enviroment_end_time=5*3393*10**-6,N_=1)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)
    #
    x_test = factor * x_test + shift
    y_test = factor * y_test + shift

    output=model.predict(x_test,batch_size=batch_size_)
    model.evaluate(x_test,y_test)
    plt.plot(y_test.reshape(-1),color='blue',alpha=1)
    plt.plot(x_test.reshape(-1),color='orange',alpha=.6)
    # plt.plot(y_test, color='blue', alpha=1)
    # plt.plot(x_test, color='orange', alpha=.6)
    plt.xlabel("time*10^-6")
    plt.ylabel("voltage")
    plt.title("blue:Truth, Orange: Noised, Green:Model")



    plt.plot(output.reshape(-1), color='green',alpha=.8)
    plt.show()
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)


def read_data(file):
    df = pd.read_csv(file)



def test_deniosing_problem_Real_MEMS_Different_frequency_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0
    reccurrent_min=5
    recurrent_max=15
    min_weight=5
    max_weight=15

    cell1 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                             stateful=stateful_arg,
                             kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                             recurrent_constraint=MinMaxNorm(min_value=reccurrent_min, max_value=recurrent_max, rate=1),
                             omega_0=14566.3706144)(input_layer)  # f=1k
    cell2 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                             stateful=stateful_arg,
                             kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                             recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 2, max_value=recurrent_max),
                             omega_0=12566.3706144)(input_layer)  # f=2k
    cell3 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                             stateful=stateful_arg,
                             kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                             recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 4, max_value=recurrent_max),
                             omega_0=18849.5559215)(input_layer)  # f=3k
    cell4 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                             stateful=stateful_arg,
                             kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                             recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 5, max_value=recurrent_max),
                             omega_0=25132.741)(input_layer)  # 4k
    cell5 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                             stateful=stateful_arg,
                             kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                             recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 7, max_value=recurrent_max),
                             omega_0=31415.9265359)(input_layer)  # 5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]
    x_test, y_test=read_data('./output/the_data.csv')
    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=5*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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

    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)
def test_deniosing_problem_Real_MEMS_window_size1_stateful_seq_to_seq_same_freq(factor=1, window_size=10, shift=0,model_path="../../models/rnn_design_latest.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_, window_size, number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0
    reccurrent_min=5
    recurrent_max=15
    min_weight=5
    max_weight=15

    cell1 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min, max_value=recurrent_max, rate=1))(input_layer)  # f=1k
    cell2 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 2, max_value=recurrent_max))(input_layer)  # f=2k
    cell3 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 4, max_value=recurrent_max))(input_layer)  # f=3k
    cell4 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 5, max_value=recurrent_max))(input_layer)  # 4k
    cell5 = RealShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                         stateful=stateful_arg,
                         kernel_constraint=MinMaxNorm(min_value=min_weight, max_value=max_weight),
                         recurrent_constraint=MinMaxNorm(min_value=reccurrent_min + 7, max_value=recurrent_max))(input_layer)  # 5k

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])

    #
    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])
    #rnn_1 = CTRNNLIB.Recurrent.SimpleRNN(3, use_bias=True, return_sequences=True, stateful=False)(input_layer)

    #rnn_1 = ShuttleCTRNN(3, vb=10,  bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=10**8)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='relu',use_bias=False))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.TimeDistributed(layers.Dense(1, activation='linear',use_bias=False))(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    model.load_weights(model_path)
    model.summary()

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.layers[1].get_weights()[0].=model.layers[1].get_weights()[0] - model.layers[1].get_weights()[0]

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=25*3393*10**-6,N_=5)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)

def train_deniosing_problem_rnn_window_size_full_stacked_stateful_seq_to_seq(factor=1, window_size=10, shift=0):
    number_of_features = 1
    batch_size_=1

    # Generate some example data

    x_train, y_train=generate_training_data(window_size=window_size, num_of_generator=6,enviroment_end_time=3*3392*10**-6)



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
    min_weight = -1000
    max_weight = 1000
    layer1=ShuttleCTRNN(5, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain)(input_layer)
    layer2=ShuttleCTRNN(5, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain)(layer1)

    output_ = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=stateful_arg,
                         input_gain=arg_input_gain, output_gain=arg_output_gain)(layer2)

    #



    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )
    # train the network
    n_epochs = 4
    for i in range(n_epochs):
        print("Epoch:", i + 1)
        for j in range(len(x_train)):
            model.fit(x_train[j], y_train[j], epochs=1, batch_size=batch_size_, shuffle=False)

            model.reset_states()
            print("Reset State")
    model.save_weights('../../models/shuttle_all_MEMS.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
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

    #testMode_modified(model, factor, shift=shift, window_size=window_size, batch_size=batch_size_)


def test_deniosing_problem_rnn_window_size_full_stacked_stateful_seq_to_seq(factor=1, window_size=10, shift=0,model_path="../../models/shuttle_all_MEMS.h5"):
    number_of_features = 1
    batch_size_=1

    # Generate some example data


    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input(batch_shape= (batch_size_,window_size,number_of_features),
                       name="input")
    #input_layer = Input((window_size, 1))
    stateful_arg=True
    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 5 * (10**5)
    #arg_output_gain = 10**0

    arg_vb = 10
    arg_input_gain = 10
    # arg_output_gain = 10**7
    arg_output_gain = 5 * (10 ** 5)
    min_weight = -1
    max_weight = 1
    layer1 = ShuttleCTRNN(5, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                          stateful=stateful_arg,
                          input_gain=arg_input_gain, output_gain=arg_output_gain)(input_layer)
    layer2 = ShuttleCTRNN(5, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                          stateful=stateful_arg,
                          input_gain=arg_input_gain, output_gain=arg_output_gain)(layer1)

    output_ = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True,
                           stateful=stateful_arg,
                           input_gain=arg_input_gain, output_gain=arg_output_gain)(layer2)

    #

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mae',
                  optimizer= tf.optimizers.Adam(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mse'],

                  )

    # train the network
    model.load_weights(model_path)
    model.summary()
    # for i in range(1,6):
    #     layer1 = model.layers[i].get_weights()
    #     layer1[1] = layer1[1] * 1  # recurrent weight
    #     if layer1[0][0]>1 or layer1[0][0]<-1:
    #         #layer1[0] = layer1[0] / layer1[0]
    #         #layer1[0]=layer1[0]/abs(layer1[0])
    #         print("change")
    #     model.layers[i].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])

    x_test, y_test = generate_training_data(window_size=window_size, num_of_generator=1,enviroment_end_time=4*3393*10**-6)
    x_test=x_test.reshape(x_test.shape[1]*x_test.shape[0], window_size, number_of_features)
    y_test=y_test.reshape(y_test.shape[1]*y_test.shape[0], window_size, number_of_features)

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
    import pandas as pd
    excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1),output.reshape(-1)])
    excel_array=np.transpose(excel_array)
    df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal', 'Model Output'])
    df.to_csv('./output/file.csv', index=False)

def test_deniosing_problem_rnn_window_size_not_stateful_stacked(factor=1, window_size=1):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="../../Data/denoising_signal_1_window_size.csv", window_size=window_size)

    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 64
    input_layer = Input((None, 1))
    rnn_1=SimpleRNN(5,return_sequences=True,stateful=False)(input_layer)
    #rnn_1 = ShuttleCTRNN(5, return_sequences=True, stateful=False)(input_layer)

    hidden_1=SimpleRNN(5,return_sequences=True,stateful=False)(rnn_1)

    output_ = SimpleRNN(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.1),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/shuttle_model_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    # get_weights_custom_function(model, [5, 2, 1])
    testMode_modified(model, factor, window_size=1, batch_size=batch_size_)


def testMode_modified(model, factor=1, window_size=20, shift=0, batch_size=32):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    gen_samples = 1000
    data=Generate_Data_Module.generate_testing_data(window_size=1,num_of_generator=1,rows=10000)
    data=data.to_numpy()
    noisedSignal=data[0]
    ground_truth_signal=data[1].to_numpy()


    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01,
                                                  gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    # corruptedSignal.tofile('deniosed_signal.csv', sep=',', format='%10.8f')

    model_signal = [0] * window_size
    # ctrnn_singal=[0]*20

    zList = []

    for i in range(window_size, gen_samples):
        input_ = np.array(corruptedSignal[i - window_size:i]).reshape(1, window_size, 1)
        testx = input_ * factor + shift
        model_ouput = model.predict(testx,batch_size=1)
        # print('input shape:  ', model_ouput.shape)
        # print('all_state_h shape: ', model_ouput.shape)
        # print('\nhidden states for the first sample: \n', model_ouput[0])
        # print('\nhidden states for the first sample at the second time step: \n', model_ouput[0][1])
        # print("\n z:",model_ouput[2])
        # zList.append(model_ouput[2][0][0])
        # print('\nsig(z-d):',tf.keras.activations.sigmoid(tf.math.multiply(model_ouput[2][0][0] - 41.8 * (10 ** -6), 5000)))
        model_signal.append(model_ouput[0][0])

    textfile = open("z_values_factor12.txt", "w")

    # max,min,avrg,std=generate_statistics(zList)
    # print(max," ",max," ", avrg," ", std)
    #
    # for element in zList:
    #     textfile.write(str(element) + "\n")
    # textfile.write("\nZ Statitics : "+"\n")
    # textfile.write("*"*100)
    #
    # textfile.write("\nmax : "+str(max) + "\n")
    # textfile.write("min : " + str(min) + "\n")
    # textfile.write("avrg : " + str(avrg) + "\n")
    # textfile.write("std : " + str(std) + "\n")
    # textfile.close()

    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [(np.sin(2 * np.pi * t)) * factor + shift for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    shifted_signal = (corruptedSignal) * factor + shift
    plt.plot(shifted_signal)
    plt.plot(model_signal, '-')
    # plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    d = {'signal': x_gt, 'deniosed': shifted_signal, 'model_output': model_signal}
    df = pd.DataFrame(d)
    df.to_csv("../../Data/output.csv")
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('Signals Blue: Noised  Black:Original, Orange:MEMS')
    plt.savefig('output.png')
    plt.show()


def train_deniosing_problem_all_cell_mems_window_size_mems_statful(factor=1, window_size=1):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="../../Data/denoising_signal_1_window_size_4generator.csv", window_size=window_size)
    x_train = x_train.reshape(x_train.shape[0], window_size, 1)

    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 32
    # input_layer = Input((None, 1))
    input_layer = Input(batch_shape=(batch_size_, window_size, 1),
                        name="input")
    alpha_ = 1 * (10 ** 5)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    v_factor_ = 1.9146 * (10 ** -20)

    alpha_ = 1 * (10 ** 5)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    v_factor_ = 1.9146 * (10 ** -20)

    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True, alpha=alpha_,
                            displacement=displacement_,
                            stopper=stopper_,
                            z_factor=z_factor_,
                            v_factor=v_factor_, stateful=True)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True, alpha=alpha_,
                               displacement=displacement_,
                               stopper=stopper_,
                               z_factor=z_factor_,
                               v_factor=v_factor_, stateful=True)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = SimpleMEMSCTRNN(1, activation='linear', alpha=alpha_,
                              displacement=displacement_,
                              stopper=stopper_,
                              z_factor=z_factor_,
                              v_factor=v_factor_, stateful=True)(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.000001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/model_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    get_weights_custom_function(model, [5, 2, 1])
    testMode_modified_statful(model, factor, window_size=1, batch_size=batch_size_)


def get_weights_custom_function(model, number_of_units=[5]):
    for count, value in enumerate(number_of_units):
        W = model.layers[count + 1].get_weights()[0]
        U = model.layers[count + 1].get_weights()[1]
        b = model.layers[count + 1].get_weights()[2]
        printweigths(W, U, b, value)
        print("*" * 100)


def testMode_modified_statful(model, factor=1, window_size=20, batch_size=32):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01,
                                                  gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    # corruptedSignal.tofile('deniosed_signal.csv', sep=',', format='%10.8f')

    model_signal = [0] * window_size
    # ctrnn_singal=[0]*20

    zList = []

    for i in range(batch_size, gen_samples - 10):
        x = np.array(corruptedSignal[i:i + batch_size])
        if (x.shape[0] != batch_size):
            break
        input_ = x.reshape(batch_size, 1, 1)
        # input_ = np.array(corruptedSignal[i - window_size:i]).reshape(1, window_size, 1)

        model_ouput = model.predict(input_ * factor)
        # print('input shape:  ', model_ouput.shape)
        # print('all_state_h shape: ', model_ouput.shape)
        # print('\nhidden states for the first sample: \n', model_ouput[0])
        # print('\nhidden states for the first sample at the second time step: \n', model_ouput[0][1])
        # print("\n z:",model_ouput[2])
        # zList.append(model_ouput[2][0][0])
        # print('\nsig(z-d):',tf.keras.activations.sigmoid(tf.math.multiply(model_ouput[2][0][0] - 41.8 * (10 ** -6), 5000)))
        model_signal.append(model_ouput[0])

    textfile = open("z_values_factor12.txt", "w")

    # max,min,avrg,std=generate_statistics(zList)
    # print(max," ",max," ", avrg," ", std)
    #
    # for element in zList:
    #     textfile.write(str(element) + "\n")
    # textfile.write("\nZ Statitics : "+"\n")
    # textfile.write("*"*100)
    #
    # textfile.write("\nmax : "+str(max) + "\n")
    # textfile.write("min : " + str(min) + "\n")
    # textfile.write("avrg : " + str(avrg) + "\n")
    # textfile.write("std : " + str(std) + "\n")
    # textfile.close()

    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t) * factor for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    plt.plot(corruptedSignal * factor)
    plt.plot(model_signal, '-')
    # plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black: Original, Blue:Noisy Signal, Orange:Rectified ')
    plt.show()


def printweigths(W, U, b, number_of_units):
    W_i = W[:, :number_of_units]
    W_f = W[:, number_of_units: number_of_units * 2]
    W_c = W[:, number_of_units * 2: number_of_units * 3]
    W_o = W[:, number_of_units * 3:]

    U_i = U[:, :number_of_units]
    U_f = U[:, number_of_units: number_of_units * 2]
    U_c = U[:, number_of_units * 2: number_of_units * 3]
    U_o = U[:, number_of_units * 3:]

    b_i = b[:number_of_units]
    b_f = b[number_of_units: number_of_units * 2]
    b_c = b[number_of_units * 2: number_of_units * 3]
    b_o = b[number_of_units * 3:]
    print("weights" * 50)
    print(W_i)
    print(b_i)


def read_and_test_model(factor=1, window_size=5, shift_=5, model_path='../../models/model_weights_window_size5_no_vw.h5'):
    input_layer = Input((None, 1))
    # input_layer = Input((1, 1))

    alpha_ = 1 * (10 ** 0)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    # z_factor_ = 0.3
    v_factor_ = 1.9146 * (10 ** -20)
    # scale_shift_layer = layers.Dense(1, activation='linear', )(input_layer)
    # rnn_1=SimpleRNN(5, activation='tanh',return_sequences=True)(input_layer)
    rnn_1, v_1, z1 = SimpleMEMSCTRNN(5, activation=None, alpha=alpha_,
                                     displacement=displacement_,
                                     stopper=stopper_,
                                     z_factor=z_factor_,
                                     v_factor=v_factor_, return_sequences=True, return_state=True)(input_layer)

    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='sigmoid'))(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    # model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]
    model = tf.keras.Model(inputs=[input_layer],
                           outputs=[output_, v_1, z1])
    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.1),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.load_weights(model_path)
    get_weights_custom_function(model, [5])
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)

    zsList = []
    vsList = []
    model_signal = [0] * window_size
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01,
                                                  gen_samples=gen_samples)
    corruptedSignal = fun_gen(0) * factor + shift_
    for i in range(window_size, gen_samples):
        input_ = np.array(corruptedSignal[i - window_size:i]).reshape(1, window_size, 1)
        model_ouput, v_1, z1 = model.predict(input_)

        # print(z1)
        print(v_1)
        vs = []
        for voltages in v_1:
            for v in voltages:
                vs.append(v)

        z = []
        for i in z1:
            for j in i:
                z.append(j)

        print('*' * 30)
        print('----' * 30)

        zsList.append(z)
        vsList.append(vs)

        model_signal.append(model_ouput[0][0][0])

    import csv

    with open('z-values_hybrid.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(zsList)

    with open('v-values_hybrid.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(vsList)
    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t) * factor + shift_ for t in t_samples]
    corruptedSignal = np.ndarray.tolist(corruptedSignal)
    from numpy.core.numeric import outer
    import pandas as pd

    import matplotlib.pyplot as plt
    model_input_output_ground_truth = pd.DataFrame(
        {'deniosed_signal': corruptedSignal,
         'model_output': model_signal,
         'ground_truth': x_gt
         })
    model_input_output_ground_truth.to_csv("model_input_output_ground_truth.csv")

    plt.plot(corruptedSignal)
    plt.plot(model_signal, '-')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()


def read_and_test_model_no_seq(factor=1, window_size=5, shift_=5,
                               model_path='../../models/model_weights_window_size5_no_vw.h5'):
    input_layer = Input((None, 1))
    # input_layer = Input((1, 1))

    alpha_ = 1 * (10 ** 0)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    v_factor_ = 1.9146 * (10 ** -20)
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, alpha=alpha_,
                            displacement=displacement_,
                            stopper=stopper_,
                            z_factor=z_factor_,
                            v_factor=v_factor_, return_sequences=False, return_state=False)(input_layer)

    hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    # model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]
    model = tf.keras.Model(inputs=[input_layer],
                           outputs=[output_])
    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.1),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.load_weights(model_path)
    get_weights_custom_function(model, [5])
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)

    zsList = []
    vsList = []
    model_signal = [0] * window_size
    gen_samples = 1000

    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01,
                                                  gen_samples=gen_samples, frequecny_factor=1)
    corruptedSignal = fun_gen(0) * factor + shift_
    for i in range(window_size, gen_samples):
        input_ = np.array(corruptedSignal[i - window_size:i]).reshape(1, window_size, 1)
        model_ouput = model.predict(input_)

        # print(z1)

        model_signal.append(model_ouput[0][0])

    import csv

    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t) * factor + shift_ for t in t_samples]
    corruptedSignal = np.ndarray.tolist(corruptedSignal)
    from numpy.core.numeric import outer
    import pandas as pd

    import matplotlib.pyplot as plt
    model_input_output_ground_truth = pd.DataFrame(
        {'deniosed_signal': corruptedSignal,
         'model_output': model_signal,
         'ground_truth': x_gt
         })
    model_input_output_ground_truth.to_csv("model_input_output_ground_truth_new.csv")

    plt.plot(corruptedSignal)
    plt.plot(model_signal, '-')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()


def check_output_of_one_cell(windowSize, factor, shift):
    batch_size_ = 1
    window_size = 1
    number_of_features = 1
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=window_size,percentage=.5)
    input_layer = Input(batch_shape=(batch_size_, window_size, number_of_features),
                        name="input")
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],number_of_features)

    #input_layer = Input((window_size, number_of_features))

    sin_signal=5*np.append(y_train,y_test)

    y_train=5*y_train
    #plt.plot(y_train, '-')
    #plt.show()
    rnn_1 = ShuttleCTRNN(1, kernel_initializer=tf.keras.initializers.Ones(),
                         recurrent_initializer=tf.keras.initializers.Ones(),
                         vb=10,
                         deltaT=10**-6,

                         bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=True, input_gain=1,output_gain=10**6)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=rnn_1)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    epochs_=0
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/test.h5')
    model.summary()
    layer1 = model.layers[1].get_weights()
    layer1[1]=layer1[1]*0+20#recurrent weight
    layer1[0]=layer1[0]*0+1

    model.layers[1].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)

    y_train=y_train.reshape(y_train.shape[0],window_size,number_of_features)
    output=model.predict(y_train,batch_size=batch_size_)
    plt.plot(output.reshape(-1), 'k')
    plt.show()

    print(output)
def check_output_of_RealShuttle_Design_one_cell(windowSize, factor, shift):
    batch_size_ = 1
    window_size = 1
    number_of_features = 1
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=window_size,percentage=.99)
    input_layer = Input(batch_shape=(batch_size_, window_size, number_of_features),
                        name="input")
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],number_of_features)

    #input_layer = Input((window_size, number_of_features))


    #plt.plot(y_train, '-')
    #plt.show()
    rnn_1 = RealShuttleCTRNN(1, kernel_initializer=tf.keras.initializers.Ones(),
                         recurrent_initializer=tf.keras.initializers.Ones(),
                         bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=True)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=rnn_1)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer= tf.optimizers.Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    epochs_=0
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/test.h5')
    model.summary()
    layer1 = model.layers[1].get_weights()
    layer1[1]=layer1[1]*0+20#recurrent weight
    layer1[0]=layer1[0]*0+1

    model.layers[1].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)

    y_train=y_train.reshape(y_train.shape[0],window_size,number_of_features)
    output=model.predict(y_train,batch_size=batch_size_)
    plt.plot(output.reshape(-1), 'k')
    plt.plot(y_train.reshape(-1), 'r')
    plt.xlabel("time*10^-6")
    plt.ylabel("voltage")
    plt.title("Red: Ground Truth, Black: Model-Output")
    plt.show()

    #print(output)

if __name__ == '__main__':
    print('hi main')
    #read_testing_data(path='./output/file.csv')
    #train_deniosing_problem_rnn_window_size_not_stateful_stacked(window_size=1, factor=0.3)
    #train_deniosing_problem_rnn_window_size_stateful(factor=1, window_size=1, shift=0)
    #train_deniosing_problem_rnn_window_size_stateful(factor=1, window_size=1, shift=0)
    #train_deniosing_problem_rnn_window_size10_stateful(factor=1, window_size=10, shift=0)
    window_size_arg=500
    factor_arg=.1
    #train_deniosing_problem_naive_rnn_window_size10_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg, shift=0)
    #test_deniosing_problem_naive_rnn_window_size10_stateful_seq_to_seq(factor=factor_arg,window_size=1,shift=0, model_path='../../models/rnn_design_latest_jun_22.h5')

    #train_deniosing_problem_MEMS_window_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg, shift=0)
    #test_deniosing_problem_MEMS_window_size10_stateful_seq_to_seq(factor=factor_arg,window_size=1,shift=0, model_path='../../models/shuttle_design_latest.h5')
    #train_deniosing_problem_Shuttle_SameFreq_MEMS_window_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg, shift=0)
    #test_deniosing_problem_Shuttel_sameFreq_MEMS_window_size10_stateful_seq_to_seq(factor=factor_arg, window_size=1, shift=0,
                                                                  #model_path='../../models/shuttle_approximate_design_sameFreq_latest.h5')

    #train_deniosing_problem_rnn_window_size_full_stacked_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg, shift=0)
    #test_deniosing_problem_rnn_window_size_full_stacked_stateful_seq_to_seq(factor=factor_arg,window_size=1,shift=0, model_path='../../models/shuttle_all_MEMS.h5')
    #check_output_of_one_cell(windowSize=1,factor=5, shift=0)
    model_path='../../models/Real_shuttle_design_same_freq_latest_oct_16_2023.h5'
    #train_deniosing_problem_Real_MEMS_window_size10_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg,shift=0,model_path=model_path)
    test_deniosing_problem_Real_MEMS_window_size10_stateful_seq_to_seq(factor=factor_arg, window_size=1, shift=0,model_path=model_path)
    #check_output_of_RealShuttle_Design_one_cell(windowSize=1,factor=.1, shift=0)
    #model_path = '../../models/Real_shuttle_design_latest_.h5'
    #test_deniosing_problem_Real_MEMS_window_size1_stateful_seq_to_seq_same_freq(factor=factor_arg, window_size=1, shift=0,model_path=model_path)

    #train_deniosing_problem_Real_MEMS_Different_frequency_stateful_seq_to_seq(factor=factor_arg, window_size=window_size_arg,shift=0)
    #test_deniosing_problem_Real_MEMS_Different_frequency_stateful_seq_to_seq(factor=factor_arg, window_size=1, shift=0,model_path='../../models/Real_shuttle_design_diff_freq_latest.h5')
    shift_arg = 5
    #train_deniosing_problem_all_cell_mems_window_size_not_stateful(factor=.1, window_size=5, shift_=shift_arg)
    # train_deniosing_problem_all_NN__window_size_not_stateful(factor=1,window_size=1)
    # read_and_test_model(factor=1,window_size=5,shift_=shift_arg,model_path="../models/model_weights_window_size1_no_pull_in.h5")
    #read_and_test_model_no_seq(factor=1, window_size=5, shift_=shift_arg,
                             #  model_path="../../models/model_weights_window_size1_no_pull_in.h5")
    #train_deniosing_problem_rnn_window_size_stateful(window_size=5,factor=1, shift=0)
    #train_deniosing_problem_all_cell_mems_window_size_mems_statful()