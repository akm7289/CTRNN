from keras import activations, Sequential, regularizers
import tensorflow as tf
from CTRNNLIB.Recurrent import SimpleMEMSCTRNN, LSTM

from keras.layers import Bidirectional
from tensorflow.keras.layers import  TimeDistributed, Dense, TimeDistributed, SimpleRNN, Input,concatenate
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import GenericFunctions
import generate_signal_square_traingle
windowSize=20

def read_dataset(dataset_path="../Data/denoising_signal_100.csv",window_size=20):
    df = pd.read_csv(filepath_or_buffer=dataset_path)
    split=45000
    training = df.iloc[:split, :]
    testing = df.iloc[split:, :]
    windowSize=window_size
    traininX = training.iloc[:, 0:windowSize]
    x_train = np.array(training.iloc[:, 0:windowSize])
    y_train = np.array(training.iloc[:, windowSize:windowSize+1])

    x_test = np.array(testing.iloc[:, 0:windowSize])
    y_test = np.array(testing.iloc[:, windowSize:windowSize+1])
    print("***************")
    print("data has been loaded successfully")
    print((x_train.shape))
    print("***************")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test

def train_deniosing_problem_all_cell_mems_window_size(factor=1, window_size=1):
    x_train, y_train, x_test, y_test = read_dataset(dataset_path="../Data/denoising_signal_window_size_5_batch_size_16_generator.csv", window_size=window_size)
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test

    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    input_layer=Input((window_size, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    rnn_1 = SimpleRNN(1, activation=None, return_sequences=True, stateful=False)(input_layer)
    hidden_1 = SimpleRNN(2, activation=None, return_sequences=True)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleRNN(1, activation=None)(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]


    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_)
    model.save_weights('../models/model_RNN_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-"*300)
    get_weights_custom_function(model, [5,2,1])
    testMode_modified(model, factor,window_size=window_size)

def get_weights_custom_function(model, number_of_units=[5]):
    for count, value in enumerate(number_of_units):
        W = model.layers[count+1].get_weights()[0]
        U = model.layers[count+1].get_weights()[1]
        b = model.layers[count+1].get_weights()[2]
        printweigths(W,U,b,value)
        print("*"*100)
def testMode_modified(model,factor=1,window_size=20):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01, gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    #corruptedSignal.tofile('deniosed_signal.csv', sep=',', format='%10.8f')


    model_signal = [0] * window_size
    # ctrnn_singal=[0]*20

    zList=[]

    for i in range(window_size, gen_samples):
        input_ = np.array(corruptedSignal[i - window_size:i]).reshape(1, window_size, 1)
        model_ouput = model.predict(input_*factor)
        # print('input shape:  ', model_ouput.shape)
        # print('all_state_h shape: ', model_ouput.shape)
        # print('\nhidden states for the first sample: \n', model_ouput[0])
        # print('\nhidden states for the first sample at the second time step: \n', model_ouput[0][1])
        # print("\n z:",model_ouput[2])
        #zList.append(model_ouput[2][0][0])
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
    x_gt = [np.sin(2 * np.pi * t)*factor for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    plt.plot(corruptedSignal*factor)
    plt.plot(model_signal, '-')
    #plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, Green:CTRNN')
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

if __name__ == '__main__':
    print('hi main')
    train_deniosing_problem_all_cell_mems_window_size(window_size=5)