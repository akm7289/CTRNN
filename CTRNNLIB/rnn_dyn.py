from time import sleep

import keras
from keras import activations, Sequential, regularizers
import tensorflow as tf
from keras import models
from keras.optimizers import Adam, SGD

from CTRNNLIB.Recurrent import SimpleMEMSCTRNN, LSTM

from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import Input
import pandas as pd
import numpy as np

windowSize=20

def read_dataset(dataset_path="../Data/denoising_signal_100.csv",window_size=20,shift=0):
    df = pd.read_csv(filepath_or_buffer=dataset_path)
    split=45000
    training = df.iloc[:split, :]
    testing = df.iloc[split:, :]
    training=training+shift
    testing=testing+shift
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


import GenericFunctions
import generate_signal_square_traingle
def train_memsctrnn_all_mems(batch_size=100,factor=1):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train=factor*x_train
    y_train=factor*y_train
    x_test=factor*x_test
    y_test=factor*y_test
    epochs_ = 2
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    tau_recip = 0.7
    rnn_1 = SimpleMEMSCTRNN(5, activation='tanh', return_sequences=True,  tau_reciprocal=tau_recip)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation='tanh',return_sequences=True,tau_reciprocal=tau_recip)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleMEMSCTRNN(1, activation='tanh', tau_reciprocal=tau_recip,return_state=True)(hidden_1)
    model = models.Model(inputs=input_layer, outputs=output_)# output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer='sgd',
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae']
                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size)
    testMode_modified(model,factor)


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


def get_weights_custom_function(model, number_of_units=[5]):
    for count, value in enumerate(number_of_units):
        W = model.layers[count+1].get_weights()[0]
        U = model.layers[count+1].get_weights()[1]
        b = model.layers[count+1].get_weights()[2]
        printweigths(W,U,b,value)
        print("*"*100)



def train_square_trainagle(batch_size_=32,factor=1):
   #x_trainf, y_trainf, x_testf, y_testf = read_dataset()
    factor = 1

    x_train, y_train, x_test, y_test = read_train_square_trainagle()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test


    x_train, y_train,_ = generate_signal_square_traingle.generate_dataset(shuffle=True)


    x_test, y_test,signal = generate_signal_square_traingle.generate_dataset(shuffle=False)


    epochs_ = 50
    # for i in x_test:
    #     print(i)
    # for j in y_test:
    #     print(j)
    #
    # exit(-1)
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    #input_layer=Input((None,11, 1))
    alpha_ = 1 * (10 ** 5)






    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    v_factor_ = 1.9146 * (10 ** -20)


    model=Sequential()


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    rnn_1 = SimpleMEMSCTRNN(units=2, activation=None, return_sequences=True, alpha=alpha_,
                            displacement=displacement_,
                            stopper=stopper_,
                            z_factor=z_factor_,
                            kernel_regularizer=regularizers.l2(l2=1),
                            bias_regularizer=regularizers.L2(1e-4),
                            v_factor=v_factor_,
                            #kernel_initializer='zeros',
                            recurrent_initializer='zeros',
                            bias_initializer='zeros',
                            )(input_layer)


    # rnn_1 = SimpleMEMSCTRNN(units=2, activation=None, return_sequences=True, alpha=alpha_,
    #                     displacement=displacement_,
    #                     stopper=stopper_,
    #                     z_factor=z_factor_,
    #                     # kernel_regularizer=regularizers.l2(l2=1),
    #                     # bias_regularizer=regularizers.L2(1e-4),
    #                     v_factor=v_factor_,
    #                     # kernel_initializer='zeros',
    #                     recurrent_initializer='zeros',
    #                     bias_initializer='zeros',
    #                     )(rnn_1)


    output_ = SimpleMEMSCTRNN(1,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(l2=1),
                   bias_regularizer=regularizers.L2(1e-4)
                   )(rnn_1)

    # output_ = SimpleMEMSCTRNN(units=1, activation='linear', alpha=alpha_,
    #                           displacement=displacement_,
    #                           stopper=stopper_,
    #                           z_factor=z_factor_,
    #                           #kernel_regularizer=regularizers.L2(l2=1),
    #                           #bias_regularizer=regularizers.L2(1e-4),
    #                           v_factor=v_factor_,
    #                           #kernel_initializer='zeros',
    #                           #recurrent_initializer='zeros',
    #                           #bias_initializer='zeros',
    #                           )(rnn_1)


    #rnn1 = SimpleRNN(units=2,return_sequences=True)(input_layer)


    #rnn1 = SimpleRNN(units=1, activation='tanh', return_sequences=True)(rnn1)


# rnn1 = LSTM(units=36, activation='tanh',return_sequences=True)(rnn1)
    #
    #
     #rnn1 = LSTM(units=36,return_sequences=True, activation='tanh')(rnn1)

    #output_ = SimpleRNN(1)(rnn1)

#output_ = LSTM(units=1)(rnn1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss=['mse'],
                  optimizer=SGD(learning_rate=0.001),
                  #optimizer=RMSprop(learning_rate=0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],
                  )



    #function(model)

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    #model.compile(loss='binary_crossentropy',  metrics=['accuracy'], optimizer=SGD(learning_rate=0.0001))
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_,callbacks=[callback])
    print("prediction"*50)
    #sleep(2)
    print("load the model"*100)
    #model.load_weights('../models/model_weights_square_trainagle_regulized.h5')


    #x_test=x_test[1:3]

    result=model.predict(x_test)
    #print(result)
    print('*'*100)
    #print(y_test)

    print("+"*100)



    #model.save_weights('../models/model_weights_square_trainagle.h5')
    # print("/"*500)
    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print(weights)
    #     print("*" * 100)


    #function(model)
    train_train_test_to_draw = np.concatenate(x_test, axis=0)
    y_prediction_ = np.repeat(result, 1)

    plot_square_trainagle_output(signal, y_prediction_)


import matplotlib.pyplot as plt
def plot_square_trainagle_output(data,prediction):
    plt.plot(np.array(data), label="data")
    plt.plot(np.array(prediction), label="prediction")

    plt.title("triangle/square prediction")
    #plt.legend(loc='down right')
    plt.ylabel('voltage')
    plt.xlabel('time')

    plt.show()

def train_memsctrnn_debug(batch_size=100):
    x_train, y_train, x_test, y_test = read_dataset()
    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    tau_recip = 0.7
    rnn_1 = SimpleMEMSCTRNN(1, activation='tanh', return_sequences=True,return_state=True,tau_reciprocal=tau_recip)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=rnn_1)

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',optimizer='sgd',metrics=['mae'])


    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size)
    testModel(model)
def testModel(model):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01, gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    model_signal = [0] * 20
    # ctrnn_singal=[0]*20



    for i in range(20, gen_samples):
        input_ = np.array(corruptedSignal[i - 20:i]).reshape(1, 20, 1)
        model_ouput = model.predict(input_)
        model_signal.append(model_ouput)

    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t) for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    plt.plot(corruptedSignal)
    plt.plot(model_signal, '.')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()


def generate_statistics(zList):
    a1D = np.array(zList)
    maximum = max(a1D)
    minimum = min(a1D)
    mean = np.average(a1D)
    std=np.std(a1D)
    return maximum,minimum,mean,std


def inspect_z_values(path='../models/model_weights.h5', factor=10):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01,
                                                  gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    model_signal = [0] * 20

    #input_layer = Input((None, 1))
    tau_recip = 0.7
    input_layer = Input((None, 1))
    alpha_ = 1 * (10 ** 5)
    displacement_ = 5.0 * (10 ** -6)
    stopper_ = .5 * (10 ** -6)
    z_factor_ = 0.89921
    v_factor_ = 1.9146 * (10 ** -20)
    hidden_1, v_1, z1 = SimpleMEMSCTRNN(5, return_sequences=True,return_state=True, alpha=alpha_,
                            displacement=displacement_,
                            stopper=stopper_,
                            z_factor=z_factor_,
                            v_factor=v_factor_)(input_layer)

    hidden_2, v_2, z2 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,return_state=True, alpha=alpha_,
                               displacement=displacement_,
                               stopper=stopper_,
                               z_factor=z_factor_,
                               v_factor=v_factor_)(hidden_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_, vo, zo = SimpleMEMSCTRNN(1, activation=None, return_state=True, alpha=alpha_,
                              displacement=displacement_,
                              stopper=stopper_,
                              z_factor=z_factor_,
                              v_factor=v_factor_)(hidden_2)





    # hidden_1, v_1, z1 = SimpleMEMSCTRNN(5, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip,
    #                                     return_state=True)(input_layer)
    #
    # hidden_2, v_2, z2 = SimpleMEMSCTRNN(2, activation='tanh', return_sequences=True, return_state=True,
    #                                     tau_reciprocal=tau_recip, )(hidden_1)
    # # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    # output_, vo, zo = SimpleMEMSCTRNN(1, activation='tanh', tau_reciprocal=tau_recip, return_sequences=True,
    #                                   return_state=True)(hidden_2)
    model = tf.keras.Model(inputs=[input_layer],
                           outputs=[output_, vo, v_2, v_1, zo, z2, z1])  # output list [output_,...]
    model.compile(loss=['mse'] ,
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'])
    print(tf.__version__)
    print(model.summary())

    # model = tf.keras.models(path, custom_objects={"SimpleMEMSCTRNN": SimpleMEMSCTRNN})
    # model_weights=tf.keras.load_weights(path)
    model.load_weights(path)
    # ctrnn_singal=[0]*20

    zsList = []

    for i in range(20, gen_samples):
        input_ = np.array(corruptedSignal[i - 20:i]).reshape(1, 20, 1)
        model_ouput, vo, v_2, v_1, zo, z2, z1 = model.predict(input_ * factor)
        # print('input shape:  ', model_ouput.shape)
        # print('all_state_h shape: ', model_ouput.shape)
        # print('\nhidden states for the first sample: \n', model_ouput[0])
        # print('\nhidden states for the first sample at the second time step: \n', model_ouput[0][1])
        # print("\n z:", model_ouput[2])
        # zList.append(model_ouput[2][0][0])
        # print('\nsig(z-d):',
        #       tf.keras.activations.sigmoid(tf.math.multiply(model_ouput[2][0][0] - 41.8 * (10 ** -6), 5000)))
        print(z1)
        print(z2)
        print(zo)
        z=[]
        for i in z1:
            for j in i:
                z.append(j)
        for i in z2:
            for j in i:
                z.append(j)
        for i in zo:
            for j in i:
                z.append(j)
        print('*'*30)
        print(z)
        print('----'*30)

        zsList.append(z)






        model_signal.append(model_ouput[0][0])

    # textfile = open("z_values_factor12.txt", "w")
    #
    # max, min, avrg, std = generate_statistics(zList)
    # print(max, " ", max, " ", avrg, " ", std)
    #
    # for element in zList:
    #     textfile.write(str(element) + "\n")
    # textfile.write("\nZ Statitics : " + "\n")
    # textfile.write("*" * 100)
    #
    # textfile.write("\nmax : " + str(max) + "\n")
    # textfile.write("min : " + str(min) + "\n")
    # textfile.write("avrg : " + str(avrg) + "\n")
    # textfile.write("std : " + str(std) + "\n")
    # textfile.close()
    #print(zsList)
    import csv

    with open('z-values.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(zsList)
    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t) * factor for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    plt.plot(corruptedSignal * factor)
    plt.plot(model_signal, '-')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()

def testMode_modified(model,factor=1,window_size=20,shift=0):
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
        testx=(input_+shift)*factor
        model_ouput = model.predict(testx)
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
    x_gt = [(np.sin(2 * np.pi * t)+shift)*factor for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    shifted_signal=(corruptedSignal+shift)*factor
    plt.plot(shifted_signal)
    plt.plot(model_signal, '-')
    #plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, Green:CTRNN')
    plt.show()
def train_deniosing_problem_all_cell_mems(factor=1):
    _shift=0
    x_train, y_train, x_test, y_test = read_dataset(dataset_path=,shift=_shift)
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test

    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((20, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)




    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    #output_ = SimpleRNN(1, activation='linear')(hidden_1)
    output_ = SimpleMEMSCTRNN(1, activation='linear',alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(hidden_1)

    #output_ = SimpleRNN(1, activation='linear')(hidden_1)

    ##
    # rnn_1 = SimpleRNN(units=5, activation='sigmoid', return_sequences=True)(input_layer)
    # hidden_1 = SimpleRNN(2, activation='sigmoid', return_sequences=True)(rnn_1)
    # output_ = SimpleRNN(1, activation='linear')(hidden_1)

    ##
    model = models.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=32)
    model.save_weights('../models/model_weights1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-"*300)
    #get_weights_custom_function(model, [5,2,1])
    testMode_modified(model, factor,shift=_shift)
def draw_deniosing_problem_all_cell_mems(factor=1):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 25
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleMEMSCTRNN(1, activation=None, return_state=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    #model.fit(x_train, y_train, epochs=epochs_, batch_size=32)
    model.load_weights('../models/model_weights.h5')

    get_weights_custom_function(model, [5,2,1])
    testMode_modified(model, factor)
def watch_z_v_values_traingle_square(factor=1):
    x_train, y_train, x_test, y_test = read_train_square_trainagle()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    rnn_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(1, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)

    model = tf.keras.Model(inputs=input_layer, outputs=hidden_1)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=128)
    model.load_weights('../models/model_weights_square_trainagle.h5')

    #model.save_weights('../models/model_weights.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    testMode_modified(model, factor)

def check_change_paratmers(factor=10):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    factor_noise=1
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)* factor_noise
    stopper_=.5 * (10 ** -6) * factor_noise
    z_factor_=0.89921 * factor_noise
    v_factor_=1.9146 * (10 ** -20) * factor_noise
    #v_factor_ =2.2308 * (10 ** -20) * factor_noise
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleMEMSCTRNN(1, activation=None, return_state=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    #model.fit(x_train, y_train, epochs=epochs_, batch_size=128)
    model.load_weights('../models/model_weights.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    testMode_modified(model, factor)




def predict_output(examples, hidden_units=5,model=None):
    wx = model.get_weights()[0]
    wh = model.get_weights()[1]
    bh = model.get_weights()[2]
    w1 = model.get_weights()[3]
    b1 = model.get_weights()[4]
    w2 = model.get_weights()[5]
    b2 = model.get_weights()[6]
    outputP = []
    # out(:, :, t) = (1-ts)*out(:, :, t - 1)+ts*(tanh( x(:, :, t) * wx + out(:, :, t - 1) * wh) + repmat(b, M, 1) );
    for x in examples:
        m = hidden_units
        h0 = np.zeros(m)

        hn = np.tanh(np.dot(x[0], wx)) + h0 + bh
        for i in range(20):
            hp = hn;
            hn = np.tanh(np.dot(x[i], wx) + np.dot(hp, wh)) + bh
        o3 = np.dot(hn, w1) + b1
        o4 = np.dot(o3, w2) + b2
        outputP.append(o4)
    return outputP

def check_the_input_output(factor=10):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    output_ = SimpleMEMSCTRNN(5, activation=None, return_state=False,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    print("Before")
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=128)
    model.save_weights('../models/model_weights.h5')
    print("After")

    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    testMode_modified(model, factor)

def watch_LSTM_values(factor=10):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    tau_recip = 0.7
    rnn_1 = LSTM(10, activation='tanh', return_sequences=True)(input_layer)

    hidden_1 = LSTM(5, activation='tanh', return_sequences=True)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = LSTM(1, activation='tanh', return_state=True)(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=SGD(learning_rate = 0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=128)
    model.save_weights('../models/model_weights_lstm.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    testMode_modified(model,factor)

def train_memsctrnn(batch_size=100):
    x_train, y_train, x_test, y_test = read_dataset()
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    tau_recip = 0.7
    rnn_1 = SimpleMEMSCTRNN(3,  activation='tanh', return_sequences=False)(input_layer)
    rnn_2 = SimpleMEMSCTRNN(2 , activation='tanh', return_sequences=False)(input_layer)

    hidden_1 = Dense(5, activation='relu')(concatenate([rnn_1, rnn_2]))
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = Dense(1, activation='linear')(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer='sgd',
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae']
                  )
    # train the network

    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    testModel(model)


def generate_traingle(number_of_point,peak=10.0):
    half=(number_of_point-1)/2
    increment=peak/half
    traingle=[]
    prevval=0
    for i in np.arange(0,half):
        traingle.append(prevval)
        prevval+=increment
    secondHalf=traingle.copy()
    secondHalf.reverse()
    traingle.append(peak)
    traingle.extend(secondHalf)
    return traingle


def generate_square(number_of_point,peak=10.0):
    square=[]
    square.append(0.0)
    for i in range(number_of_point-2):
        square.append(peak)

    square.append(0)
    return square


def generate_traingle_square_dataset():
    square  = [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 6] # last number is the label 0 square
    traingle= [0, 2, 4,  6,  8,  10,  8,  6,  4, 2, 0, 1]   # last number is the label 1 triangle

    #square  = [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 3] # last number is the label 0 square
    #traingle= [0, 1, 2,  3,  4,  5,  4,  3,  2, 1, 0, -1]   # last number is the label 1 triangle
    # square=generate_square(21,10)
    # traingle=generate_traingle(21,10)
    # square.append(1)
    # traingle.append(5)
    traingles=[]
    squares=[]
    for i in np.arange(1,300,1):
        tmp=np.multiply(traingle, 1)
        traingles.append(tmp)
        tmp = np.multiply(square, 1)
        squares.append(tmp)
    return (np.array(traingles)+3, np.array(squares)+3)


def read_train_square_trainagle():
    (trainagles, squares) = generate_traingle_square_dataset()
    arr=np.concatenate((trainagles, squares), axis=0)
    np.random.shuffle(arr)
    dataLenght=arr.shape[1]-1
    trainx= arr[0:500, 0:dataLenght]
    trainy= arr[0:500, dataLenght:dataLenght+1]
    testx = arr[500:, 0:dataLenght]
    testy = arr[500:, dataLenght:dataLenght+1]
    return trainx,trainy,testx,testy
def generate_paramers(area, omega, tau, epsolin,k,deltaT):
    tmp = deltaT/tau
    z_factor=1-tmp
    print("z_factor", z_factor)
    a=tmp*(epsolin*area/(2*k))
    print("a",a)


def train_deniosing_problem_all_cell_mems_window_size(factor=1, window_size=1):
    x_train, y_train, x_test, y_test = read_dataset(dataset_path="../Data/denoising_signal_1_window_size.csv", window_size=window_size)
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 5
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    alpha_=1 * (10 ** 5)
    displacement_=5.0 * (10 ** -6)
    stopper_=.5 * (10 ** -6)
    z_factor_=0.89921
    v_factor_=1.9146 * (10 ** -20)
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleMEMSCTRNN(1, activation=None, return_state=True,alpha=alpha_,
               displacement=displacement_,
               stopper=stopper_,
               z_factor=z_factor_,
               v_factor=v_factor_)(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    print(tf.__version__)
    print(model.summary())
    # enable this if pydot can be installed
    # pip install pydot
    # plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate = 0.001),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=32)
    model.save_weights('../models/model_weights_window_size1.h5')
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-"*300)
    get_weights_custom_function(model, [10,10,1])
    testMode_modified(model, factor,window_size=1)


if __name__ == '__main__':
    print('hi main')
    # generate_paramers(area=10**-7,omega=50392,tau=9.9222*(10**-6),epsolin=8.854 * (10**-12), k=2.3303,deltaT=10**-6)
    # generate_paramers(area=10 ** -7, omega=50392, tau=9.9222 * (10 ** -6), epsolin=8.854 * (10 ** -12), k=2.0,deltaT=10 ** -6)
    #generate_paramers(area=10 ** -7, omega=50392, tau=9.9222 * (10 ** -6), epsolin=8.854 * (10 ** -12), k=2.6,deltaT=10 ** -6)
    factor_=1
    train_deniosing_problem_all_cell_mems(factor=factor_)
    #GenericFunctions.generate_dataset(windows_size=5 ,batch_size=16)
    #train_deniosing_problem_all_cell_mems_window_size()
    #draw_deniosing_problem_all_cell_mems(factor=factor_)
    #inspect_z_values(path='../models/model_weights.h5', factor=factor_)

    #check_change_paratmers(factor=factor_)
    # print(generate_square(101,10))
    # print(generate_traingle(101, 10))
    #x,y=generate_signal_square_traingle.generate_dataset()
    #train_square_trainagle()
    #watch_z_v_values_traingle_square()
    #check_the_input_output(factor=factor_)

    #train_memsctrnn_all_mems(batch_size=64,factor=13)
    #train_memsctrnn_debug()



##############
