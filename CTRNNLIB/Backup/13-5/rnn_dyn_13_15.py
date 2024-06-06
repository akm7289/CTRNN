from keras import activations, Sequential
import tensorflow as tf
from tensorflow.keras.layers import  TimeDistributed, Dense, TimeDistributed, SimpleRNN, Input,concatenate
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import SGD

from CTRNNLIB.Recurrent import SimpleCTRNN, SimpleMEMSCTRNN, LSTM


def read_dataset():
    df = pd.read_csv(filepath_or_buffer="../Data/denoising_signal_100.csv")
    training = df.iloc[:40000, :]
    testing = df.iloc[45000:, :]
    traininX = training.iloc[:, 0:20]
    x_train = np.array(training.iloc[:, 0:20])
    y_train = np.array(training.iloc[:, 20:21])

    x_test = np.array(testing.iloc[:, 0:20])
    y_test = np.array(testing.iloc[:, 20:21])
    print("***************")
    print("data has been loaded successfully")
    print((x_train.shape))
    print("***************")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test

def train_rnn(batch_size=100):
    x_train, y_train,x_test,y_test=read_dataset()
    epochs_ = 100
    #Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None,1))
    tau_recip=0.75
    rnn_1 = SimpleCTRNN(1, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip)(input_layer)
    rnn_2 = SimpleCTRNN(1, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip)(input_layer)
    rnn_3 = SimpleCTRNN(1, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip)(input_layer)
    rnn_4 = SimpleCTRNN(1, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip)(input_layer)
    rnn_5 = SimpleCTRNN(1, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip)(input_layer)

    hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1, rnn_2, rnn_3, rnn_4, rnn_5]))
    #hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = TimeDistributed(Dense(1, activation='linear'))(hidden_1)
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

    testModel(model)
def train_ctrnn_1(batch_size=100):
    x_train, y_train,x_test,y_test=read_dataset()
    epochs_ = 10
    #Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None,1))
    tau_recip=0.7
    rnn_1 = SimpleCTRNN(2, activation='tanh', return_sequences=False, tau_reciprocal=tau_recip)(input_layer)
    rnn_2 = SimpleCTRNN(2, activation='tanh', return_sequences=False, tau_reciprocal=tau_recip)(input_layer)


    hidden_1 = Dense(5, activation='relu')(concatenate([rnn_1, rnn_2]))
    #hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
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
    testModel(model)


def train_ctrnn_2(batch_size=100):
    x_train, y_train,x_test,y_test=read_dataset()
    #Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size = 100
    step = 20
    input_shape = (step, 1)
    units = 5
    dropout = 0
    epochs_ = 20

    # model is RNN with x units, input is 1-dim vector 20 timesteps
    model = Sequential()
    model.add(SimpleCTRNN(units=units,tau_reciprocal=.86,
                        dropout=dropout,
                        input_shape=input_shape, activation='tanh'))
    model.add(Dense(5, activation='relu'))

    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.summary()
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


    _, error_metric = model.evaluate(x_test,
                                     y_test,
                                     batch_size=batch_size,
                                     verbose=0)

    print("test set eval-metric:", error_metric)
    testModel(model)


def train_rnn_3(batch_size=100):
    x_train, y_train,x_test,y_test=read_dataset()
    #Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size = 100
    step = 20
    input_shape = (step, 1)
    units = 5
    dropout = 0
    epochs_ = 10

    # model is RNN with x units, input is 1-dim vector 20 timesteps
    model = Sequential()
    model.add(SimpleCTRNN(units=units,tau_reciprocal=.86,
                        dropout=dropout,
                        input_shape=input_shape, activation='tanh'))
    model.add(Dense(5, activation='relu'))

    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.summary()
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


    _, error_metric = model.evaluate(x_test,
                                     y_test,
                                     batch_size=batch_size,
                                     verbose=0)

    print("test set eval-metric:", error_metric)
    testModel(model)



import GenericFunctions

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
    model = tf.keras.Model(inputs=input_layer, outputs=output_)# output list [output_,...]

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

    model.compile(loss='mse',
                  optimizer='sgd',
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae']
                  )
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

    input_layer = Input((None, 1))
    tau_recip = 0.7
    hidden_1, v_1, z1 = SimpleMEMSCTRNN(5, activation='tanh', return_sequences=True, tau_reciprocal=tau_recip,
                                        return_state=True)(input_layer)

    hidden_2, v_2, z2 = SimpleMEMSCTRNN(2, activation='tanh', return_sequences=True, return_state=True,
                                        tau_reciprocal=tau_recip, )(hidden_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_, vo, zo = SimpleMEMSCTRNN(1, activation='tanh', tau_reciprocal=tau_recip, return_sequences=True,
                                      return_state=True)(hidden_2)
    model = tf.keras.Model(inputs=[input_layer],
                           outputs=[output_, vo, v_2, v_1, zo, z2, z1])  # output list [output_,...]
    model.compile(loss=['mse'] * 5,
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






        model_signal.append(model_ouput[0][0][0])

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
    plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()

def testMode_modified(model,factor=1):
    print("start testing")
    # change gen_samples
    gen_samples = 1000
    fun_gen = GenericFunctions.function_generator(training=False, noise_multiplier=.25, N=5, sample_delay=0.01, gen_samples=gen_samples)
    corruptedSignal = fun_gen(0)
    model_signal = [0] * 20
    # ctrnn_singal=[0]*20

    zList=[]

    for i in range(20, gen_samples):
        input_ = np.array(corruptedSignal[i - 20:i]).reshape(1, 20, 1)
        model_ouput = model.predict(input_*factor)
        # print('input shape:  ', model_ouput.shape)
        # print('all_state_h shape: ', model_ouput.shape)
        # print('\nhidden states for the first sample: \n', model_ouput[0])
        # print('\nhidden states for the first sample at the second time step: \n', model_ouput[0][1])
        # print("\n z:",model_ouput[2])
        zList.append(model_ouput[2][0][0])
        # print('\nsig(z-d):',tf.keras.activations.sigmoid(tf.math.multiply(model_ouput[2][0][0] - 41.8 * (10 ** -6), 5000)))
        model_signal.append(model_ouput[0])


    textfile = open("z_values_factor12.txt", "w")


    max,min,avrg,std=generate_statistics(zList)
    print(max," ",max," ", avrg," ", std)

    for element in zList:
        textfile.write(str(element) + "\n")
    textfile.write("\nZ Statitics : "+"\n")
    textfile.write("*"*100)

    textfile.write("\nmax : "+str(max) + "\n")
    textfile.write("min : " + str(min) + "\n")
    textfile.write("avrg : " + str(avrg) + "\n")
    textfile.write("std : " + str(std) + "\n")
    textfile.close()

    t_samples = [i / 100 for i in range(gen_samples)]
    x_gt = [np.sin(2 * np.pi * t)*factor for t in t_samples]
    from numpy.core.numeric import outer
    import matplotlib.pyplot as plt
    plt.plot(corruptedSignal*factor)
    plt.plot(model_signal, 'o')
    plt.plot(x_gt, 'k')
    plt.ylabel('sin(2w*pi*x)')
    plt.xlabel('x  black:orginal, blue:noisy orange:RNN signal, Green:CTRNN')
    plt.show()
def watch_z_v_values(factor=10):
    x_train, y_train, x_test, y_test = read_dataset()
    x_train = factor * x_train
    y_train = factor * y_train
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 10
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None, 1))
    tau_recip = 0.7
    rnn_1 = SimpleMEMSCTRNN(5, activation=None, return_sequences=True, tau_reciprocal=tau_recip)(input_layer)

    hidden_1 = SimpleMEMSCTRNN(2, activation=None, return_sequences=True, tau_reciprocal=tau_recip)(rnn_1)
    # hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1]))
    output_ = SimpleMEMSCTRNN(1, activation=None, tau_reciprocal=tau_recip, return_state=True)(hidden_1)
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
    model.save_weights('../models/model_weights.h5')
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
    rnn_1 = SimpleMEMSCTRNN(3,  activation='tanh', return_sequences=False, tau_reciprocal=tau_recip)(input_layer)
    rnn_2 = SimpleMEMSCTRNN(2 , activation='tanh', return_sequences=False, tau_reciprocal=tau_recip)(input_layer)

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






if __name__ == '__main__':
    print('hi main')
    #train_ctrnn_2()
    #train_ctrnn_1()
    #train_rnn_3()
    #train_memsctrnn()
    factor_=13
    #watch_LSTM_values(factor_)
    watch_z_v_values(factor=factor_)
    #inspect_z_values(path='../models/model_weights.h5', factor=factor_)
    #train_memsctrnn_all_mems(batch_size=64,factor=13)
   #train_memsctrnn_debug()

