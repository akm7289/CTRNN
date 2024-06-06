# The function generator provides an instantiation of a noisy environment that can be sampled.
# (N+1) is the number of microphones
import numpy as np
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from keras.constraints import MinMaxNorm
from keras.models import Sequential
from keras.layers import RNN,GRU,LSTM, Dense, Flatten, Dropout
from scipy import stats
from keras.layers import RNN
from sklearn.model_selection import train_test_split
import pprint
from tensorflow import keras
from tensorflow import keras
from keras.layers import SimpleRNN
from CTRNNLIB.Recurrent import SimpleCTRNN
import sklearn.metrics as metrics


# The function generator provides an instantiation of a noisy environment that can be sampled.
# This provides multiple microphone signals
# N is the number of microphones


# The function generator provides an instantiation of a noisy environment that can be sampled.
# Use gt_direction to specify the source direction
def function_generator(noise_multiplier=0.01, N=1, sample_delay=0.01, gen_samples=10, gt_direction=1):
    # Selected Parameters:
    A = 1.0
    delta = 2 * np.pi * 0.5
    gt_amplitude = A

    # Physical parameters
    c_sound = 343
    f_sound = 500
    wavelen = c_sound / f_sound
    spacing = wavelen / 2
    max_delay = 1 / f_sound / 2
    delay1 = max_delay * np.sin(-np.pi / 2)
    delay2 = max_delay * np.sin(-np.pi / 4)
    delay3 = max_delay * np.sin(np.pi * 0)
    delay4 = max_delay * np.sin(np.pi / 4)
    delay5 = max_delay * np.sin(np.pi / 2)

    if gt_direction == 1:
        delay = delay1
    elif gt_direction == 2:
        delay = delay2
    elif gt_direction == 3:
        delay = delay3
    elif gt_direction == 4:
        delay = delay4
    elif gt_direction == 5:
        delay = delay5
    else:
        print("Not an allowed direction")

    # Function to be called by the ode solver. The function can be evaluated at any time, t, to mimic the behavior
    # of a continuous system.
    def return_function(t):
        t_samples = np.array(np.arange((t - gen_samples * sample_delay), t, sample_delay))[:gen_samples]
        x_base1 = gt_amplitude * np.sin(2 * np.pi * f_sound * t_samples)
        x1 = x_base1 + noise_multiplier * (np.random.randn(gen_samples))
        delay_gt = delay
        x_base2 = gt_amplitude * np.sin(2 * np.pi * f_sound * t_samples - delay_gt)
        x2 = x_base2 + noise_multiplier * (np.random.randn(gen_samples))

        x_gt = delay_gt

        #return (x1, x2, x_gt)
        return (x1, x2, gt_direction-1)

    # Return the signal environment with the selected noise parameters:
    return return_function
def generate_training_data(num_of_generator=16, signal_frequency=300, window_size=1, rows=10000, MEMS_freq=10 ** -6,
                           enviroment_end_time=3392 * 10 ** -6, N_=5):  # number of generator =number of batch-size
    generators_lst = []
    enviroment_end_time = int(enviroment_end_time * 10 ** 6)
    for i in range(num_of_generator):
        generators_lst.append(function_generator(noise_multiplier=1, N=N_, sample_delay=MEMS_freq,
                                                 gen_samples=enviroment_end_time, gt_direction=i + 1))
    X1_list = []
    X2_list = []
    groundTruthList = []
    generatorCounter = 0
    while (generatorCounter < num_of_generator):
        generatorID = generatorCounter % num_of_generator
        time = 0
        # X = np.random.normal(size=(n_batches, batch_size, seq_len, 1))
        X1 = []
        X2=[]
        Y = []
        microphone1,microphone2, gt_signal = generators_lst[generatorID](time * MEMS_freq)
        gt_signal = np.ones(microphone1.shape) * gt_signal
        for i in range(enviroment_end_time // window_size):
            start_index = i * window_size
            X1.append(microphone1[start_index:start_index + window_size])
            X2.append(microphone2[start_index:start_index + window_size])
            Y.append(gt_signal[start_index])
        generatorCounter += 1
        # plt.plot(np.array(X).reshape(-1))
        # plt.plot(np.array(Y).reshape(-1))
        X1_list=X1_list+X1
        X2_list=X2_list+X2
        groundTruthList=groundTruthList+Y
        #X1_list.append(np.array(X1).reshape(len(X1), window_size, 1))
        #X2_list.append(np.array(X2).reshape(len(X2), window_size, 1))

        #groundTruthList.append(np.array(Y).reshape(len(Y), 1, 1))
    X1_list = np.array(X1_list)
    X2_list=np.array(X2_list)
    training=np.stack((X1_list, X2_list), axis=1)

    groundTruthList = np.array(groundTruthList)
    # plt.plot(groundTruthList.reshape(-1))
    # plt.plot(X1_list.reshape(-1))
    # plt.plot(X2_list.reshape(-1))

    return training,groundTruthList


def tmp():
    Xtraining,Ytraining  = generate_training_data(window_size=50, num_of_generator=5, enviroment_end_time=10 * 3392 * 10 ** -6,
                                            N_=1)

    # excel_array = np.array([microphone1.reshape(-1), micrphone2.reshape(-1),direction.reshape(-1)])
    # excel_array = np.transpose(excel_array)
    # df = pd.DataFrame(excel_array, columns=['microphone1', 'microphone2','direction'])
    # # df.to_csv("./NewTrainingData/microphone_latest_data.csv", index=False)
    # # testing.to_csv("./NewTrainingData/testing_10window_size.csv", header=False, index=False)
    #
    # plt.plot(microphone1.reshape(-1), '-')
    # plt.plot(micrphone2.reshape(-1), '-')
    #
    #
    # plt.plot(direction.reshape(-1)*1000)
    #
    # plt.show()

def trainRNN(x_train, y_train,x_test, y_test,number_of_epoches_ = 20):

    batch_size_ = 512
    # check if the data is ready for training by using simple model
    print("Simple RNN")
    input_shape_ = (x_train.shape[1], x_train.shape[2])

    input_layer_ = keras.Input(shape=input_shape_, name="mnist_input_")
    layer1_ = SimpleRNN(units=16, activation='tanh')(input_layer_)
    layer2_ = Dense(units=10, activation='relu')(layer1_)
    output_layer_ = Dense(units=6, activation='softmax')(layer2_)
    rnn1 = keras.Model(input_layer_, output_layer_, name="rnn1")
    rnn1.summary()
    rnn1.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    history = rnn1.fit(x_train, y_train, batch_size=512, epochs=number_of_epoches_, validation_split=0.1)
    rnn1.save('./MasterModels/rnn_model.keras')
    with open('./MasterModels/rnn_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    rnn1.evaluate(x_test, y_test)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    #tmp()
    Xtraining, Ytraining = generate_training_data(window_size=50, num_of_generator=5,
                                                  enviroment_end_time=10 * 3392 * 10 ** -6,
                                                  N_=1)
    import numpy as np

    # Sample class labels (you should replace this with your actual data)

    # Define the number of classes
    num_classes = 5

    # Convert to one-hot encoding
    Ytraining = [int(label) for label in Ytraining]  # Convert to integers

    one_hot_encoded = np.eye(num_classes)[Ytraining]

    # Display the one-hot encoded data
    print(one_hot_encoded)
    trainRNN(Xtraining, one_hot_encoded,Xtraining, one_hot_encoded,number_of_epoches_ = 20)
    # fg=function_generator(gt_direction=5)
    # for i in range(100):
    #     print(fg(i/100))
