from keras import activations
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNNCell, TimeDistributed, Dense, TimeDistributed, SimpleRNN, Input,concatenate
import pandas as pd
import numpy as np
def read_dataset():
    df = pd.read_csv(filepath_or_buffer="./Data/denoising_signal_100.csv")
    training = df.iloc[:40000, :]
    testing = df.iloc[45000:, :]
    traininX = training.iloc[:, 0:20]
    x_train = np.array(training.iloc[:, 0:20])
    y_train = np.array(training.iloc[:, 20:21])

    x_test = np.array(testing.iloc[:, 0:20])
    y_test = np.array(testing.iloc[:, 20:21])
    print("***************")
    print((x_train.shape))
    print("***************")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test

def train_rnn_SimpleRNNCell(batch_size=100):
    x_train, y_train,x_test,y_test=read_dataset()
    epochs_ = 10
    #Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    input_layer = Input((None,1))
    rnn_1 = SimpleRNN(1, activation='tanh', return_sequences=True)(input_layer)
    rnn_2 = SimpleRNN(1, activation='tanh', return_sequences=True)(input_layer)
    rnn_3 = SimpleRNN(1, activation='tanh', return_sequences=True)(input_layer)
    rnn_4 = SimpleRNN(1, activation='tanh', return_sequences=True)(input_layer)
    rnn_5 = SimpleRNN(1, activation='tanh', return_sequences=True)(input_layer)

    hidden_1 = TimeDistributed(Dense(5, activation='relu'))(concatenate([rnn_1, rnn_2, rnn_3, rnn_4, rnn_5]))
    output_ = TimeDistributed(Dense(1, activation='linear'))(hidden_1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_)

    print(tf.__version__)
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


if __name__ == '__main__':
    print('hi main')
    train_rnn_SimpleRNNCell()