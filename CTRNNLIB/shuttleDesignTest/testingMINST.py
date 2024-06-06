import pickle

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from CTRNNLIB.Recurrent import SimpleCTRNN
from CTRNNLIB.shuttleDesignTest import WISDM_Prepocessing

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

model = Sequential()
model.add(SimpleRNN(8, input_shape=(x_train.shape[1:]), activation='relu'))

#model.add(SimpleCTRNN(8, activation='relu',h=.01,input_frequency=30))

model.add(Dense(8, activation='relu'))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

history=model.fit(x_train,
          y_train,
          epochs=10,
          validation_data=(x_test, y_test))

model.save('./MasterModels/ctrnn_mnist_model.keras')
with open('./MasterModels/ctrnn_mnist_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
model.evaluate(x_test, y_test)
#WISDM_Prepocessing.drawHistory('./MasterModels/Mnist/RNN/rnn_mnist_history')
#print(model.layers[0].cell.tau)
# for layer in model.layers:
#     print(layer.get_weights())