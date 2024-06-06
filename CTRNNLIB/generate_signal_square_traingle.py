import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


def generate_dataset(shuffle=True):
    freq = 2
    amp = 4
    time = np.linspace(0, 100, 100)
    shift = 7

    signal1 = amp * np.sin(2 * np.pi * freq * time) + shift

    # plt.plot(time, signal1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    signal2 = amp * sg.square(2 * np.pi * freq * time, duty=0.95) + shift

    # plt.figure(figsize=(10,4))
    # plt.plot(time, signal2)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()
    signal3 = amp * sg.sawtooth(2 * np.pi * freq * time, width=0.5) + shift

    # plt.plot(time, signal3)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    training_signal = np.concatenate((signal2, signal3, signal2[0:50], signal3[0:50],signal3,signal2))
    newtime = np.linspace(0, 500, 500)
    rectangle = 4
    traingle = 9
    labels = np.concatenate(
        (np.full((100), rectangle), np.full((100), traingle), np.full((50), rectangle), np.full((50), traingle),np.full((100), traingle), np.full((100), rectangle))
    )

    dataset = pd.DataFrame({'label': labels, 'data': training_signal}, columns=['label', 'data'])
    # print("start drawing")
    # plt.plot(newtime, training_signal, label="data")
    # plt.plot(newtime, labels, label="label")
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # leg = plt.legend(loc='lower left')

    plt.show()
    d, t = generate_sequence_data(dataset, 15, 1, 0.5)
    d = np.array(d)
    t = np.array(t)
    if shuffle:
        shuffler = np.random.permutation(len(d))
        d = d[shuffler]

        t = t[shuffler]
    d = d.reshape(d.shape[0], d.shape[2])
    t = t.reshape(t.shape[0], 1)
    return d, t,training_signal


def generate_sequence_data(dataframe, window_size, sliding, precentage_ignore=0.5):
    print('window_size: ', window_size)
    print('sliding: ', sliding)
    print('precentage_ignore: ', precentage_ignore)

    index_ = 0
    trainingX = []
    trainingY = []
    while (index_ < len(dataframe) - window_size):
        tmp = stats.mode(dataframe['label'][index_: index_ + window_size])
        mode = tmp[0][0]
        mode_count = tmp.count[0]
        percentage = mode_count / window_size

        end_index = index_ + window_size
        if percentage >= precentage_ignore:
            y = dataframe['data'].values[index_: end_index]
            trainingX.append([y])
            trainingY.append(mode)

        index_ += sliding

    X = np.asarray(trainingX, dtype=np.float32).reshape(-1, window_size, 1)
    hot_encoding = pd.get_dummies(trainingY)
    Y = np.asarray(hot_encoding, dtype=np.float32)
    assert (X.shape[0] == Y.shape[0])
    return trainingX,trainingY


