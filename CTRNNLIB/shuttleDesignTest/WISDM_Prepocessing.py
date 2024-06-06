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




def cleanData(df):
    # cleaning

    print(f'size before droping na :{df.shape}')
    df = df.dropna()
    print(f'size after droping na :{df.shape}')
    df.replace(';', '', regex=True, inplace=True)
    df['Z'] = df['Z'].apply(lambda z: float(z))
    print(df.head())


    df = df[df['Timestamp'] != 0]
    print("dataset after cleaning")
    print(df.head())
    return df


'''
normalize x,y & z
'''
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def normalize_data(dataframe, colums=[]):
    normalized_df = dataframe.copy()
    for col in colums:
        normalized_df[col] = MinMaxScaler().fit_transform(np.array(normalized_df[col]).reshape(-1, 1))
    return normalized_df


# plot acc
def show_acc(history):
    plt.plot(np.array(history['accuracy']), label="Training accuracy")
    plt.plot(np.array(history['val_accuracy']), label="Validation accuracy")

    plt.title("Training/Validation accuracy")
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.show()


# plot loss
def show_loss(history):
    plt.plot(np.array(history['loss']), label="Training loss")
    plt.plot(np.array(history['val_loss']), label="Validation loss")

    plt.title("Training/Validation loss")
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()


def generate_sequence_data(dataframe, window_size, sliding, precentage_ignore=0.5):
    print('window_size: ', window_size)
    print('sliding: ', sliding)
    print('precentage_ignore: ', precentage_ignore)

    index_ = 0
    trainingX = []
    trainingY = []
    while (index_ < len(dataframe) - window_size):
        tmp = stats.mode(dataframe['Activity'][index_: index_ + window_size])
        mode = tmp[0][0]
        mode_count = tmp.count[0]
        percentage = mode_count / window_size

        end_index = index_ + window_size
        if percentage >= precentage_ignore:
            x = dataframe['X'].values[index_: end_index]
            y = dataframe['Y'].values[index_: end_index]
            z = dataframe['Z'].values[index_: end_index]
            trainingX.append([x, y, z])
            trainingY.append(mode)

        index_ += sliding

    X = np.asarray(trainingX, dtype=np.float32).reshape(-1, window_size, 3)
    hot_encoding = pd.get_dummies(trainingY)
    Y = np.asarray(hot_encoding, dtype=np.float32)
    assert (X.shape[0] == Y.shape[0])
    return X, Y, hot_encoding


def generate_augment_sequence_data(dataframe, window_size, sliding, precentage_ignore=0.5):
    print('window_size: ', window_size)
    print('sliding: ', sliding)
    print('precentage_ignore: ', precentage_ignore)

    index_ = 0
    trainingX = []
    trainingY = []
    sliding_arg = sliding
    while (index_ < len(dataframe) - window_size):
        tmp = stats.mode(dataframe['Activity'][index_: index_ + window_size])
        mode = tmp[0][0]
        mode_count = tmp.count[0]
        percentage = mode_count / window_size

        end_index = index_ + window_size
        if percentage >= precentage_ignore:
            x = dataframe['X'].values[index_: end_index]
            y = dataframe['Y'].values[index_: end_index]
            z = dataframe['Z'].values[index_: end_index]
            trainingX.append([x, y, z])
            trainingY.append(mode)

        if mode == 'Downstairs' or mode == 'Upstairs':
            sliding = sliding_arg - 6
        elif mode == 'Sitting' or mode == 'Standing':
            sliding = sliding_arg - 8
        else:
            sliding = sliding_arg

        index_ += sliding

    X = np.asarray(trainingX, dtype=np.float32).reshape(-1, window_size, 3)
    hot_encoding = pd.get_dummies(trainingY)
    Y = np.asarray(hot_encoding, dtype=np.float32)
    assert (X.shape[0] == Y.shape[0])
    return X, Y, hot_encoding


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
def loadRNN(x_test, y_test, model_path=r''):
    rnn1= keras.models.load_model(model_path)
    rnn1.evaluate(x_test, y_test)
    return rnn1


def printWeights(ctrnn_model):
    layer1 = ctrnn_model.layers[1]
    tau = layer1.cell.tau
    print(tau)
    print("*********************************")
    for layer in ctrnn_model.layers:
        print(layer.get_weights())
        print("*********************************")


def trainCTRNN(x_train, y_train,x_test, y_test,number_of_epoches_ = 20,h=.01):
    batch_size_ = 512
    # check if the data is ready for training by using simple model
    print("Simple CTRNN")
    input_shape_ = (x_train.shape[1], x_train.shape[2])
    min_tau=0
    max_tau=10

    input_layer_ = keras.Input(shape=input_shape_, name="mnist_input_")
    #layer1_ = SimpleCTRNN(units=16, activation='tanh',h=h,input_frequency=30,tau_constraint=MinMaxNorm(min_value=min_tau, max_value=max_tau))(input_layer_)
    layer1_ = SimpleCTRNN(units=16, activation='tanh',h=h,input_frequency=30)(input_layer_)
    layer2_ = Dense(units=10, activation='relu')(layer1_)
    output_layer_ = Dense(units=6, activation='softmax')(layer2_)
    ctrnn_model = keras.Model(input_layer_, output_layer_, name="CTRNN_Model")
    ctrnn_model.summary()
    ctrnn_model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #printWeights(ctrnn_model)

    history = ctrnn_model.fit(x_train, y_train, batch_size=batch_size_, epochs=number_of_epoches_, validation_split=0.1)
    with open('./MasterModels/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    ctrnn_model.save_weights('./MasterModels/model.h5')

    show_loss(history.history)
    show_acc(history.history)
    #printWeights(ctrnn_model)
    #keras.models.save_model(ctrnn_model, './MasterModels/model.h5')

    #ctrnn_model.save('./MasterModels/ctrnn_training_model.keras')
    ctrnn_model.evaluate(x_test, y_test)
def loadCTRNN(x_test, y_test,h=.01,model_path=r"",firstLayer=16,secondLayer=10):
    batch_size_ = 512
    # check if the data is ready for training by using simple model
    print("Simple CTRNN")
    input_shape_ = (x_test.shape[1], x_test.shape[2])
    min_tau=0
    max_tau=10

    input_layer_ = keras.Input(shape=input_shape_, name="mnist_input_")
    layer1_ = SimpleCTRNN(units=firstLayer, activation='tanh',h=h,input_frequency=30,tau_constraint=MinMaxNorm(min_value=min_tau, max_value=max_tau))(input_layer_)
    #layer1_ = SimpleCTRNN(units=16, activation='tanh',h=h,input_frequency=30)(input_layer_)
    layer2_ = Dense(units=secondLayer, activation='relu')(layer1_)
    output_layer_ = Dense(units=6, activation='softmax')(layer2_)
    ctrnn_model = keras.Model(input_layer_, output_layer_, name="rnn1")
    ctrnn_model.summary()
    ctrnn_model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    ctrnn_model.load_weights(model_path)
    ctrnn_model.evaluate(x_test, y_test)
    return ctrnn_model
    #printWeights(ctrnn_model)



def train_no_tau_training_CTRNN(x_train, y_train,x_test, y_test,number_of_epoches_ = 20,h=.01):
    batch_size_ = 512
    # check if the data is ready for training by using simple model
    print("Simple CTRNN")
    input_shape_ = (x_train.shape[1], x_train.shape[2])

    input_layer_ = keras.Input(shape=input_shape_, name="mnist_input_")
    layer1_ = SimpleCTRNN(units=16, activation='tanh',h=h,input_frequency=30)(input_layer_)
    layer2_ = Dense(units=10, activation='relu')(layer1_)
    output_layer_ = Dense(units=6, activation='softmax')(layer2_)
    ctrnn_model = keras.Model(input_layer_, output_layer_, name="rnn1")
    ctrnn_model.summary()
    ctrnn_model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    history = ctrnn_model.fit(x_train, y_train, batch_size=512, epochs=number_of_epoches_, validation_split=0.1)
    print(ctrnn_model.layers[1].cell.tau)
    with open('./MasterModels/ctrnn_no_tau_training_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # show_loss(history)
    # show_acc(history)
    ctrnn_model.save('./MasterModels/ctrnn_no_tau_training_model.keras')
    ctrnn_model.evaluate(x_test, y_test)

def drawHistory(file_path):
    with open(file_path, "rb") as file_pi:
        history = pickle.load(file_pi)
        show_loss(history)
        show_acc(history)

def preprocessing():
    df = pd.read_csv(r'D:\Master\Dataset\WISDM_ar_v1.1_raw.txt', header=None,
                     names=['ID', 'Activity', 'Timestamp', 'X', 'Y', 'Z'], on_bad_lines='skip')
    print(df.head())
    print(df.describe())
    print(df.shape)
    df = cleanData(df)

    print("Description about the data")
    print(df.describe())
    print("*" * 100)

    print(df['Activity'].value_counts())
    noralizedDF = normalize_data(df, ['X', 'Y', 'Z'])
    print(noralizedDF.head())
    Training_X, Training_Y, hot_encoding = generate_sequence_data(df, 50, 10, 0)
    x_train, x_test, y_train, y_test = train_test_split(Training_X, Training_Y, random_state=17, test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)

    Training_X, Training_Y, hot_encoding = generate_augment_sequence_data(df, 50, 10, 0)
    x_train, x_test, y_train, y_test = train_test_split(Training_X, Training_Y, random_state=17, test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)
    print(Training_Y)
    #trainCTRNN(x_train, y_train, x_test, y_test, number_of_epoches_=50)
    output = open('./WISDM_split_dataset/x_train.pkl', 'wb')
    pickle.dump(x_train, output)
    output.close()

    output = open('./WISDM_split_dataset/y_train.pkl', 'wb')
    pickle.dump(y_train, output)
    output.close()
    output = open('./WISDM_split_dataset/hot_encoding.pkl', 'wb')
    pickle.dump(hot_encoding, output)
    output.close()
    output = open('./WISDM_split_dataset/y_test.pkl', 'wb')
    pickle.dump(y_test, output)
    output.close()

    output = open('./WISDM_split_dataset/x_ticks.pkl', 'wb')
    pickle.dump(hot_encoding, output)
    output.close()


    pkl_file = open('./WISDM_split_dataset/x_test.pkl', 'rb')

    data1 = pickle.load(pkl_file)
    pprint.pprint(data1)

    pkl_file.close()

    print('end')

def load_data(file_path):
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
def Load_training_data():
    x_train=load_data('./WISDM_split_dataset/x_train.pkl')
    y_train=load_data('./WISDM_split_dataset/y_train.pkl')
    return x_train,y_train
def load_testing_data():
    y_test=load_data('./WISDM_split_dataset/y_test.pkl')
    x_test=load_data('./WISDM_split_dataset/x_test.pkl')
    return x_test,y_test


def trainBigRNN(x_train, y_train, x_test, y_test, number_of_epoches_):
    batch_size_ = 512
    # check if the data is ready for training by using simple model
    print("Simple RNN")
    input_shape_ = (x_train.shape[1], x_train.shape[2])

    input_layer_ = keras.Input(shape=input_shape_, name="mnist_input_")
    layer1_ = SimpleRNN(units=32, activation='tanh')(input_layer_)
    layer2_ = Dense(units=10, activation='relu')(layer1_)
    output_layer_ = Dense(units=6, activation='softmax')(layer2_)
    rnn1 = keras.Model(input_layer_, output_layer_, name="rnn1")
    rnn1.summary()
    rnn1.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    history = rnn1.fit(x_train, y_train, batch_size=512, epochs=number_of_epoches_, validation_split=0.1)
    rnn1.save('./MasterModels/big_rnn_model.keras')
    with open('./MasterModels/big_rnn_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    rnn1.evaluate(x_test, y_test)
    drawHistory('./MasterModels/big_rnn_history')


def draw_conf_matrix(trained_model,x_test,y_test,hot_encoding):
    predictions = trained_model.predict(x_test)
    test = np.argmax(y_test, axis=1)
    #hot_encoding = pd.get_dummies(y_test)
    predict = np.argmax(predictions, axis=1)
    confusion_matrix = metrics.confusion_matrix(test, predict)
    xticks=['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
    yticks=['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

    sns.heatmap(confusion_matrix, fmt='d', cmap='PuBuGn',xticklabels=hot_encoding.columns.tolist(),yticklabels=hot_encoding.columns.tolist(), annot=True)
    #
    plt.title("Confusion matrix")
    plt.xlabel('Predicted activity')
    plt.ylabel('Ground truth')
    plt.show()


def load_one_hot_encoding_data():
    one_hot_encoding = load_data('./WISDM_split_dataset/hot_encoding.pkl')
    return one_hot_encoding

def plot_activity(activity, df,window_size=400):

    data = df[df['Activity'] == activity][['X', 'Y', 'Z']][:window_size]
    axis = data.plot(subplots=True, figsize=(10, 10),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
def compareBetweenCTRNNModelsAndRNN(attribute,title,legend_loc='upper right',y_axis='Loss',x_axis='Epoch'):
    CtrnnTrainingTimeConstantHistoryPath='./MasterModels/CTRNN_Weights/CTRNN_Orignal_Model/history'
    CtrnnTimeConstant_equals_one_HistoryPath='./MasterModels/CTRNN_Weights/CTRNN_Constant_Tau_Equal_ones/history'
    CtrnnTimeConstant_random_InitilzationPath='./MasterModels/CTRNN_Weights/CTRNN_NO_TAU_TRAINING_RANDOM_INITIALIZATION/history'
    RNN_random_InitilzationPath='./MasterModels/RNN/rnn_model_2/history'
    ctrnnTrainingTimeConstantFile=open(CtrnnTrainingTimeConstantHistoryPath, "rb")
    CtrnnTimeConstant_equals_one_HistoryFile=open(CtrnnTimeConstant_equals_one_HistoryPath,'rb')
    CtrnnTimeConstant_random_InitilzationFile=open(CtrnnTimeConstant_random_InitilzationPath,'rb')
    RNN_random_InitilzationFile = open(RNN_random_InitilzationPath, 'rb')

    history1 = pickle.load(ctrnnTrainingTimeConstantFile)
    history2 = pickle.load(CtrnnTimeConstant_equals_one_HistoryFile)
    history3 = pickle.load(CtrnnTimeConstant_random_InitilzationFile)
    history4 = pickle.load(RNN_random_InitilzationFile)


    plt.rcParams['font.family'] = 'Times New Roman'

    # Plotting the validation loss
    plt.plot(np.array(history1[attribute]), label="CTRNN Training Time Constant")
    plt.plot(np.array(history3[attribute]), label="CTRNN Time Constant is Random")
    plt.plot(np.array(history2[attribute]), label="CTRNN Time Constant is One")
    plt.plot(np.array(history4[attribute]), label="RNN")


    # Customize the plot
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)

    # Close the opened files
    ctrnnTrainingTimeConstantFile.close()
    CtrnnTimeConstant_equals_one_HistoryFile.close()
    CtrnnTimeConstant_random_InitilzationFile.close()
    RNN_random_InitilzationFile.close()

    # Set the DPI to 300 for high-quality image
    plt.savefig('./MasterModels/CTRNNDifferentConfigurationAndRNN/'+title+'.png', dpi=300)  # Change the file format as needed
    plt.close()

def compareBetweenCTRNNModels(attribute,title,legend_loc='upper right',y_axis='Loss',x_axis='Epoch'):
    CtrnnTrainingTimeConstantHistoryPath='./MasterModels/CTRNN_Weights/CTRNN_Orignal_Model/history'
    CtrnnTimeConstant_equals_one_HistoryPath='./MasterModels/CTRNN_Weights/CTRNN_Constant_Tau_Equal_ones/history'
    CtrnnTimeConstant_random_InitilzationPath='./MasterModels/CTRNN_Weights/CTRNN_NO_TAU_TRAINING_RANDOM_INITIALIZATION/history'
    ctrnnTrainingTimeConstantFile=open(CtrnnTrainingTimeConstantHistoryPath, "rb")
    CtrnnTimeConstant_equals_one_HistoryFile=open(CtrnnTimeConstant_equals_one_HistoryPath,'rb')
    CtrnnTimeConstant_random_InitilzationFile=open(CtrnnTimeConstant_random_InitilzationPath,'rb')

    history1 = pickle.load(ctrnnTrainingTimeConstantFile)
    history2 = pickle.load(CtrnnTimeConstant_equals_one_HistoryFile)
    history3 = pickle.load(CtrnnTimeConstant_random_InitilzationFile)

    plt.rcParams['font.family'] = 'Times New Roman'

    # Plotting the validation loss
    plt.plot(np.array(history1[attribute]), label="Training Time Constant")
    plt.plot(np.array(history2[attribute]), label="Time Constant is One")
    plt.plot(np.array(history3[attribute]), label="Time Constant is Random")

    # Customize the plot
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)

    # Close the opened files
    ctrnnTrainingTimeConstantFile.close()
    CtrnnTimeConstant_equals_one_HistoryFile.close()
    CtrnnTimeConstant_random_InitilzationFile.close()

    # Set the DPI to 300 for high-quality image
    plt.savefig('./MasterModels/CTRNNDifferentConfiguration/'+title+'.png', dpi=300)  # Change the file format as needed
    plt.close()
def compareBetweenCTRNNMRNNodels(attribute,title,legend_loc='upper right',y_axis='Loss',x_axis='Epoch'):
    CtrnnTrainingTimeConstantHistoryPath='./MasterModels/CTRNN_Weights/CTRNN_Orignal_Model/history'
    RNNHistoryPath='./MasterModels/RNN/rnn_model_2/history'

    ctrnnTrainingTimeConstantFile=open(CtrnnTrainingTimeConstantHistoryPath, "rb")
    RNNHistoryPath_HistoryFile=open(RNNHistoryPath,'rb')

    history1 = pickle.load(ctrnnTrainingTimeConstantFile)
    history2 = pickle.load(RNNHistoryPath_HistoryFile)

    plt.rcParams['font.family'] = 'Times New Roman'

    # Plotting the validation loss
    plt.plot(np.array(history1[attribute]), label="CTRNN")
    plt.plot(np.array(history2[attribute]), label="RNN")

    # Customize the plot
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)

    # Close the opened files
    ctrnnTrainingTimeConstantFile.close()
    RNNHistoryPath_HistoryFile.close()

    # Set the DPI to 300 for high-quality image
    plt.savefig('./MasterModels/RNNVSCTRNN/'+title+'.png', dpi=300)  # Change the file format as needed
    plt.close()


def ComparereBetweenCTRNNModles():
    compareBetweenCTRNNModels(attribute='val_loss', title='Validation Loss')
    compareBetweenCTRNNModels(attribute='loss', title='Training Loss')
    compareBetweenCTRNNModels(attribute='accuracy', title="Training Accuracy", legend_loc='lower right',y_axis='Accuracy')
    compareBetweenCTRNNModels(attribute='val_accuracy', title="Validation Accuracy", legend_loc='lower right',y_axis='Accuracy')

def ComparereBetweenCTRNNModlesAndRNN():
    compareBetweenCTRNNModelsAndRNN(attribute='val_loss', title='Validation Loss')
    compareBetweenCTRNNModelsAndRNN(attribute='loss', title='Training Loss')
    compareBetweenCTRNNModelsAndRNN(attribute='accuracy', title="Training Accuracy", legend_loc='lower right',y_axis='Accuracy')
    compareBetweenCTRNNModelsAndRNN(attribute='val_accuracy', title="Validation Accuracy", legend_loc='lower right',y_axis='Accuracy')

def CompareBetweenCTRNNAndRNNModels():
    compareBetweenCTRNNMRNNodels(attribute='val_loss', title='Validation Loss')
    compareBetweenCTRNNMRNNodels(attribute='loss', title='Training Loss')
    compareBetweenCTRNNMRNNodels(attribute='val_accuracy', title='Validation Accuracy',legend_loc='lower right',y_axis='Accuracy')
    compareBetweenCTRNNMRNNodels(attribute='accuracy', title='Training Accuracy',legend_loc='lower right',y_axis='Accuracy')




if __name__=="__main__":
    #preprocessing()
    # df = pd.read_csv(r'D:\Master\Dataset\WISDM_ar_v1.1_raw.txt', header=None,
    #                  names=['ID', 'Activity', 'Timestamp', 'X', 'Y', 'Z'], on_bad_lines='skip')
    x_train, y_train= Load_training_data()
    x_test, y_test= load_testing_data()
    #onehotencoding=load_one_hot_encoding_data()

    #plot_activity("Downstairs", df[10000:])
    #plot_activity("Upstairs", df[10000:])
    #print( pd.get_dummies(y_test))
    #model=loadRNN(x_test,y_test,model_path   ='./MasterModels/RNN/rnn_model_32_10_6/big_rnn_model.keras')
    #trainRNN(x_train, y_train, x_test, y_test, number_of_epoches_=20)
    #trainBigRNN(x_train, y_train, x_test, y_test, number_of_epoches_=50)
    trainCTRNN(x_train, y_train, x_test, y_test, number_of_epoches_=50)
    #model=loadCTRNN(x_test,y_test,model_path='./MasterModels/CTRNN_Weights/CTRNN_Constant_Tau_Equal_ones/model.h5',firstLayer=16,secondLayer=10)
    #draw_conf_matrix(model,x_test,y_test,hot_encoding=onehotencoding)
    #drawHistory(file_path='./MasterModels/CTRNN_Weights/CTRNN_Orignal_Model/history')
    
    #ComparereBetweenCTRNNModles()
    #ComparereBetweenCTRNNModlesAndRNN()
    #CompareBetweenCTRNNAndRNNModels()
    #loadCTRNN(x_test,y_test,model_path='./MasterModels/CTRNN_Weights/CTRNN_Orignal_Model/model.h5')
    #loadCTRNN(x_test,y_test,model_path='./MasterModels/model.h5')

    #train_no_tau_training_CTRNN(x_train, y_train, x_test, y_test, number_of_epoches_=50,h=.01)

    # ctrnn_model = tf.keras.models.load_model('./MasterModels/model.h5', custom_objects={'SimpleCTRNN': SimpleCTRNN})
    # ctrnn_model.evaluate(x_train, y_train)
    # printWeights(ctrnn_model)

    #drawHistory('./MasterModels/CTRNN_model/1_ctrnn_history_h_01/1_ctrnn_history_h_0.01')

