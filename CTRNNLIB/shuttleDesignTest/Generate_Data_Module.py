import math

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
def function_generator(training=True, noise_multiplier=.25, N=5, sample_delay=0.01, gen_samples=10,frequecny_factor=1,freq_gt=1):
  # Selection Parametrs:
  A= 0.5 * np.random.random(size=N)
  #delta=2*np.pi *(np.random.random(size=N)-0.5)#
  delta = 2 * np.pi * (np.random.random(size=N))
  #omega=1.33334+0.17776*np.random.random(size=N)
  omega=0.8+0.4 * np.random.random(size=N)
  #omega=0.008+.04*np.random.random(size=N)
  #omega=omega*frequecny_factor

  #Function to be called by the old solver the fucntion can be evaluated at any time ,t ,to mmic the behavior of the continous function
  def return_function(t):
    t_samples=np.array(np.arange((t-gen_samples*sample_delay),t,sample_delay))[:gen_samples]
    x_gt=np.sin(2*np.pi*freq_gt*t_samples)
    x=x_gt+noise_multiplier*(np.random.randn(gen_samples))
    #aditive interaction beween the N other signals:
    for i in range(N):
      x=x+A[i]*np.sin(freq_gt*(2*np.pi*t_samples-delta[i])/omega[i])+noise_multiplier*(np.random.randn(gen_samples))
    # if the model is being used for training or performance evaluation , then return the ground truth signal as well

    if training:
      return(x,x_gt)
    return x
    #return the singal env with the selected noise paramters
  return return_function




def generate_training_data(num_of_generator=16, signal_frequency=300, window_size=1, rows=10000, MEMS_freq=10**-6, enviroment_end_time=3392*10**-6,N_=5):#number of generator =number of batch-size
    generators_lst=[]
    enviroment_end_time=int(enviroment_end_time*10**6)
    for i in range(num_of_generator):
        generators_lst.append(function_generator(training=True, noise_multiplier=.25, N=N_, sample_delay=MEMS_freq,
                                                 gen_samples=enviroment_end_time, freq_gt=signal_frequency))
    noiesedList = []
    groundTruthList=[]
    generatorCounter = 0
    while(generatorCounter<num_of_generator):
        generatorID = generatorCounter % num_of_generator
        time = 0
        #X = np.random.normal(size=(n_batches, batch_size, seq_len, 1))
        X=[]
        Y=[]
        noise, gt_signal = generators_lst[generatorID](time * MEMS_freq)

        for i in range(enviroment_end_time // window_size):
            start_index=i*window_size
            X.append(noise[start_index:start_index+window_size])
            Y.append(gt_signal[start_index:start_index+window_size])
        generatorCounter += 1
        # plt.plot(np.array(X).reshape(-1))
        # plt.plot(np.array(Y).reshape(-1))
        # plt.show()
        noiesedList.append(np.array(X).reshape(len(X),window_size,1))
        groundTruthList.append(np.array(Y).reshape(len(Y),window_size,1))
    noiesedList=np.array(noiesedList)
    groundTruthList=np.array(groundTruthList)
    # plt.plot(groundTruthList.reshape(-1))
    # plt.plot(noiesedList.reshape(-1))
    return noiesedList,groundTruthList

import matplotlib.pyplot as plt
import pandas as pd

x_test,y_test=generate_training_data(window_size=1, num_of_generator=1,enviroment_end_time=30*3392*10**-6,N_=5)

excel_array=np.array([y_test.reshape(-1), x_test.reshape(-1)])
excel_array=np.transpose(excel_array)
df = pd.DataFrame(excel_array,columns=['Ground Truth', 'Noised Signal'])
#df.to_csv("./NewTrainingData/latest_data.csv", index=False)
#testing.to_csv("./NewTrainingData/testing_10window_size.csv", header=False, index=False)


# plt.plot(x_test.reshape(-1),'-')
#
# plt.plot(y_test.reshape(-1))

#plt.show()



