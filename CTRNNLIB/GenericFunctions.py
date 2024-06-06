import numpy as np
import pandas as pd
def function_generator(training=True, noise_multiplier=.25, N=5, sample_delay=0.01, gen_samples=10,frequecny_factor=1,freq_gt=1):
  # Selection Parametrs:
  A= 0.5 * np.random.random(size=N)
  delta=2*np.pi *(np.random.random(size=N)-0.5)#
  #delta = 2 * np.pi * (np.random.random(size=N))
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
      x=x+A[i]*np.sin((2*np.pi*t_samples-delta[i])/omega[i])+noise_multiplier*(np.random.randn(gen_samples))
    # if the model is being used for training or performance evaluation , then return the ground truth signal as well

    if training:
      return(x,[x_gt[-1]])
    return x
    #return the singal env with the selected noise paramters
  return return_function


def generate_dataset(windows_size=10,batch_size=32):
    print('start generate dataset')
    x = []
    y = []
    generators=[]
    for i in range(batch_size):
        generators.append(function_generator(gen_samples=windows_size, sample_delay=0.01, noise_multiplier=.25))
    for i in range(0, int(50000/batch_size)):

        for gen in generators:
            samplej=gen(i/100)
            x.append(samplej[0][0:windows_size])
            y.append(samplej[1][0])



        # print(sample)
        # print(x)
        # print(y)

    df = pd.DataFrame(x)
    df[windows_size]=y

    #df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv(path_or_buf="../Data/denoising_signal_window_size_"+str(windows_size)+"_"+"batch_size_"+str(batch_size)+"_generator.csv", header=False, index=False)
    #df.plot('x', 'y', kind='scatter')
    print("end")
if __name__ == '__main__':
   print('hi main')

