import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print((41.8 * (10 ** -6)))
def f(x):
    return tf.keras.activations.sigmoid(x - 41.8e-6)
    #return x**2*np.exp(-x**2)
def sigmiod_sharper(x,factor=1):
    return tf.keras.activations.sigmoid(factor*(x - 41.8e-6))
def hardSig(x):
    return tf.keras.activations.hard_sigmoid(x - 41.8e-6)
def activationZ(x):
    epsMulA=8.85400e-19
    d=5e-6
    return epsMulA/(d-x)

def unitstep(x):
    d=tf.constant(
        1, dtype=float, shape=None, name='Const'
    )
    return tf.experimental.numpy.heaviside(x- 41.8e-6, 41.8e-6)
    #return np.heaviside(x, 1)
x = np.linspace ( start = 0    # lower limit
                , stop =   .1      # upper limit
                , num = 10000      # generate 51 points between 0 and 3
                )
y = f(x)    # This is already vectorized, that is, y will be a vector!
y1=sigmiod_sharper(x,10)
y2=sigmiod_sharper(x,4)
y3=unitstep(x)
y4=activationZ(x)
plt.plot(x, y,color='green')
plt.plot(x,y1,color='red')
plt.plot(x,y2,color='orange')

plt.plot(x,y3,color='yellow')
plt.plot(x,y4,color='black')





print("start")
print(1.463201e-11)
print(f(1.463201e-11))
print('unit')
print(unitstep(1.463201e-11))
print(sigmiod_sharper(1.463201e-11,5000000))
plt.show()
