import tensorflow as tf
import numpy as np
def test():
    matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
    matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')

    with tf.device('/CPU:0'):
        tfm1 = tf.constant(matrix1)
        tfm2 = tf.constant(matrix2)
        mat_product = tf.matmul(tfm1, tfm2)
        mat_add = tf.add(tfm1, tfm2)
        matrix_3 = np.array([(2, 7, 2), (1, 4, 2), (9, 0, 2)], dtype='float32')
        print(matrix_3)
        matrix_det = tf.linalg.det(matrix_3)
        print(mat_product)
        print(mat_add)
        print(matrix_det)


if __name__ == '__main__':
    test()
