import numpy as np
import tensorflow as tf
#print(tf.__version__)
#print(tf.test.is_gpu_available())

a=tf.constant(10)
print(a)   #o/p= tf.Tensor(10, shape=(), dtype=int32)
b=tf.constant(10.2)
print(b)    #tf.Tensor(10.2, shape=(), dtype=float32)
c=tf.constant('Amiya')
print(c)    #tf.Tensor(b'Amiya', shape=(), dtype=string)

np_arr=tf.constant(np.array([1,2,3,4]))
print(np_arr)  #tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)

t_2d=tf.constant([1,2,3,4],shape=(2,2),dtype="int32")
print(t_2d) #2d tensor

t_3d=tf.constant([[[1,2],[2,3],[3,4]]],dtype="int32")
print(t_3d) #3d tensor
print(type(t_3d))
print(t_3d.shape) #Will get just the shape
print(t_3d.numpy()) #Get accurate array 
print(t_3d.numpy()[0][2])
print(t_3d.dtype)

