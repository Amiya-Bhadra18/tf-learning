import numpy as np
import tensorflow as tf

#V should be capital of Variable
v=tf.Variable(1)
print(v)  #<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>

v1=tf.Variable([1,2,3,4]) #Variable data can be float,bool,complex or string as well
print(v1)  #<tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([1, 2, 3, 4])>
print(v1.name)
print(v1.dtype)
print(v1.numpy())

#using constant create variable
t_con=tf.constant([1,2,3,4])
print(t_con)
v5=tf.Variable(t_con)
print(v5)

#TF variable of diff shape
t_2d=tf.Variable([[1,2],[3,4]])
print(t_2d)

#Get index of highest value
print(tf.argmax(t_2d).numpy())

#Convert anything to tensor
t10=tf.convert_to_tensor(t_2d)
print(t10)

#Change and asign a new value to tensor
t_2d.assign([[45,65],[78,28]])
print(t_2d)  #Changes occer in the same memory location

print(t_2d.assign_add([[56,52],[96,15]]))  #Changes occur in a new memory location
