import numpy as np
import tensorflow as tf

#NOTE : Placeholder is not accessable in tensorflow 2.X so we need to access it from version 1

#Create placeholder
tf.compat.v1.disable_eager_execution()  #Mendatory line to exicute for placeholder
p=tf.compat.v1.placeholder(dtype=tf.float32,shape=(400,400))  #Compact v1 means compact from version 1.X
p2=tf.compat.v1.placeholder(dtype=tf.float32,shape=(400,400))  #Compact v1 means compact from version 1.X

print(p)

#Perform mathmatical operation
p3=tf.add(p,p2)
print(p3)   #o/p= Tensor("Add:0", shape=(400, 400), dtype=float32) #Can't add


#but we can do it with creating numpy array
a=np.ones(shape=(400,400),dtype=np.int32)
print(a)

with tf.compat.v1.Session() as sess:
    d=sess.run(p3,feed_dict={p:a,p2:a})
print(d)
