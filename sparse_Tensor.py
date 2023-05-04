import numpy as np
import tensorflow as tf

# Note : Sparse Tensor = tensor containing 1d,2d,3d... array which contains maximum zero valuesm zero values.
#syntax : tf.sparse.SparseTensor(indices,values,dence_space)

sp=tf.SparseTensor(indices=[[0,3],[5,4]],values=[10,20],dense_shape=[3,10])
print(sp)


#Dense Tensor = tensor containing array which contains maximum non zero values.
arr=np.array([[1,0,0,0],[1,0,0,0],[0,0,0,0],[1,0,0,0]])
print(arr)

#Now array to sparse tensor
sp2=tf.sparse.from_dense(arr)
print(sp2)
print(sp2.values.numpy().tolist()) #To get actual values
print(sp2.indices.numpy().tolist()) #To get actual value as input given
print(sp2.dense_shape.numpy().tolist())  #To get the shape of the dense tensor

#Sparse tensor to Dense tensor
d2=tf.sparse.to_dense(sp2)
print(d2)

#Dense tensor to numpy
print(d2.numpy())

#Mathematical operation with sparse tensor

m=tf.sparse.add(sp,sp)
print(m)
print(tf.sparse.to_dense(sp))