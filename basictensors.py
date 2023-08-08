import tensorflow as tf
#print(tf.version)

rank1_tensor = tf.Variable([["Anmol", "Jaz"],["Anu","Jasrehmat"]] , tf.string)
tr=tf.rank(rank1_tensor)
#print(tr)

ts=tf.shape(rank1_tensor)
# print(ts)

tensor1 = tf.ones([1,2,3])
#print(tensor1)

tensor2 = tf.reshape(tensor1,[2,3,1])
#print(tensor2)

tensor3 = tf.reshape(tensor1,[3,-1])    #-1 tells the tensor to calculate the size of the dimension in that place 
#print(tensor3)                          #-1 reshapes the tensor to [3,2] here

with tf.Session() as sess:
    tensor2.eval()