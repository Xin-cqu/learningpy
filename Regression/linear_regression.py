__author__ = 'xinwen'
import tensorflow as tf
import numpy as np
import scipy.io as sio

data=np.loadtxt('ex1data1.txt',delimiter=',').astype('float')
need_predict=tf.Variable([35000.0,70000.0])

x_input=data[:,0]
y_input=data[:,1]
X=tf.placeholder(dtype='float')
Y=tf.placeholder(dtype='float')
w=tf.Variable(tf.zeros([2,1]),name='weight')
b=tf.Variable(tf.zeros([2,1]),name='bias')
y_=tf.add(b,tf.mul(X,w))
cost=tf.pow(Y-y_,2)
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
for i in range(30):
    for (x,y) in zip(x_input,y_input):
        sess.run(train_op,feed_dict={X:x,Y:y})
print need_predict
print sess.run(tf.add(b,tf.mul(need_predict,w)))
