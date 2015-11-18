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
w=tf.Variable(0.0,name='weight')
b=tf.Variable(0.0,name='bias')
def model(x,w):
    return tf.mul(x,w)
y_=model(X,w)
cost=tf.pow(Y-y_,2)
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess=tf.Session()
init=tf.initialize_all_variables()

sess.run(init)
for i in range(150):
    for (x,y) in zip(x_input,y_input):
        sess.run(train_op,feed_dict={X:x,Y:y})
print sess.run(model(need_predict,w))
