import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
xy=np.loadtxt('zoo.csv',delimiter=',',dtype=np.float32)
y_data=xy[:,[-1]]
x_data=xy[:,0:-1]
X=tf.placeholder(tf.float32,shape=[None,16])
Y=tf.placeholder(tf.int32,shape=[None,1])
y_onehot=tf.one_hot(Y,7)
y_onehot=tf.reshape(y_onehot,[-1,7])
W=tf.Variable(tf.random_normal([16,7]))
b=tf.Variable(tf.random_normal([7]))
logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=tf.stop_gradient([y_onehot])))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
prediction=tf.argmax(hypothesis,1)
correct_prediction=tf.equal(prediction,tf.argmax(y_onehot,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100000):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step%1000==0:
            loss,acc=sess.run([cost,accuracy],feed_dict={X:x_data,Y:y_data})
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    pred=sess.run(prediction,feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

