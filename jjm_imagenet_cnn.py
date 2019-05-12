import tensorflow as tf
import random
import numpy as np
from keras.datasets.cifar10 import load_data
def next_batch(num, data, labels):
  '''
  `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  '''
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

learning_rate=0.001
training_epochs=15
batch_size=100
keep_prob=tf.placeholder(tf.float32)

X=tf.placeholder(tf.float32,shape=[None,32,32,3])
Y=tf.placeholder(tf.float32,shape=[None,10])
(x_train,y_train),(x_test,y_test)=load_data()
y_train_one_hot=tf.squeeze(tf.one_hot(y_train,10),axis=1)
y_test_one_hot=tf.squeeze(tf.one_hot(y_test,10),axis=1)

W1=tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))
L1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1=tf.nn.dropout(L1,keep_prob=keep_prob)

W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2=tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L2=tf.nn.dropout(L2,keep_prob=keep_prob)

W3=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
L3=tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3=tf.nn.relu(L3)
L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L3=tf.nn.dropout(L3,keep_prob=keep_prob)
L3_flat=tf.reshape(L3,[-1,4*4*128])

W4=tf.get_variable("W4",shape=[128*4*4,625],initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([625]))
L4=tf.nn.relu(tf.matmul(L3_flat,W4)+b4)
L4=tf.nn.dropout(L4,keep_prob=keep_prob)

W5=tf.get_variable("W5",shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([10]))
hypothesis=tf.matmul(L4,W5)+b5
L5=tf.nn.relu(tf.matmul(L4,W5)+b5)
L5=tf.nn.dropout(L5,keep_prob=keep_prob)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Learning started. It takes sometime.')
# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    # 모든 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())

    # 10000 Step만큼 최적화를 수행합니다.
    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
            loss_print = cost.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})

            print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        sess.run(optimizer, feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.8})

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={X: test_batch[0], Y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)
