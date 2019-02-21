import tensorflow as tf
a=tf.placeholder(tf.int32,[3])
b=tf.constant(2)
op=a*b
sess=tf.Session()
r1=sess.run(op,feed_dict={a:[1,2,3]})
print(r1)
r2=sess.run(op,feed_dict={a:[10,20,30]})
print(r2)