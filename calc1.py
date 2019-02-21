import tensorflow as tf
a=tf.constant(1234)
b=tf.constant(5000)
operation=a+b
sess=tf.Session()
res=sess.run(operation)
print(res)