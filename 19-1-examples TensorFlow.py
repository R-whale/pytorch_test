import tensorflow as tf
import numpy as np
import time

#使用TensorFlow定义神经网络
start = time.clock()

N, D_in, H, D_out = 64, 1000, 100, 10

x = tf.placeholder(tf.float32, shape=(None, D_in))  # 为x\y占位
y = tf.placeholder(tf.float32, shape=(None, D_out))
#定义过程，执行时赋值

w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))
#Variable()是一个变量构造函数

h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

loss = tf.reduce_sum((y - y_pred)**2.0)

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
#实现loss函数对w1，w2的求导

learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    for _ in range(500):
        k = _

        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        #可以提供feed作为run的调用参数，只在调用它的方法内有效, 方法结束, feed就会消失。
        #标记的方法是使用tf.placeholder()为这些操作创建占位符。
        print(k, loss_value)

end = time.clock()
print("time: {}".format(end - start))


