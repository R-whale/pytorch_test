import numpy as np

#使用numpy实现简单的网络，手动实现梯度传递
N, D_in, H, D_out = 64, 1000, 100, 10
x = np.random.randn(N, D_in)#64x1000
y = np.random.randn(N, D_out)#64x10 正态分布

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)#x与y按位比较取大
    y_pred = h_relu.dot(w2)

    #计算损失函数
    loss = np.square(y_pred - y).sum()#损失函数
    #print(t, loss)

    #计算梯度，将梯度返回到网络参数中
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    #更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



