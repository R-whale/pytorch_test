import torch

#使用torch实现手动传递梯度值的神经网络
dtype = torch.float
device = torch.device('cpu')

#处理输入输出
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, dtype=dtype, device=device)
y = torch.randn(N, D_out, dtype=dtype, device=device)

#定义权重
w1 = torch.randn(D_in, H, dtype=dtype, device=device)
w2 = torch.randn(H, D_out, dtype=dtype, device=device)

#学习效率
learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    #损失函数
    loss = (y_pred - y).pow(2).sum().item()
    print(t,loss)

    #计算梯度，并传递
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    #更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2