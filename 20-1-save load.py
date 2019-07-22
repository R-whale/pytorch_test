import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import os

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #输入维度、输出维度、卷积核大小
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        #不确定reshape成几行就用-1，列是确定的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TheModelClass()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Model's state_dict:")
#state_dict相当于一个参数字典
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

PATH = "/model_d"
torch.save(model.state_dict(), PATH)

model = TheModelClass()
model.load_state_dict(torch.load(PATH))
# model.load_state_dict(参数)，参数必须是state_dict的对象
model.eval()

torch.save(model, os.path.join(PATH))

model = torch.load(PATH)
model.eval()