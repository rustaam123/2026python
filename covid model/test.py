import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
import time


class Covid_dataset(Dataset):
    def __init__(self, file_path, mode):  # mode说明数据集是什么类型 训练集还是测试集
        with open(file_path, "r") as f:
            csv_data = list(csv.reader(f))
            data = np.array(csv_data[1:])
            if mode == "train":
                indices = [i for i in range(len(data)) if i % 5 != 0]
            elif mode == "val":
                indices = [i for i in range(len(data)) if i % 5 == 0]

            if mode == "test":
                x = data[:, 1:].astype(float)
                x = torch.tensor(x)
            else:
                x = data[indices, 1:-1].astype(float)
                x = torch.tensor(x)
                y = data[indices, -1].astype(float)
                self.y = torch.tensor(y)

            self.x = x - x.mean(dim=0, keepdim=True) / x.std(dim=0, keepdim=True)

            self.mode = mode

    def __getitem__(self, item):# 返回数据
        if self.mode == "test":
            return self.x[item].float()  # 测试集没标签。   注意data要转为模型需要的float32型
        else:  # 否则要返回带标签数据
            return self.x[item].float(), self.y[item].float()

    def __len__(self): # 返回数据集长度
        return len(self.x)


class myModel(nn.Module):
    def __init__(self, dim):  #dim表示维度
        # 初始化父类
        super(myModel, self).__init__()
        # 第一层全连接层：输入维度dim，输出维度100
        self.fc1 = nn.Linear(dim, 100)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二层全连接层：输入维度100，输出维度1
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze(1)
        else:
            return x


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):
    model = model.to(device)  # 将模型移动到指定设备

    plt_train_loss = []  # 记录训练损失
    plt_val_loss = []    # 记录验证损失
    min_val_loss = 9999999999  # 初始化最小验证损失

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()
# model调用mymodel类
        model.train()  # 设置模型为训练模式
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            pred = model(x)
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()
            optimizer.step()  # 更新参数 之后要梯度清零否则会累积梯度
            optimizer.zero_grad()
            train_loss += train_bat_loss.cpu().item()

        plt_train_loss.append(train_loss / train_loader.dataset.__len__())

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)
                val_loss += val_bat_loss.cpu().item()
        plt_val_loss.append(val_loss / val_loader.dataset.__len__())
        if val_loss < min_val_loss:  # 保存最佳模型
            torch.save(model, save_path)
            min_val_loss = val_loss

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f' % \
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1])
              )  # 打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()


def evaluate(save_path, device, test_loader, rel_path):
    model = torch.load(save_path).to(device)
    rel = []

    with torch.no_grad():
        for x in test_loader:
            pred = model(x.to(device))
            rel.append(pred.cpu().item())

    print(rel)

    with open(rel_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "test_positive"])
        for i in range(len(rel)):
            csv_writer.writerow([str(i), str(rel[i])])
        print("文件已经保存到" + rel_path)


device = "cuda" if torch.cuda.is_available() else "cpu"

train_file = "covid.train.csv"
test_file = "covid.test.csv"

train_data = Covid_dataset(train_file, "train")
val_data = Covid_dataset(train_file, "val")
test_data = Covid_dataset(train_file, "test")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)  # 测试集的batchsize一般为1 且不可以打乱

dim = 93

config = {
    "lr": 0.001,
    "momentum": 0.9,
    "epochs": 20,
    "save_path": "model_save/model.pth",
    "rel_path": "pred.csv"
}

model = myModel(dim)

loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])  # 优化器

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])

evaluate(config["save_path"], device, test_loader, config["rel_path"])