import torch
import torch.utils.data as Data
import torchvision
import numpy as np
from sklearn.preprocessing import StandardScaler
from func import *
import torch.nn as nn

scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test = divide_data('x_rr.csv', 'y.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

x_train_tensor = torch.from_numpy(np.array(x_train)).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
x_test_tensor = torch.from_numpy(np.array(x_test)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()

class TDataset(Data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=3),
            # nn.BatchNorm1d(num_features=30),
            nn.ReLU(inplace=True)
        )

        self.maxpool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=15, out_channels=25, kernel_size=3),
            # nn.BatchNorm1d(num_features=50),
            nn.ReLU(inplace=True)
        )

        self.maxpool2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fullconnect = nn.Sequential(
            nn.Linear(225, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fullconnect(x)
        return x

dataset = TDataset(data_tensor=x_train_tensor, 
    target_tensor=y_train_tensor)

BS = 64
data_loader = Data.DataLoader(dataset=dataset, batch_size=BS, shuffle=True, drop_last=True)

CNN = CNN()

LR = 0.001
optimizer = torch.optim.Adam(CNN.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(500):
    if epoch % 10 == 0:
        LR = LR * 1
    for step, (x, b_label) in enumerate(data_loader):
        b_x = x.view(BS, -1, 44)
        b_y = b_label

        cnn_x = CNN(b_x)

        loss = loss_func(cnn_x, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('Epoch:', epoch, '|train loss: %4f' % loss.data.numpy())

torch.save(CNN.state_dict(), 'CNN_NET.pkl')

CNN_net = torch.load('CNN_NET.pkl') # 提取训练好的encoder
CNN_net = CNN()
CNN_net.load_state_dict(torch.load('CNN_NET.pkl'))
CNN_net.eval()
y_fore_test = []
y_fore_train = []
# for each in x_test_tensor:
#     x_train_encoded.append(encoder_net(torch.from_numpy(np.array(each)).float()).detach().numpy())
for each in x_test_tensor:
    each = each.view(1, -1, 44)
    y_fore_test.append(CNN_net(torch.from_numpy(np.array(each)).float()).detach().numpy())

for each in x_train_tensor:
    each = each.view(1, -1, 44)
    y_fore_train.append(CNN_net(torch.from_numpy(np.array(each)).float()).detach().numpy())
y_test = scaler2.inverse_transform(y_test)
y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = scaler2.inverse_transform(y_fore_test)

y_fore_test2 = []
y_fore_train2 = []
for each in y_fore_test:
    y_fore_test2.append(each[0][0])
for each in y_fore_train:
    y_fore_train2.append(each[0][0])
get_results_cnn(y_train, y_fore_train, y_test, y_fore_test2)
