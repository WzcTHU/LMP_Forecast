import torch
import torch.utils.data as Data
import torchvision
import numpy as np
from sklearn.preprocessing import StandardScaler
from func import *
import torch.nn as nn

scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test = divide_data('x.csv', 'y.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
# x_test = scaler1.transform(x_test)
# y_test = scaler2.transform(y_test)

x_train_tensor = torch.from_numpy(np.array(x_train)).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
# x_test_tensor = torch.from_numpy(np.array(x_test)).float()
# y_test_tensor = torch.from_numpy(np.array(y_test)).float()

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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(44, 37),
            nn.ReLU(),
            nn.Linear(37, 31),
            nn.ReLU(),
            nn.Linear(31, 26),
            nn.ReLU(),
            nn.Linear(26, 22),
        )

        self.decoder = nn.Sequential(
            nn.Linear(22, 26),
            nn.ReLU(),          
            nn.Linear(26, 31),
            nn.ReLU(),
            nn.Linear(31, 37),
            nn.ReLU(),
            nn.Linear(37, 44),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

dataset = TDataset(data_tensor=x_train_tensor, 
    target_tensor=y_train_tensor)

data_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

autoencoder = AutoEncoder()

LR = 0.005
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(500):
    if epoch % 10 == 0:
        LR = LR * 0.98
    for step, (x, b_label) in enumerate(data_loader):
        b_x = x.view(-1, 44)
        b_y = x.view(-1, 44)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('Epoch:', epoch, '|train loss: %4f' % loss.data.numpy())

torch.save(autoencoder.encoder, 'net.pkl')  # 保存整个encoder网络