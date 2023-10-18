import torch
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=7)
        self.batch1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=9,bias = False)
        self.batch2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(980, 256)
        self.batch3 = torch.nn.BatchNorm1d(256)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256,10 )


    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = self.batch1(x)
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
        x = self.batch2(x)
        x = x.view(-1, 980)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
