import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.pooling=nn.MaxPool2d(2,2)
        self.flatten=nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)
        self.output=nn.Linear(128,3)
