import torch

class MLP_Imagenet(torch.nn.Module):

    def __init__(self,
                 out_size:int=0,
                 n_hid:int=128,
                 device:str="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.n_hid = n_hid
        self.fc1 = torch.nn.Linear(3 * 224 * 224, n_hid).to(device)
        self.fc2 = torch.nn.Linear(n_hid, out_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.softmax = torch.nn.Softmax(dim=1).to(device)

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)
        x = torch.tanh(self.fc1(x))
        x = self.relu(x)
        x = torch.tanh(self.fc2(x))
        x = self.softmax(x)
        return x
