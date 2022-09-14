import torch

class MLP_Imagenet(torch.nn.Module):

    def __init__(self,
                 out_size:int=0,
                 n_hid:int=128,
                ):
        super().__init__()
        self.n_hid = n_hid
        
        self.fc1 = torch.nn.Linear(3 * 224 * 224, n_hid)
        self.fc2 = torch.nn.Linear(n_hid, out_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.softmax(x)
        return x