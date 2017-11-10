# simple_model.py
#
import torch
from torch.autograd import Variable

# Add class for fully connected, 3-layer model
class FC(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 lr=0.0001,
                 num_dnodes=128):
        super(FC, self).__init__()
        print ("Ins/outs: ", n_inputs, n_outputs)
        self.bl1 = torch.nn.Linear(n_inputs, num_dnodes)
        self.bl2 = torch.nn.ReLU()
        self.bl3 = torch.nn.Linear(num_dnodes, n_outputs)
        self.bl4 = torch.nn.Sigmoid()
        self.bls = torch.nn.ModuleList([self.bl1, self.bl2, self.bl3, self.bl4])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def loss(self, inputs, targets):
        # use Mean Squared Error (MSE) as loss function
        loss_fn = torch.nn.MSELoss(size_average=False)
        prediction = self(inputs)
        # print ("prediction: ", prediction)
        # input("")
        # Calculate loss for one example
        loss = loss_fn(prediction, targets)
        # Return final running average for batch
        return loss

    def train(self, inputs, targets, use_cuda=False):
        self.opt.zero_grad()
        # print ("inputs: ", inputs.shape)
        # print ("targets: ", targets.shape)
        L = self.loss(Variable(torch.from_numpy(inputs).float()), Variable(torch.from_numpy(targets).float()))
        L.backward()
        self.opt.step()
        return L.data.cpu()[0]

    def forward(self, inputs):
        x = inputs
        for bl in self.bls:
            # print (x)
            x = bl(x)
        # x = self.bl1(inputs)
        # x = self.bl2(x)
        # x = self.bl3(x)
        return x
