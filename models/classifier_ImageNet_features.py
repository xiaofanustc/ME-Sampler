import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, column_units):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(2048, 2048, kernel_size=3, stride=2, dilation=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

    def forward(self, x):
        x = self.block1(x)
        # print(x.size())

        # averaging features in time dimension
        x = x.mean(-1)

        return x


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(1, 2048, 72))
    model = Model(2048).cuda()
    output = model(input_tensor.cuda())
    print("Final output shape = {}".format(output.size()))
