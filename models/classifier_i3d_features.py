import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, column_units):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1024, 2048, kernel_size=(3, 3, 3), stride=(2, 2, 2)),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

    def forward(self, x):
        x = self.block1(x)

        # averaging features in time dimension
        x = x.mean(-1).mean(-1).mean(-1)

        return x


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(1, 1024, 9, 7, 7))
    model = Model(2048).cuda()
    output = model(input_tensor.cuda())
    print(output.size())
