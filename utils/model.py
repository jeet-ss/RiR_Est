import torch
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ):
        super().__init__()

        self.seq = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()

                )
        
    def forward(self, input_1D):
        return self.seq(input_1D)
    

class Geometry_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.convLayers = nn.Sequential(
                            Conv_block(1, 32, kernel_size=4, stride=4),
                            Conv_block(32, 32, kernel_size=2, stride=2),
                            Conv_block(32, 128, kernel_size=8, stride=8),
                            Conv_block(128, 128, kernel_size=2, stride=2),
                            Conv_block(128, 512, kernel_size=2, stride=2),
                            Conv_block(512, 512, kernel_size=4, stride=4),
                            Conv_block(512, 1024, kernel_size=4, stride=4),
                            Conv_block(1024, 1024, kernel_size=1, stride=1)
                        )
        self.fcLayers = nn.Sequential(
                            nn.Linear(1024, 160),
                            nn.Linear(160, 64),
                            nn.Linear(64, 3)
                        )
        
    def forward(self, inp):
        # conv layers
        batch = inp.size()[0]
        inp = self.convLayers(inp)
        # flatten
        inp = inp.reshape(batch, -1)
        # fc layers
        inp = self.fcLayers(inp)
        # return
        return inp
