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
        # inp shape = (b, c, 4096)
        # conv layers
        batch = inp.size()[0]
        inp = self.convLayers(inp)
        # flatten
        inp = inp.reshape(batch, -1)
        # fc layers
        inp = self.fcLayers(inp)
        # return
        return inp


class MLP_reflectionCoeff(nn.Module):
    """
        This model estimates 8 reflection coeffs!
        shouldnt there be 6 !?!
    """
    def __init__(self):
        super().__init__()
        # define layers
        self.mlp = nn.Sequential(
                        nn.Linear(2048, 1024), nn.ReLU(),
                        nn.Linear(1024, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 8), nn.ReLU(),
                    )
        
    def forward(self, inp):
        return self.seq(inp)
    

class Link_model(nn.Module):
    """
        This model links the room geometry estimates with the reflection coeffs
        Also, if this model produces the room geo with corresponding reflection coeffs, then whats the need of the mlp model !!!
    """
    
    def __init__(self):
        super().__init__()
        # define layers
        self.convLayers = nn.Sequential(
                            Conv_block(1, 32, kernel_size=3, stride=3),
                            Conv_block(32, 32, kernel_size=5, stride=5),
                            Conv_block(32, 128, kernel_size=3, stride=3),
                            Conv_block(128, 128, kernel_size=5, stride=5),
                            Conv_block(128, 512, kernel_size=4, stride=4),
                            Conv_block(512, 512, kernel_size=4, stride=4),
                            Conv_block(512, 1024, kernel_size=1, stride=1),
                            Conv_block(1024, 1024, kernel_size=1, stride=1)
                        )
        self.fcLayers = nn.Sequential(
                            # 1024 + 3 room geometry
                            nn.Linear(1027, 160),
                            nn.Linear(160, 64),
                            nn.Linear(64, 9)
                        )
        
    def forward(self, inp, room_geo):
        # inp shape = (b, 1, 4096) ,  room_geo = (b ,1, 3)
        batch = inp.size()[0]
        # add conditoning
        inp = torch.concatenate((inp, room_geo), axis = 2)
        # conv layers
        inp = self.convLayers(inp)
        # flatten
        inp = inp.reshape(batch, -1)
        # add conditioning
        # 1024 + 3
        inp = torch.cat((inp, room_geo.squueze(axis=1)), axis=1)
        # fc layers
        inp = self.fcLayers(inp)
        # return
        return inp
    
    