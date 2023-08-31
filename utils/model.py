import torch
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ):
        super().__init__()

        self.seq = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU()

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


model_f = nn.Sequential(
                        nn.Linear(4096, 2048), nn.ReLU(),
                        nn.Linear(2048, 1024), nn.ReLU(),   #1 
                        nn.Linear(1024, 512), nn.ReLU(),    #2
                        nn.Linear(512, 256), nn.ReLU(),     #3
                        nn.Linear(256, 128), nn.ReLU(),     #4
                        nn.Linear(128, 64), nn.ReLU(),      #5
                        nn.Linear(64, 32), nn.ReLU(),       #6
                        nn.Linear(32, 16), nn.ReLU(),       #7
                        nn.Linear(16, 8), nn.ReLU(),        #8
                        nn.Linear(8, 6)                     #9
                    )

class MLP_reflectionCoeff(nn.Module):
    """
        This model estimates 6 reflection coeffs!
    """
    def __init__(self):
        super().__init__()
        # define layers
        self.mlp = nn.Sequential(
                        nn.Linear(4096, 2048), nn.ReLU(),
                        nn.Linear(2048, 1024), nn.ReLU(),   #1 
                        nn.Linear(1024, 512), nn.ReLU(),    #2
                        nn.Linear(512, 256), nn.ReLU(),     #3
                        nn.Linear(256, 128), nn.ReLU(),     #4
                        nn.Linear(128, 64), nn.ReLU(),      #5
                        nn.Linear(64, 32), nn.ReLU(),       #6
                        nn.Linear(32, 16), nn.ReLU(),       #7
                        nn.Linear(16, 8), nn.ReLU(),        #8
                        nn.Linear(8, 6)                     #9
                    )
        
    def forward(self, inp):
        return self.mlp(inp)
    

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
        # inp shape = (b, 4096) ,  room_geo = (b , 3)
        batch = inp.size()[0]
        # add conditoning

        inp = torch.concatenate((inp, room_geo), axis = 1).unsqueeze(axis=1)
        # conv layers
        inp = self.convLayers(inp)
        # flatten
        inp = inp.reshape(batch, -1)
        # add conditioning
        # 1024 + 3
        inp = torch.cat((inp, room_geo), axis=1)
        # fc layers
        inp = self.fcLayers(inp)
        # return
        return inp
    
    