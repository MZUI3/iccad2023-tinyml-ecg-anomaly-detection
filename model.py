import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, opts):
        super(model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(70,1), stride=(20,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(10, 1), stride=(1,1), padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(4, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.layer3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=212, out_features=80),
            nn.ReLU(True),
        )

        self.layer4 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(in_features=80, out_features=40),
            nn.ReLU(True),
        )

        self.layer5 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(in_features=40, out_features=2),
        )

    # def forward(self, input):
    #     print(input.shape)
    #     layer1_output = self.layer1(input)
    #     print(layer1_output.shape)
    #     layer2_output = self.layer2(layer1_output)
    #     print(layer2_output.shape)
    #     fc_output = layer2_output.view(-1,212)
    #     print(fc_output.shape)
        
    #     layer3_output = F.relu(self.layer3(fc_output))
    #     print(layer3_output.shape)
    #     layer4_output = self.layer4(layer3_output)
    #     print(layer4_output.shape)
    #     layer5_output = self.layer5(layer4_output)
    #     print(layer5_output.shape)

    #     return layer5_output
    
    def forward(self, input):
        layer1_output = self.layer1(input)
        layer2_output = self.layer2(layer1_output)
        fc_output = layer2_output.view(-1,212)
        
        layer3_output = self.layer3(fc_output)
        layer4_output = self.layer4(layer3_output)
        layer5_output = self.layer5(layer4_output)

        return layer5_output

import torch
temp_model = model(0)
temp = torch.rand(1,1,1250,1)
out = temp_model(temp)