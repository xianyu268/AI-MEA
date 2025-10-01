from datetime import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# 
class Classify_Model(nn.Module):
    def __init__(self, input_dim, output_size):
        super(Classify_Model, self).__init__()
        # 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #
        self.Layer1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    # nn.Dropout(p=0.2)
                                    )
        self.res_Layer1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=10, kernel_size=2, stride=2),
                                    )


        #
        self.Layer2 = nn.Sequential(nn.Conv1d(in_channels=10, out_channels=100, kernel_size=7, padding=3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    # nn.Dropout(p=0.2)
                                    )
        self.res_Layer2 = nn.Sequential(nn.Conv1d(in_channels=10, out_channels=100, kernel_size=2, stride=2),
                                        )

        self.Layer3 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=200, kernel_size=7, padding=3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2),
                                    # nn.Dropout(p=0.2)
                                    )
        self.res_Layer3 = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=200, kernel_size=2, stride=2),
                                        )

        #
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        #
        self.Layer_classify = nn.Sequential(nn.Linear(200, output_size),
                                            nn.Softmax(dim=1),
                                            )

    def forward(self, x):

        out = self.Layer1(x) + self.res_Layer1(x)
        out = self.Layer2(out) + self.res_Layer2(out)
        out = self.Layer3(out) + self.res_Layer3(out)
        out = self.avg_pool(out)
        outputs = self.Layer_classify(out.reshape(out.size()[0],-1))
        # outputs = self.Layer_classify(x)

        return outputs


