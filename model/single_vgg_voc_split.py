import torch.nn as nn
from model.vgg_deeplab import Vgg_Deeplab as Deeplab
from model.projection_split import Projection

import torch

STRONG_CLASSES = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog",
                  "horse", "motorbike", "person"]

class Our_Model(nn.Module):
    def __init__(self,split):
        super(Our_Model,self).__init__()
        self.vgg = Deeplab()
        self.projection = Projection("all","all",split)

    def forward(self, x, which_W=None,which_branch = None):
        x = self.vgg(x)
        assert which_W in ["strong", "weak", "all", None]
        return self.projection(x,which_W)



    def get_1x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        for i in self.vgg.features:
            jj = 0
            for k in i.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

    def optim_parameters_1x(self, args):
        return [{"params": self.get_1x_lr_params(), "lr": 1 * args.learning_rate}]

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.projection.projection.parameters())
        b.append(self.vgg.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_10x(self, args):
        return [{"params": self.get_10x_lr_params(), "lr": 10 * args.learning_rate}]