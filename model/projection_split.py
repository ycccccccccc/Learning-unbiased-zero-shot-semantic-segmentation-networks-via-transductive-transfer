import torch.nn as nn
from model.util import get_embeddings, get_Ws_split
import torch


class ClassifierModule(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class Projection(nn.Module):

    def __init__(self, strong_or_weak, which_embedding,split):
        super(Projection, self).__init__()
        self.which_embedding = which_embedding
        self.strong_or_weak = strong_or_weak
        self.hidden = 300
        self.split = split
        if which_embedding == "all":
            self.hidden = 600
        self.projection = nn.Conv2d(1024, self.hidden, 1)
        self.strong_W, self.weak_W, self.all_W = self._get_W()

        nn.init.kaiming_normal_(self.projection.weight, a=1)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, x, which_W=None):
        if not which_W:
            which_W = self.strong_or_weak
        x = self.projection(x)
        x = x.permute([0, 2, 3, 1])
        assert which_W in ["strong", "weak", "all"]
        if which_W == "strong":
            x = torch.matmul(x, self.strong_W)
        elif which_W == "weak":
            x = torch.matmul(x, self.weak_W)
        elif which_W == "all":
            x = torch.matmul(x, self.all_W)
        x = x.permute([0, 3, 1, 2])
        return x

    def _get_W(self):
        embeddings = get_embeddings()
        Ws = get_Ws_split(embeddings, self.split)
        string = self.which_embedding + "_strong"
        strong = torch.tensor(Ws[string].T, dtype=torch.float).cuda()
        string = self.which_embedding + "_weak"
        weak = torch.tensor(Ws[string].T, dtype=torch.float).cuda()
        string = self.which_embedding + "_all"
        all = torch.tensor(Ws[string].T, dtype=torch.float).cuda()

        return strong, weak, all

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.projection.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{"params": self.get_10x_lr_params(), "lr": 10 * args.learning_rate}]
