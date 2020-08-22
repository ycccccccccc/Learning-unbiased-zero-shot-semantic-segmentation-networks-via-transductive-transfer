import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg_Deeplab(nn.Module):
    def __init__(self,*args, **kwargs):
        super(Vgg_Deeplab, self).__init__()
        vgg16 = torchvision.models.vgg16()

        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=1, padding=1))

        layers.append(nn.Conv2d(512,
                                512,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512,
                                512,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512,
                                512,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=1, padding=1))
        self.features = nn.Sequential(*layers)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride=1, padding=1))
        classifier.append(nn.Conv2d(512,
                                    1024,
                                    kernel_size=3,
                                    stride=1,
                                    padding=12,
                                    dilation=12))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        self.classifier = nn.Sequential(*classifier)

        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_vgg = vgg.features.state_dict()
        self.features.load_state_dict(state_vgg)

        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        # b = []
        #
        # b.append(self.conv1)
        # b.append(self.bn1)
        # b.append(self.layer1)
        # b.append(self.layer2)
        # b.append(self.layer3)
        # b.append(self.layer4)

        for i in self.features:
            #for j in self.features[i].modules():
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
        # b = []
        # b.append(self.layer.parameters())

        for i in self.classifier:
            #for j in self.classifier[i].modules():
            jj = 0
            for k in i.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

    def optim_parameters_10x(self, args):
        return [{"params": self.get_10x_lr_params(), "lr": 10 * args.learning_rate}]


if __name__ == "__main__":
    net = Vgg_Deeplab(3, 10)
    in_ten = torch.randn(1, 3, 224, 224)
    out = net(in_ten)
    print(out.size())

    in_ten = torch.randn(1, 3, 64, 64)
    mod = nn.Conv2d(3,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=2)
    out = mod(in_ten)
    print(out.shape)
