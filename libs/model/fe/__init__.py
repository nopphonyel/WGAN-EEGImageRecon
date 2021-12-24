from preimport_module import *


class AlexNetExtractor(nn.Module):
    """
    This class expected image as input with size (64x64x3)
    """

    def __init__(self, output_class_num, in_channel=3, feature_size=200, pretrain=False):
        super(AlexNetExtractor, self).__init__()
        self.feature_size = feature_size
        self.num_classes = output_class_num
        if not pretrain:
            self.in_channel = in_channel
        else:
            print("<I> Pre-trained has been set, using in_channel=3")
            self.in_channel = 3
        self.features = nn.Sequential(
            # Alex1
            nn.Conv2d(self.in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img
        if (pretrain):
            import torchvision
            ori_alex = torchvision.models.alexnet(pretrained=True)
            ori_weight = ori_alex.state_dict()
            ori_weight.pop('classifier.1.weight')
            ori_weight.pop('classifier.1.bias')
            ori_weight.pop('classifier.4.weight')
            ori_weight.pop('classifier.4.bias')
            ori_weight.pop('classifier.6.weight')
            ori_weight.pop('classifier.6.bias')
            self.load_state_dict(ori_weight)
            del (ori_alex)
            del (ori_weight)

        self._add_classifier(self.num_classes, self.feature_size)

    def _add_classifier(self, num_classes, feature_size):
        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )
        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )
        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label
