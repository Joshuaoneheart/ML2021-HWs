import torch
import torch.nn as nn

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 11,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.layer_1 = nn.Linear(512 * 7 * 7, 4096),
        self.act_fn = nn.ELU(0.1),
        self.dropout_cnn = nn.Dropout(0.8),
        self.layer_2 = nn.Linear(4096, 4096),
        self.dropout = nn.Dropout(0.5),
        self.layer_3 = nn.Linear(4096, 4096),
        self.output_layer = nn.Linear(4096, num_classes),
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_1 = self.layer_1(x)
        x = torch.cat([x, x_1], dim = 1)
        x_1 = self.layer_2(x)
        x = torch.cat([x, x_1], dim = 1)
        x_1 = self.layer_3(x)
        x = torch.cat([x, x_1], dim = 1)
        out = self.output_layer(x)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ELU(0.1)]
            else:
                layers += [conv2d, nn.ELU(0.1)]
            in_channels = v
    return nn.Sequential(*layers)
