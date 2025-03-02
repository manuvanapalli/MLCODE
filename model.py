import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 1)

        # Extract feature maps from all layers except final FC layer
        self.feature_map = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        # Extract feature map
        feature_map = self.feature_map(x)

        # Adaptive Average Pooling to reduce spatial dimensions
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        avg_pool_output = torch.flatten(avg_pool_output, 1)  # Flatten

        # Final prediction
        pred = self.model.fc(avg_pool_output)

        return pred, feature_map  # Ensure both values are returned

    @staticmethod
    def load_model(checkpoint_path):
        model = PneumoniaModel.load_from_checkpoint(checkpoint_path, strict=False)
        model.eval()
        return model
