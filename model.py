import torch
import torch.nn as nn
import torchvision.models as models


#Encoder Backbone (ResNet-18 with last layer removed)
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super(ResNetSimCLR, self).__init__()

        # Load pretrained ResNet and remove the final FC layer
        self.encoder = getattr(models, base_model)(pretrained=False)
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # Remove classification layer

        # Projection head: MLP with one hidden layer
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)               # Encoder output (representation)
        z = self.projection_head(h)       # Projected representation
        return h, z



class LinearClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()  # remove projection head if any
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # freeze encoder
        return self.classifier(features)
