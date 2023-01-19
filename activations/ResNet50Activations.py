from activations.ActivationsModule import ActivationsModule
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


class ResNet50Activations(ActivationsModule):

    def __init__(self):
        self.name = "resnet50"
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).cuda()
        self.model.eval()
        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        self.model.avgpool.register_forward_hook(get_activation('avgpool'))
        return
        
    def get_activation_batch(self, img_batch):
        img_batch = F.interpolate(img_batch, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            self.model(img_batch)
            activation = self.activation['avgpool']
            activation = activation.squeeze(3).squeeze(2)

        return activation
    
    