from activations.ActivationsModule import ActivationsModule
from pytorch_fid.inception import InceptionV3
import torch

class InceptionActivations(ActivationsModule):
    
    def __init__(self, recompute=False):        
        self.name = "inception"
        self.activations_size = 2048
        self.transform = None
        
        super().__init__(recompute=recompute)
        
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx], normalize_input=False).cuda()
        self.model.eval()
        return
        
    def get_activation_batch(self, img_batch):
        with torch.no_grad():
            activation = self.model(img_batch)[0]

        activation = activation.squeeze(3).squeeze(2)
        return activation
    
    