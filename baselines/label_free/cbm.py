import os
import torch
from . import data_utils

class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, model_dir, device="cuda"):
        super().__init__()
        model, _ = data_utils.get_target_model(device=device, model_dir=model_dir)
        self.backbone = model
            
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c
    
def load_cbm(load_dir, model_dir, device):
    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model("dinov2", W_c, W_g, b_g, proj_mean, proj_std, model_dir, device)
    return model
