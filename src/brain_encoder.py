import torch
import yaml

from omegaconf import OmegaConf
from bm.models.simpleconv import SimpleConv

def build_brain_encoder(config):
    checkpoint = torch.load(config.encoder.checkpoint_path, map_location=torch.device('cpu'))
    cfg = checkpoint['xp.cfg']
    yaml_str = OmegaConf.to_yaml(cfg)

    with open(config.encoder.meta_config_path, 'w') as f:
        f.write(yaml_str)

    with open(config.encoder.clip_conv_config_path, 'r') as file:
        data = yaml.safe_load(file)['simpleconv']
        
    data['in_channels'] = {'meg': config.datasets.target_channels}
    data['out_channels'] = config.encoder.out_channels
    weights = checkpoint['best_state']
    model = SimpleConv(**data)

    for name, param in model.named_parameters():
        if '0.' + name in weights:
            param.data = weights['0.' + name]
        else:
            print(f"No weights found for {name}")
    return model