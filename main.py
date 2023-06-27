import argparse
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import json

import data
import gmm, flow

device = 'cuda' # cuda, cpu

parser = argparse.ArgumentParser("")
parser.add_argument(
    "--config", default="settings.yaml", type=str, help="path to .yaml config"
)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config) as file:
        yaml_cfg = yaml.safe_load(file)
        cfg = json.loads(
            json.dumps(yaml_cfg), object_hook=lambda d: SimpleNamespace(**d)
        )

    x = data.generate(n_samples=2000, dataset=cfg.dataset).to(device)

    if cfg.model == 'gmm':
        gmm.estimate_density(x=x, cfg=cfg.gmm)

    elif cfg.model == 'flow':
        flow.estimate_density(x=x, cfg=cfg.flow)

    else:
        raise NotImplementedError

    print('here')