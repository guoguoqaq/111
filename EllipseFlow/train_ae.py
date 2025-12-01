"""
    Train a Variational Graph Autoencoder (VGAE). This is later used to train a Latent Diffusion/Flow-Matching Graph Net.
    Run with:
        python train_ae.py --experiment_id 0 --gpu 0
"""

import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn

torch.multiprocessing.set_sharing_strategy('file_system')


argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int)
argparser.add_argument('--gpu',  type=int, default=0)
args = argparser.parse_args()

# Initial seed
seed = 0
torch.manual_seed(seed)

# Dictionary of experiments
experiment = {
    0: {
        'name':                 'AE_NAME',
        'fnns_width':           126,
        'latent_node_features': 1,
        'kl_reg':               1e-6,
        'depths':               [2,2,1],
        'nt':                   10, # Limit the length of the training simulations to 10 timesteps
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints',
    # checkpoint    = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board  = './boards',
    chk_interval  = 1,
    training_loss = dgn.nn.losses.VaeLoss(kl_reg=experiment['kl_reg']),
    epochs        = 5000,
    batch_size    = 32,
    lr            = 1e-4,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 50, "loss": 'training'},
    stopping      = 1e-8,
    step_sampler  = dgn.nn.diffusion.ImportanceStepSampler,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
transform = transforms.Compose([
    dgn.transforms.ConnectKNN(6),
    dgn.transforms.ScaleEdgeAttr(0.15),
    dgn.transforms.ScaleNs({'u': (-1.8,1.8), 'v': (-1.8,1.8), 'p': (-3, 3), 'Re': (500,1000)}, format='uvp'),
    dgn.transforms.AddDirichletMask(3, [0,1], dirichlet_boundary_id=[2, 4]),
    dgn.transforms.MeshCoarsening(
        num_scales      =  5,
        rel_pos_scaling = [0.15, 0.3, 0.6, 1.2, 2.4],
        scalar_rel_pos  = True, 
    ),
    dgn.transforms.Copy('target', 'field'), # Because the target is the input field
])
dataset = dgn.datasets.uvpAroundEllipse(
    path          = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.uvpAroundEllipseTrain).file_path,
    T             = experiment['nt'],
    transform     = transform,
    preload       = False,
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 16,
)   

# Model
arch = {
    'in_node_features':     3, # u, v, p
    'cond_node_features':   4, # Re, d_inner, d_inlet, d_wall
    'cond_edge_features':   2, # x_j - x_i, y_j - y_i
    'latent_node_features': experiment['latent_node_features'],
    'depths':               experiment['depths'],
    'fnns_depth':           2,
    'fnns_width':           experiment['fnns_width'],
    'aggr':                 'sum',
    'dropout':              0.1,
}
model = dgn.nn.VGAE(arch = arch)
print(f'Number of trainable parameters: {model.num_params}')

# Training
model.fit(train_settings, dataloader)
