"""
    This script trains a Variational Graph Autoencoder (VGAE) on the ellipse dataset. This autorncoder is later used to train latent diffusion/flow-matching models.
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
        'name':                 'MODEL_NAME',
        'latent_node_features': 1,
        'kl_reg':               1e-6,
        'depths':               [2,2,1],
        'width':                128,
        'nt':                   10, # Limit the length of the training simulations to 10 timesteps
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name             = experiment['name'],
    folder           = './checkpoints',
    # checkpoint       = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board     = './boards',
    chk_interval     = 1,
    training_loss    = dgn.nn.losses.VaeLoss(kl_reg=experiment['kl_reg']),
    epochs           = 5000,
    batch_size       = 64,
    lr               = 1e-4,
    grad_clip        = {"epoch": 0, "limit": 1},
    scheduler        = {"factor": 0.1, "patience": 50, "loss": "training"},
    stopping         = 1e-8,
    device           = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
transform = transforms.Compose([
    dgn.transforms.MeshEllipse(),                               # Create a mesh on the ellipse
    dgn.transforms.ScaleEdgeAttr(0.02),                         # Scale the relative position stored as `edge_attr`
    dgn.transforms.EdgeCondFreeStreamProjection(),              # Add the projection of the free stream velocity along the edges as `edge_cond`
    dgn.transforms.ScaleAttr('target', vmin=-1.05, vmax=0.84),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('glob',   vmin=500,   vmax=1000),  # Scale the global feature (Re)
    dgn.transforms.ScaleAttr('loc',    vmin=2,     vmax=3.5),   # Scale the local feature (distances to the walls)
    dgn.transforms.MeshCoarsening(                              # Create 3 lower-resolution graphs and normalise the relative position betwen the inter-graph nodes.
        num_scales      =  4,
        rel_pos_scaling = [0.02, 0.06, 0.15, 0.3],
        scalar_rel_pos  = True, 
    ),
    dgn.transforms.Copy('target', 'field'),                     # Because the target is the input field
])
dataset = dgn.datasets.pOnEllipse(
    # The dataset can be downloaded from the web:
    path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnEllipseTrain).file_path,
    # Or you can directly provide the dataset path if already downloaded:
    # path    = "DATASET_PATH",
    T         = experiment['nt'],
    transform = transform,
    preload   = True,
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 8,
)  

# Model
arch = {
    'in_node_features':     1, # p
    'cond_node_features':   3, # Re, d_bottom, d_top
    'cond_edge_features':   3, # x_j - x_i, y_j - y_i, U_\inf projection
    'latent_node_features': experiment['latent_node_features'],
    'depths':               experiment['depths'],
    'fnns_depth':           2,
    'fnns_width':           experiment['width'],
    'aggr':                 'sum',
    'dropout':              0.1,
    'norm_latents':         True,
}
model = dgn.nn.VGAE(arch = arch)

# Training
model.fit(train_settings, dataloader)