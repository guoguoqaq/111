import os
import torch
import random
import h5py
import numpy as np
from typing import Callable, Dict, Union
from enum import Enum
import requests

from .graph import Graph
from .transforms import cells_to_edge_index


class DatasetUrl(Enum):
    """URLs for the datasets."""
    # pOnEllipse datasets
    pOnEllipseTrain  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseTrain.h5'
    pOnEllipseInDist = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseInDist.h5'
    pOnEllipseLowRe  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseLowRe.h5'
    pOnEllipseHighRe = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseHighRe.h5'
    pOnEllipseThin   = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseThin.h5'
    pOnEllipseThick  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/pOnEllipseThick.h5'
    # uvpAroundEllipse datasets
    uvpAroundEllipseTrain  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseTrain.h5'
    uvpAroundEllipseInDist = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseInDist.h5'
    uvpAroundEllipseLowRe  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseLowRe.h5'
    uvpAroundEllipseHighRe = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseHighRe.h5'
    uvpAroundEllipseThin   = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseThin.h5'
    uvpAroundEllipseThick  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/uvpAroundEllipseThick.h5'
    # Files with the number of time-steps in the datasets
    TimeEllipseTrain  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseTrain.npy'
    TimeEllipseInDist = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseInDist.npy'
    TimeEllipseLowRe  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseLowRe.npy'
    TimeEllipseHighRe = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseHighRe.npy'
    TimeEllipseThin   = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseThin.npy'
    TimeEllipseThick  = 'https://huggingface.co/datasets/mariolinov/Ellipse/resolve/main/TimeEllipseThick.npy'
    # pOnWing datasets
    pOnWingTrain  = 'https://huggingface.co/datasets/mariolinov/Wing/resolve/main/pOnWingTrain.h5'
    pOnWingInDist = 'https://huggingface.co/datasets/mariolinov/Wing/resolve/main/pOnWingInDist.h5'
    

class DatasetDownloader:
    """A class to download the datasets.

    Args:
        dataset_url (DatasetUrl): The URL of the dataset.
        path (str, optional): The path where the dataset is downloaded. (default: :obj:`'.'`)
        overwrite (bool, optional): If :obj:`True`, then the dataset is overwritten if it already exists. (default: :obj:`False`)

    Example:
        >>> downloader = DatasetDownloader(DatasetUrl.EllipseTrain, path='.', overwrite=False)
        >>> dataset_path = downloader.file_path
    
    """

    def __init__(
        self,
        dataset_url: DatasetUrl,
        path:        str = '.',
        overwrite:   bool = False,
    ) -> None:
        assert dataset_url in DatasetUrl, f"Dataset URL {dataset_url} not found."
        self.dataset_url = dataset_url
        self.path = path
        self.extension = dataset_url.value.split('.')[-1]
        self.file_path = os.path.join(self.path, self.dataset_url.name + '.' + self.extension)
        if os.path.exists(self.file_path) and not overwrite:
            print(f"Dataset already exists.")
            return
        if os.path.exists(self.file_path) and overwrite:
            print(f"Dataset already exists at {self.path}. Overwriting...")
            os.remove(self.file_path)
        self.download()

    def download(self):
        print(f"Downloading dataset from {self.dataset_url.value}...")
        response = requests.get(self.dataset_url.value)
        with open(self.file_path, 'wb') as f:
            f.write(response.content)
        # Wait for the file to be written
        while not os.path.exists(self.file_path):
            pass
        print("Dataset downloaded.")

    def numpy(
            self,
        ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        r"""Load the dataset as a numpy array. If the dataset is an h5 file, then the data is loaded as a dictionary."""
        if self.extension == 'npy':
            return np.load(self.file_path, allow_pickle=True)
        elif self.extension == 'h5':
            with h5py.File(self.file_path, 'r') as f:
                return {key: np.array(f[key], allow_pickle=True) for key in f.keys()}


class Dataset(torch.utils.data.Dataset):
    r"""A base class for representing a Dataset.

    Args:
        path (string): Path to the h5 file.
        transform (callable, optional): A function/transform that takes in a :obj:`graphs4cfd.graph.Graph` object
            and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        max_indegree (int, optional): The maximum number of edges per node. Applies only if the mesh is provided. (default: :obj:`None`)
        training_info (dict, optional): A dictionary containing values of type :obj:`ìnt` for the keys `n_in`,
            `step` and `T`. (default: :obj:`None`)
        idx (int, optional): The index of the simulation to load. If :obj:`None`, then all the simulations are loaded. (default: :obj:`None`)
        preload (bool, optional): If :obj:`True`, then the data is loaded in memory. If :obj:`False`, then the data
            is loaded from the h5 file at every access. (default: :obj:`False`)
    """

    def __init__(
        self,
        path:          str,
        max_indegree:  int              = None,
        transform:     Callable         = None,
        training_info: Dict             = None,
        idx:           Union[int, list] = None,
        preload:       bool             = False,
    ) -> None:
        self.path          = path
        self.max_indegree  = max_indegree
        self.transform     = transform
        self.training_info = training_info
        if self.training_info is None:
            self.training_info = {}
        self.training_info["n_in"]  = 1
        self.training_info["step"]  = 1
        self.training_sequences_length = self.training_info["n_in"] * self.training_info["step"] - self.training_info["step"] + 1
        self.preload = preload
        # Load only the given simulation idx
        if idx is not None:
            if preload == False:
                raise ValueError(
                    'If input argument to Dataset.__init__() idx is not None, then argument preload must be True.')
            h5_file = h5py.File(self.path, "r")
            self.h5_data = torch.tensor(np.array(h5_file["data"][idx]), dtype=torch.float32)
            self.h5_mesh = torch.tensor(np.array(h5_file['mesh'][idx]), dtype=torch.long) if 'mesh' in h5_file else None
            if self.h5_data.ndim == 2:
                self.h5_data = self.h5_data.unsqueeze(0)
                if self.h5_mesh is not None:
                    self.h5_mesh = self.h5_mesh.unsqueeze(0)
            h5_file.close()
        # Load all the simulations
        else:
            if self.preload:
                self.load()
            else:
                self.h5_data = None
                self.h5_mesh = None

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        if self.h5_data is not None:
            return self.h5_data.shape[0]
        else:
            h5_file = h5py.File(self.path, 'r')
            num_samples = h5_file['data'].shape[0]
            h5_file.close()
            return num_samples

    def __getitem__(
        self,
        idx: int
    ) -> Graph:
        r"""Get the idx-th training sequence."""
        assert self.training_info is not None, "Training info must be provided."
        assert "T" in self.training_info, "T must be provided in the training info."
        if isinstance(self.training_info["T"], int):
            T = self.training_info["T"]
        elif isinstance(self.training_info["T"], np.ndarray):
            T = self.training_info["T"][idx]
        else:
            raise ValueError("T must be of type int or np.ndarray.")
        sequence_start = random.randint(0, T - self.training_sequences_length)
        return self.get_sequence(idx, sequence_start, n_in=self.training_info["n_in"], step_size=self.training_info["step"], cell_list=getattr(self, 'cell_list', False))

    def get_sequence(
        self, 
        idx:            int,
        sequence_start: int = 0,
        n_in:           int = 1, 
        step_size:      int = 1, 
        cell_list:      bool = False,
    ) -> Graph:
        r"""Get the idx-th sequence.

        Args:
            idx (int): The index of the sample.
            sequence_start (int, optional): The starting index of the sequence. (default: :obj:`0`)
            n_in (int, optional): The number of input time-steps. (default: :obj:`1`)
            step_size (int, optional): The step between two consecutive time-steps. (default: :obj:`1`)
            cell_list (bool, optional): If :obj:`True`, then the mesh cells are stored in a list if the mesh is provided. (default: :obj:`False`)

        Returns:
            :obj:`graphs4cfd.graph.Graph`: The graph containing the sequence.
        """
        # Load the data
        if self.preload:
            data = self.h5_data[idx]
            mesh = self.h5_mesh[idx] if self.h5_mesh is not None else None
        else:
            h5_file = h5py.File(self.path, 'r')
            data = torch.tensor(h5_file['data'][idx], dtype=torch.float32)
            mesh = torch.tensor(h5_file['mesh'][idx], dtype=torch.long) if 'mesh' in h5_file else None
            h5_file.close()
        # Compute the indices
        idx0 = sequence_start
        idx1 = sequence_start + n_in * step_size
        # Create the graph (only point cloud)
        graph = self.data2graph(data, idx0, idx1)
        # Transform the mesh cells to graph edges if the mesh is provided
        if mesh is not None:
            # Remove the cells with only negative indices (ghost cells)
            mask = torch.logical_not((mesh < 0).all(dim=1))
            mesh = mesh[mask]
            graph.edge_index = cells_to_edge_index(mesh, max_indegree=self.max_indegree, pos=graph.pos)
            if hasattr(graph, 'pos'):
                graph.edge_attr = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
            if cell_list:
                mask = (mesh >= 0)
                graph.cell_list = [cell[cell_mask] for cell, cell_mask in zip(mesh, mask)]
        else:
            if self.max_indegree is not None:
                print("Warning: max_indegree parameter is not used because the mesh is not provided.")
        # Apply the transformations
        return self.transform(graph) if self.transform is not None else graph

    def load(self):
        r"""Load the dataset in memory."""
        print("Loading dataset:", self.path)
        h5_file = h5py.File(self.path, "r")
        self.h5_data = torch.tensor(np.array(h5_file["data"]), dtype=torch.float32)
        self.h5_mesh = torch.tensor(np.array(h5_file['mesh']), dtype=torch.long) if 'mesh' in h5_file else None
        h5_file.close()
        self.preload = True
  
    def data2graph(
        self,
        data: torch.Tensor,
        idx0: int,
        idx1: int,
    ) -> Graph:
        r"""Convert the data to a `Graph` object."""
        graph = Graph()
        '''
        graph.pos    = ...
        graph.glob   = ...
        graph.loc    = ...
        graph.target  = ...
        graph.omega  = ...
        .
        .
        .
        '''
        return graph
    

class uvpAroundEllipse(Dataset):

    def __init__(
        self,
        *args,
        T,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(T, int):
            T = np.array([T] * super().__len__())
        self.training_info = {'n_in': 1, 'step': 1, 'T': T}

    def data2graph(
        self,
        data: torch.Tensor,
        idx0: int,
        idx1: int
    ) -> Graph:
        # Check number of nodes (not np.nan)
        N = (data[:, 0] == data[:, 0]).sum()
        # Remove np.nan and only keep the real nodes
        data = data[:N]
        # Build graph
        graph = Graph()
        graph.pos  = data[:, :2] # x, y
        graph.glob = data[:,2:3] # Re
        graph.target = data[:, 4:].reshape(N, -1, 6)[:, idx0:idx1, :3].reshape(N, -1) # u, v, p
        # BCs
        '''
        In data[:,3]:
            0 -> Inner flow
            1 -> Periodic boundary
            2 -> Inlet
            3 -> Outlet
            4 -> Wall
        '''
        graph.bound = data[:, 3].type(torch.uint8)
        # Indicate the node types:
        graph.omega = torch.zeros(N, 3)
        # 1. Inner nodes
        graph.omega[(graph.bound == 0) + (graph.bound == 1) + (graph.bound == 3), 0] = 1.
        # 2. Inlet nodes
        graph.omega[graph.bound == 2, 1] = 1.
        # 3. Wall nodes
        graph.omega[graph.bound == 4, 2] = 1.
        return graph


class pOnEllipse(Dataset):

    def __init__(
        self,
        *args,
        T: Union[int, np.ndarray],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(T, int):
            T = np.array([T] * super().__len__())
        self.training_info = {'n_in': 1, 'step': 1, 'T': T}

    def data2graph(
        self,
        data: torch.Tensor,
        idx0: int,
        idx1: int,
    ) -> Graph:
        # Check number of nodes (not np.nan)
        N = (data[:, 0] == data[:, 0]).sum()
        # Remove np.nan and only keep the real nodes
        data = data[:N]
        # Build graph
        graph = Graph()
        graph.pos    = data[:, :2] - data[:, :2].mean(dim=0) # x, y
        graph.glob   = data[:, 2:3] # Re
        graph.loc    = torch.stack([data[:, 1], data[:, 3] - data[:, 1]], dim=-1) # d_bottom, d_top
        graph.target = data[:, 4 + idx0 : 4 + idx1] # p
        return graph
    

class pOnWing(Dataset):

    def __init__(
        self,
        *args,
        T:       Union[int, np.ndarray],
        normals: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(T, int):
            T = np.array([T] * super().__len__())
        self.training_info = {'n_in': 1, 'step': 1, 'T': T}
        self.normals = normals

    def data2graph(
        self,
        data: torch.Tensor,
        idx0: int,
        idx1: int,
    ) -> Graph:
        # Check number of nodes (not np.nan)
        N = (data[:, 0] == data[:, 0]).sum()
        # Remove np.nan and only keep the real nodes
        data = data[:N]
        # Build graph
        graph = Graph()
        graph.pos    = data[:, :3]  # x, y, z
        graph.loc    = data[:, 3:6] # nx, ny, nz
        graph.target = data[:, 6 + idx0 : 6 + idx1]
        return graph