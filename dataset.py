import torch
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
from typing import Sequence

import ase
from ase.neighborlist import neighbor_list

from ace_extractor import ClusterExpansion
from ace_config import ACEConfig


class ACEDataset(Dataset):

    def __init__(self,
                 atoms: list[ase.Atoms],
                 config: ACEConfig,
                 extractor: ClusterExpansion,
                 atom_indices: list[int] | None = None,
                 target_y: np.ndarray | torch.Tensor | Sequence | None = None,
                 dtype: torch.dtype = torch.float32,
                 ):

        self.atoms = atoms
        self.atom_indices = atom_indices
        self.extractor = extractor
        self.target_y = target_y
        self.dtype = dtype

        X, y = self.build_dataset()

        if not isinstance(X, list):
            raise ValueError("X must be list of tensors")
        
        if len(X) != y.shape[0]:
            raise ValueError("X and y must have the same number of structures")

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        #if X.shape[0] != y.shape[0]:
        #    raise ValueError("X and y must have the same number of samples")

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_all(self):
        return self.X, self.y

    def build_dataset(self):

        x_dataset = []

        for i in tqdm(range(len(self.atoms))):

            if self.atom_indices is not None:
                descriptors = self.extractor(atoms = self.atoms[i],
                                             atom_indices = self.atom_indices[i])
            else:
                descriptors = self.extractor(atoms = self.atoms[i])

            x_dataset.append(descriptors)

        x_dataset = [torch.tensor(x, dtype=self.dtype) for x in x_dataset]
        y_dataset = torch.tensor(self.target_y, dtype=self.dtype)

        return x_dataset, y_dataset

def atoms_near_carbon(
        atoms,
        radius = 2.4,
        carbon_symbol="C",
        oxygen_symbol="O",
        exclude_carbon=True,
        return_distance=False,
        return_carbon_index=False,
):
    symbols = np.array(atoms.get_chemical_symbols())
    carbon_idx = np.where(symbols == carbon_symbol)[0]

    def _empty_result(atoms_cut):
        if return_distance and return_carbon_index:
            return (
                atoms_cut,
                np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=int),
            )
        elif return_distance:
            return atoms_cut, np.array([], dtype=int), np.array([], dtype=float)
        elif return_carbon_index:
            return atoms_cut, np.array([], dtype=int), np.array([], dtype=int)
        return atoms_cut, np.array([], dtype=int)

    if len(carbon_idx) == 0:
        return _empty_result(atoms[:0])

    # Search in the original structure
    i, j, d = neighbor_list("ijd", atoms, radius)

    # Keep only pairs where atom i is carbon
    mask = np.isin(i, carbon_idx)
    i_c = i[mask]
    j_c = j[mask]
    d_c = d[mask]

    # Exclude carbon self-pairs
    if exclude_carbon:
        non_self = j_c != i_c
        i_c = i_c[non_self]
        j_c = j_c[non_self]
        d_c = d_c[non_self]

    if len(j_c) == 0:
        atoms_cut = atoms[:int(np.min(carbon_idx))]
        return _empty_result(atoms_cut)

    # Exclude oxygen atoms from the final selection
    non_oxygen = symbols[j_c] != oxygen_symbol
    i_c = i_c[non_oxygen]
    j_c = j_c[non_oxygen]
    d_c = d_c[non_oxygen]

    if len(j_c) == 0:
        atoms_cut = atoms[:int(np.min(carbon_idx))]
        return _empty_result(atoms_cut)

    # For each selected atom, keep only the nearest carbon
    best = {}
    for c_idx, atom_idx, dist in zip(i_c, j_c, d_c):
        if atom_idx not in best or dist < best[atom_idx][0]:
            best[atom_idx] = (dist, c_idx)

    selected = np.array(sorted(best.keys()), dtype=int)

    # Cut atoms only at the end
    first_carbon = int(np.min(carbon_idx))
    atoms_cut = atoms[:first_carbon]

    outputs = [atoms_cut, selected]

    if return_distance:
        distances = np.array([best[idx][0] for idx in selected], dtype=float)
        outputs.append(distances)

    if return_carbon_index:
        nearest_carbons = np.array([best[idx][1] for idx in selected], dtype=int)
        outputs.append(nearest_carbons)

    return atoms_cut, selected

def calc_mindist(
        atoms: ase.atoms
):
    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)

    dmin = D.min()

    return dmin