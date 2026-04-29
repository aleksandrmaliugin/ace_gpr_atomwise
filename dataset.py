import torch
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm

import ase
from ase.io import read

from ace_extractor import *


class ACE_Dataset(Dataset):

    PROPERTIES = ["TOTEN", "E_ADS", "FREQ"]

    def __init__(self,
                 target_y: str = "TOTEN",
                 directory: str = None,
                 ref_directory: str = None,
                 ref_ads_directory: str = None,
                 max_order: int = 2,
                 atom_indices: list = None,
                 ):
        """
        """

        if target_y not in self.PROPERTIES:
            raise KeyError(f"Unknown target_y='{target_y}'. Available: {list(self.PROPERTIES)}")

        if target_y == "E_ADS" and not ref_directory:
            raise KeyError("ref_directory must be specified")

        self.directory = directory
        self.ref_directory = ref_directory
        self.ref_ads_directory = ref_ads_directory
        self.max_order = max_order
        self.atom_indices = atom_indices

        self.target_y = target_y

        X, y, conf_dir = self.build_dataset()

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
        self.conf = conf_dir

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_all(self):
        return self.X, self.y

    def build_dataset(self):

        global y_dataset
        dir_list = os.listdir(self.directory)
        dir_list = [x for x in dir_list if not x.startswith(".")]

        outcars = []
        conf_dir = []
        for pathway in dir_list:
            outcars.append(read(f'{self.directory}/{pathway}/OUTCAR'))
            conf_dir.append(pathway)

        mindist = calc_mindist(outcars[0])

        one = mindist * 1.2
        sqrt2 = mindist * np.sqrt(2.6)
        sqrt3 = mindist * np.sqrt(3.6)
        two = mindist * 2.1

        shells = {
            "pairs": (0, one),
            "sqrt2_pairs": (one, sqrt2),
            "sqrt3_pairs": (sqrt2, sqrt3),
            "two_pairs": (sqrt3, two),
        }

        if self.target_y == "TOTEN":
            y_dataset = [outcars[i].get_potential_energy() for i in range(len(outcars))]


        if self.target_y == "E_ADS":

            ref_dir_list = os.listdir(self.ref_directory)
            ref_dir_list = [x for x in ref_dir_list if not x.startswith(".")]

            ref_outcars = []
            for pathway in ref_dir_list:
                ref_outcars.append(read(f'{self.ref_directory}/{pathway}/OUTCAR'))

            energy = np.array([outcars[i].get_potential_energy() for i in range(len(outcars))])
            ref_energy = np.array([ref_outcars[i].get_potential_energy() for i in range(len(ref_outcars))])

            ref_ads = read(f'{self.ref_ads_directory}/OUTCAR').get_potential_energy()

            y_dataset = energy - ref_energy - ref_ads

        if self.target_y == "FREQ":
            raise ValueError("'FREQ' not implemented yet.")
            #y_dataset = [outcars[i].get_frequencies()[-1] for i in range(len(outcars))]


        x_dataset = []
        atom_indices = self.atom_indices

        for i in tqdm(range(len(outcars))):

            if self.target_y == "E_ADS":
                outcars[i], atom_indices = atoms_near_carbon(outcars[i])

                mindist = calc_mindist(outcars[0])

                one = mindist * 1.2
                sqrt2 = mindist * np.sqrt(2.6)
                sqrt3 = mindist * np.sqrt(3.6)
                two = mindist * 2.2

                shells = {
                    "pairs": (0, one),
                    "sqrt2_pairs": (one, sqrt2),
                    "sqrt3_pairs": (sqrt2, sqrt3),
                    "two_pairs": (sqrt3, two),
                }

            descriptors = Cluster_Expansion(
                outcars[i],
                shells=shells,
                max_order=self.max_order,
                atom_indices=atom_indices
            )

            x_dataset.append(descriptors.descriptor)

        x_dataset = [torch.tensor(x, dtype=torch.float64) for x in x_dataset]
        y_dataset = torch.tensor(y_dataset, dtype=torch.float64)

        return x_dataset, y_dataset, conf_dir

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