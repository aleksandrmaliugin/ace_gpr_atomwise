import ase
import torch
import numpy as np
import gpytorch

from ace_extractor import *
from dataset import calc_mindist, atoms_near_carbon


class Calculator:
    def __init__(self, model, mode="TOTEN", dtype=torch.float64, device=None):
        self.model = model
        self.mode = mode
        self.dtype = dtype

        if device is None:
            device = next(model.parameters()).device

        self.device = device

    def build_descriptor(self, atoms: ase.Atoms):
        if self.mode == "TOTEN":
            atom_indices = None
            mindist = calc_mindist(atoms)
            two = mindist * 2.1

        elif self.mode == "E_ADS":
            atoms, atom_indices = atoms_near_carbon(atoms)
            mindist = calc_mindist(atoms)
            two = mindist * 2.2

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        one = mindist * 1.2
        sqrt2 = mindist * np.sqrt(2.6)
        sqrt3 = mindist * np.sqrt(3.6)

        shells = {
            "pairs": (0, one),
            "sqrt2_pairs": (one, sqrt2),
            "sqrt3_pairs": (sqrt2, sqrt3),
            "two_pairs": (sqrt3, two),
        }

        descriptor = Cluster_Expansion(
            atoms=atoms,
            max_order=3,
            shells=shells,
            atom_indices=atom_indices,
        ).descriptor

        return torch.tensor(
            descriptor,
            dtype=self.dtype,
            device=self.device,
        )

    def __call__(self, atoms: ase.Atoms):
        self.model.eval()

        X_new = self.build_descriptor(atoms)

        with self.model.covar_module.register_query_structures([X_new]) as test_ids:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self.model(test_ids)

        energy = pred.mean.detach().cpu().numpy()
        stddev = pred.stddev.detach().cpu().numpy()

        return energy, stddev