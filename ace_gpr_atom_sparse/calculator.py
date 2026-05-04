import numpy as np
import torch

from ase.calculators.calculator import Calculator, all_changes


class ACEGPRCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, model, extractor, device="cpu", dtype=torch.float64, **kwargs):
        super().__init__(**kwargs)

        self.model = model.to(device)
        self.model.eval()

        self.extractor = extractor
        self.device = device
        self.dtype = dtype

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        desc = self.extractor(atoms)

        if isinstance(desc, np.ndarray):
            desc = torch.as_tensor(desc, dtype=self.dtype, device=self.device)
        else:
            desc = desc.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            energy = self.model([desc])

        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().item()

        self.results["energy"] = float(energy)