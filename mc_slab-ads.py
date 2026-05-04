# mpirun -np 4 python test_mpi_ads.py \
#   -c POSCAR \
#   -m slab_gpr.pt \
#   -a ads_gpr.pt \
#   -t 300 \
#   -n 5000 \
#   -o structure_mc.xyz

import os
import sys
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from mpi4py import MPI

import numpy as np
import torch

from scipy.constants import k, eV

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import read, write

from ace_config import ACEConfig
from ace_extractor import ClusterExpansion
from dataset import calc_mindist
from gpr import SparseAtomicGPR


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI multiple-try MC: E = E_slab + E_ads"
    )

    parser.add_argument("-c", "--cell", default=None)
    parser.add_argument(
        "-s", "--supercell",
        type=int,
        nargs=3,
        metavar=("Nx", "Ny", "Nz"),
        default=(3, 3, 6),
    )
    parser.add_argument("-x", "--composition", type=float, default=0.5)
    parser.add_argument("-t", "--temperature", type=float, default=100.0)
    parser.add_argument("-n", "--nsteps", type=int, default=3000)

    parser.add_argument(
        "-m", "--model",
        default="sparse_atomic_gpr.pt",
        help="Slab energy model",
    )
    parser.add_argument(
        "-a", "--ads-model",
        default="sparse_atomic_gpr_ads_best.pt",
        help="Adsorption energy model",
    )

    parser.add_argument("-o", "--output", default="structure_mc.xyz")

    return parser.parse_args()


def build_supercell(
    structure: str,
    lattice_constant: float,
    composition: dict[str, float],
    supercell,
    seed: int | None = None,
) -> Atoms:

    elements = list(composition.keys())

    atoms = bulk(
        elements[0],
        crystalstructure=structure,
        a=lattice_constant,
        cubic=True,
    )

    atoms = make_supercell(atoms, np.array(supercell, dtype=int))

    fractions = np.array(list(composition.values()), dtype=float)
    fractions = fractions / fractions.sum()

    n_atoms = len(atoms)
    counts = np.rint(fractions * n_atoms).astype(int)
    counts[-1] += n_atoms - counts.sum()

    symbols = []
    for el, n in zip(elements, counts):
        symbols.extend([el] * n)

    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)

    atoms.set_chemical_symbols(symbols)

    return atoms


def load_or_build_structure(args) -> Atoms:
    if args.cell is not None:
        if not os.path.exists(args.cell):
            raise FileNotFoundError(f"File {args.cell} does not exist")
        atoms = read(args.cell, format="vasp")

    elif os.path.exists("POSCAR"):
        atoms = read("POSCAR", format="vasp")

    else:
        atoms = build_supercell(
            structure="fcc",
            lattice_constant=3.98398,
            composition={"Pd": args.composition, "Pt": 1.0 - args.composition},
            supercell=[
                [args.supercell[0], 0, 0],
                [0, args.supercell[1], 0],
                [0, 0, args.supercell[2]],
            ],
            seed=1234,
        )

        atoms.center(vacuum=15.0, axis=2)
        atoms.pbc = (True, True, False)

        write("structure_init.xyz", atoms, format="extxyz")

    return atoms


def find_top_center_atom(atoms: Atoms, z_tol: float = 2e-1) -> int:

    positions = atoms.get_positions()
    cell = atoms.get_cell().array

    z = positions[:, 2]
    z_max = z.max()

    top_indices = np.where(z >= z_max - z_tol)[0]

    xy = positions[top_indices, :2]
    xy_center = 0.5 * (cell[0, :2] + cell[1, :2])

    distances = np.linalg.norm(xy - xy_center[None, :], axis=1)

    return int(top_indices[np.argmin(distances)])


def atoms_to_data(atoms: Atoms) -> dict:
    return {
        "numbers": atoms.get_atomic_numbers().astype(np.int32),
        "positions": atoms.get_positions(),
        "cell": atoms.get_cell().array,
        "pbc": atoms.get_pbc(),
    }


def data_to_atoms(data: dict) -> Atoms:
    return Atoms(
        numbers=data["numbers"],
        positions=data["positions"],
        cell=data["cell"],
        pbc=data["pbc"],
    )


def make_config_from_atoms(atoms: Atoms) -> ACEConfig:
    return ACEConfig(
        elements=("Pd", "Pt"),
        mindist=calc_mindist(atoms),
        shells=(0, 1.2, np.sqrt(2.6), np.sqrt(3.6), 2.1),
    )


def load_model_and_extractor(model_file: str, atoms: Atoms):
    model = SparseAtomicGPR(model_path=model_file)
    model.eval()

    config = getattr(model, "config", None)
    if config is None:
        config = make_config_from_atoms(atoms)

    extractor = ClusterExpansion(config)

    return model, extractor


def model_energy(
    atoms: Atoms,
    model: SparseAtomicGPR,
    extractor: ClusterExpansion,
    atom_indices: list[int] | None = None,
) -> float:

    desc = extractor(atoms, atom_indices=atom_indices)

    if isinstance(desc, np.ndarray):
        desc = torch.as_tensor(desc, dtype=torch.float64, device="cpu")
    else:
        desc = desc.to(device="cpu", dtype=torch.float64)

    with torch.no_grad():
        energy = model([desc])

    if isinstance(energy, torch.Tensor):
        energy = energy.detach().cpu().item()

    return float(energy)


def compute_total_energy(
    work_atoms: Atoms,
    numbers: np.ndarray,
    slab_model: SparseAtomicGPR,
    slab_extractor: ClusterExpansion,
    ads_model: SparseAtomicGPR,
    ads_extractor: ClusterExpansion,
    site_index: int,
) -> float:

    work_atoms.set_atomic_numbers(numbers.astype(np.int32))

    e_slab = model_energy(
        atoms=work_atoms,
        model=slab_model,
        extractor=slab_extractor,
        atom_indices=None,
    )

    e_ads = model_energy(
        atoms=work_atoms,
        model=ads_model,
        extractor=ads_extractor,
        atom_indices=[site_index],
    )

    return e_slab + e_ads


def propose_swap(numbers: np.ndarray, frozen_indices: set[int] | None = None) -> np.ndarray:
    numbers = np.asarray(numbers, dtype=np.int32)

    frozen_indices = frozen_indices or set()
    allowed = np.array(
        [i for i in range(len(numbers)) if i not in frozen_indices],
        dtype=int,
    )

    if len(allowed) < 2:
        raise ValueError("Not enough swappable atoms.")

    i = np.random.choice(allowed)

    possible_j = allowed[numbers[allowed] != numbers[i]]

    if len(possible_j) == 0:
        raise ValueError("Only one species is present among swappable atoms.")

    j = np.random.choice(possible_j)

    new_numbers = numbers.copy()
    new_numbers[i], new_numbers[j] = new_numbers[j], new_numbers[i]

    return new_numbers


def choose_by_cdf(dE: np.ndarray, beta: float) -> int:
    dE = np.asarray(dE, dtype=float)

    weights = np.exp(-beta * (dE - dE.min()))
    probs = weights / weights.sum()

    cdf = np.cumsum(probs)
    r = np.random.rand()

    return int(np.searchsorted(cdf, r))


args = parse_args()

if not os.path.exists(args.model):
    if rank == 0:
        print(f"Slab model file {args.model} does not exist", flush=True)
    sys.exit(1)

if not os.path.exists(args.ads_model):
    if rank == 0:
        print(f"Adsorption model file {args.ads_model} does not exist", flush=True)
    sys.exit(1)


T = args.temperature
kB = k / eV
beta = 1.0 / (kB * T)

n_steps = args.nsteps

np.random.seed(1234 + rank)


##########################################
#      Initial structure transfer        #
##########################################

if rank == 0:
    try:
        atoms0 = load_or_build_structure(args)
        site_index = find_top_center_atom(atoms0)

    except Exception as exc:
        print(f"Failed to load/build cell: {exc}", flush=True)
        for dest in range(1, size):
            comm.send({"error": str(exc)}, dest=dest, tag=100)
        sys.exit(1)

    atoms_data = atoms_to_data(atoms0)
    atoms_data["site_index"] = site_index

    print(f"rank 0 initial structure: {atoms0}", flush=True)
    print(f"on-top site index = {site_index}", flush=True)

    for dest in range(1, size):
        comm.send(atoms_data, dest=dest, tag=100)

else:
    atoms_data = comm.recv(source=0, tag=100)

    if isinstance(atoms_data, dict) and "error" in atoms_data:
        print(
            f"rank {rank}: failed to receive cell: {atoms_data['error']}",
            flush=True,
        )
        sys.exit(1)


atoms = data_to_atoms(atoms_data)
site_index = int(atoms_data["site_index"])


##########################################
#              Models                    #
##########################################

work_atoms = atoms.copy()

slab_model, slab_extractor = load_model_and_extractor(args.model, work_atoms)
ads_model, ads_extractor = load_model_and_extractor(args.ads_model, work_atoms)

current_numbers = atoms.get_atomic_numbers().astype(np.int32)

current_energy = compute_total_energy(
    work_atoms=work_atoms,
    numbers=current_numbers,
    slab_model=slab_model,
    slab_extractor=slab_extractor,
    ads_model=ads_model,
    ads_extractor=ads_extractor,
    site_index=site_index,
)

accepted = 0
n_atoms = len(current_numbers)

pd_number = 46
av_comp_sum = np.zeros(n_atoms, dtype=float)
av_comp_count = 0

frozen_indices = {site_index}


##########################################
#           Monte Carlo swaps            #
##########################################

for step in range(n_steps):

    candidate_numbers = propose_swap(
        current_numbers,
        frozen_indices=frozen_indices,
    ).astype(np.int32)

    candidate_energy = compute_total_energy(
        work_atoms=work_atoms,
        numbers=candidate_numbers,
        slab_model=slab_model,
        slab_extractor=slab_extractor,
        ads_model=ads_model,
        ads_extractor=ads_extractor,
        site_index=site_index,
    )

    send_E = np.array([candidate_energy], dtype=np.float64)

    if rank == 0:
        all_E = np.empty(size, dtype=np.float64)
    else:
        all_E = None

    comm.Gather(send_E, all_E, root=0)

    send_numbers = candidate_numbers.astype(np.int32)

    if rank == 0:
        all_numbers = np.empty((size, n_atoms), dtype=np.int32)
    else:
        all_numbers = None

    comm.Gather(send_numbers, all_numbers, root=0)

    if rank == 0:
        dE = np.empty(size + 1, dtype=np.float64)
        dE[0] = 0.0
        dE[1:] = all_E - current_energy

        chosen = choose_by_cdf(dE, beta)

        if chosen == 0:
            new_numbers = current_numbers.copy()
            new_energy = current_energy
        else:
            new_numbers = all_numbers[chosen - 1].copy()
            new_energy = float(all_E[chosen - 1])
            accepted += 1

        package = {
            "chosen": int(chosen),
            "numbers": new_numbers.astype(np.int32),
            "energy": float(new_energy),
        }

        for dest in range(1, size):
            comm.send(package, dest=dest, tag=200)

    else:
        package = comm.recv(source=0, tag=200)

    chosen = package["chosen"]
    current_numbers = package["numbers"].astype(np.int32)
    current_energy = float(package["energy"])

    atoms.set_atomic_numbers(current_numbers)
    work_atoms.set_atomic_numbers(current_numbers)

    if chosen != 0:
        av_comp_sum += (current_numbers == pd_number).astype(float)
        av_comp_count += 1

    if rank == 0 and step % 100 == 0:
        print(
            f"iter = {step:5d}, "
            f"E_total = {current_energy:.8f} eV, "
            f"accepted = {accepted}",
            flush=True,
        )


##########################################
#         Save final structure           #
##########################################

if rank == 0:
    final_atoms = data_to_atoms(atoms_data)
    final_atoms.set_atomic_numbers(current_numbers)

    if av_comp_count > 0:
        av_comp = av_comp_sum / av_comp_count
    else:
        av_comp = (current_numbers == pd_number).astype(float)

    site_marker = np.zeros(n_atoms, dtype=float)
    site_marker[site_index] = 1.0

    final_atoms.set_array("av_comp", av_comp)
    final_atoms.set_array("ads_site", site_marker)

    write(args.output, final_atoms, format="extxyz")

    print(f"Saved {args.output}", flush=True)
    print(f"on-top site index: {site_index}", flush=True)
    print(f"Accepted structures used for av_comp: {av_comp_count}", flush=True)