import numpy as np
import ase
from itertools import combinations, combinations_with_replacement
from collections import defaultdict
from ase.neighborlist import neighbor_list
from ase.neighborlist import NeighborList

from ace_config import ACEConfig


class ClusterExpansion:
    def __init__(self, config: ACEConfig):
        self.config = config

        self.shells = getattr(config, "shells_dict", config.shells_dict)

        self.max_order = config.max_order
        self.elements = tuple(config.elements)

        self.names = None

        self._validate_config()

    def _validate_config(self):
        if self.max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3.")

        if self.max_order >= 2 and not self.shells:
            raise ValueError("shells must be provided when max_order >= 2.")

        for shell_name, bounds in self.shells.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    f"Shell '{shell_name}' must be a tuple/list of the form (rmin, rmax)."
                )

            rmin, rmax = bounds

            if rmin < 0:
                raise ValueError(f"Shell '{shell_name}' has negative rmin: {rmin}")

            if rmax <= rmin:
                raise ValueError(
                    f"Shell '{shell_name}' must satisfy rmax > rmin, got {(rmin, rmax)}"
                )

    def _validate_atoms(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
    ):
        elements_list = atoms.get_chemical_symbols()

        unknown = set(elements_list) - set(self.elements)
        if unknown:
            raise ValueError(
                f"Unknown elements in atoms: {unknown}. "
                f"Allowed elements: {self.elements}"
            )

        if atom_indices is not None:
            n_atoms = len(atoms)

            for idx in atom_indices:
                if idx < 0 or idx >= n_atoms:
                    raise ValueError(
                        f"Atom index {idx} is out of bounds for {n_atoms} atoms."
                    )

        return elements_list

    def chemical_labels_atomic(self, order: int):
        if order == 1:
            return list(self.elements)

        if order == 2:
            return [c + n for c in self.elements for n in self.elements]

        if order == 3:
            labels = []

            for c in self.elements:
                for a, b in combinations_with_replacement(self.elements, 2):
                    labels.append(c + a + b)

            return labels

        raise ValueError("order must be 1, 2, or 3.")

    def _build_single_clusters(self, atoms: ase.Atoms):
        n_atoms = len(atoms)
        return {"singles": [[i] for i in range(n_atoms)]}

    def _build_pair_clusters(self, atoms: ase.Atoms):
        pair_clusters = {}

        for shell_name, (rmin, rmax) in self.shells.items():
            i_arr, j_arr, S_arr, d_arr = neighbor_list("ijSd", atoms, rmax)

            pairs_in_shell = [
                (int(i), int(j), tuple(map(int, S)), float(d))
                for i, j, S, d in zip(i_arr, j_arr, S_arr, d_arr)
                if i != j and rmin <= d < rmax
            ]

            pairs_in_shell.sort(key=lambda x: (x[0], x[1], x[2]))
            pair_clusters[shell_name] = pairs_in_shell

        return pair_clusters

    @staticmethod
    def pairlist_to_center_dict(pair_list):
        neigh = defaultdict(list)

        for i, j, Sj, d in pair_list:
            neigh[i].append((j, Sj, d))

        return neigh

    @staticmethod
    def image_distance(
        atoms: ase.Atoms,
        j: int,
        Sj: tuple[int, int, int],
        k: int,
        Sk: tuple[int, int, int],
    ):
        positions = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())

        Sj = np.asarray(Sj, dtype=float)
        Sk = np.asarray(Sk, dtype=float)

        rj = positions[j] + Sj @ cell
        rk = positions[k] + Sk @ cell

        return float(np.linalg.norm(rj - rk))

    def _build_triplet_clusters(self, atoms: ase.Atoms, pair_clusters: dict):
        triplet_clusters = {}
        shell_names = list(self.shells.keys())

        for hips_name in shell_names:
            hips_pairs = pair_clusters[hips_name]
            neigh = self.pairlist_to_center_dict(hips_pairs)

            for base_name in shell_names:
                base_rmin, base_rmax = self.shells[base_name]
                triplet_name = f"trip_hips_{hips_name}_base_{base_name}"

                triplets = []

                for center, nbrs in neigh.items():
                    for (j, Sj, _), (k, Sk, _) in combinations(nbrs, 2):
                        d_jk = self.image_distance(atoms, j, Sj, k, Sk)

                        if base_rmin <= d_jk < base_rmax:
                            if (j, Sj) <= (k, Sk):
                                triplet = (center, j, Sj, k, Sk)
                            else:
                                triplet = (center, k, Sk, j, Sj)

                            triplets.append(triplet)

                triplet_clusters[triplet_name] = triplets

        return triplet_clusters

    def build_clusters(self, atoms: ase.Atoms):
        clusters = {}

        if self.max_order >= 1:
            clusters.update(self._build_single_clusters(atoms))

        pair_clusters = {}

        if self.max_order >= 2:
            pair_clusters = self._build_pair_clusters(atoms)
            clusters.update(pair_clusters)

        if self.max_order >= 3:
            triplet_clusters = self._build_triplet_clusters(atoms, pair_clusters)
            clusters.update(triplet_clusters)

        return clusters

    def ordered_geom_types(self):
        geom_types = []

        if self.max_order >= 1:
            geom_types.append("singles")

        if self.max_order >= 2:
            for shell_name in self.shells.keys():
                geom_types.append(shell_name)

        if self.max_order >= 3:
            for hips_name in self.shells.keys():
                for base_name in self.shells.keys():
                    geom_types.append(f"trip_hips_{hips_name}_base_{base_name}")

        return geom_types

    def count_descriptors_atomic(
        self,
        elements_list: list[str],
        clusters: dict,
        atom_indices: list[int] | None = None,
    ):
        n_atoms = len(elements_list)

        selected_centers = None
        if atom_indices is not None:
            selected_centers = set(atom_indices)

        names = []
        blocks = []

        for geom_type in self.ordered_geom_types():
            if geom_type == "singles":
                order = 1
            elif geom_type.startswith("trip"):
                order = 3
            else:
                order = 2

            labels = self.chemical_labels_atomic(order)
            label_to_col = {label: i for i, label in enumerate(labels)}

            block = np.zeros((n_atoms, len(labels)), dtype=float)
            cluster_list = clusters.get(geom_type, [])

            for cluster in cluster_list:
                if geom_type == "singles":
                    center = cluster[0]
                    key = elements_list[center]

                elif geom_type.startswith("trip"):
                    center = cluster[0]
                    j = cluster[1]
                    k = cluster[3]

                    neigh_elements = sorted(
                        [elements_list[j], elements_list[k]]
                    )

                    key = (
                        elements_list[center]
                        + neigh_elements[0]
                        + neigh_elements[1]
                    )

                else:
                    center = cluster[0]
                    j = cluster[1]
                    key = elements_list[center] + elements_list[j]

                if selected_centers is not None and center not in selected_centers:
                    continue

                if key not in label_to_col:
                    raise ValueError(
                        f"Descriptor key '{key}' is not in label list for "
                        f"geometry type '{geom_type}'. "
                        f"Allowed labels: {labels}"
                    )

                block[center, label_to_col[key]] += 1.0

            blocks.append(block)
            names.extend([f"{geom_type}:{label}" for label in labels])

        descriptor = np.concatenate(blocks, axis=1)

        return descriptor, names

    def build_clusters_local(self, atoms: ase.Atoms, atom_indices: list[int]):
        clusters = {}
    
        if self.max_order >= 1:
            clusters["singles"] = [[i] for i in atom_indices]
    
        pair_clusters = {}
    
        if self.max_order >= 2:
            pair_clusters = self._build_pair_clusters_local(atoms, atom_indices)
            clusters.update(pair_clusters)
    
        if self.max_order >= 3:
            clusters.update(self._build_triplet_clusters(atoms, pair_clusters))
    
        return clusters
    
    
    def _build_pair_clusters_local(self, atoms: ase.Atoms, atom_indices: list[int]):
        max_rmax = max(rmax for _, rmax in self.shells.values())
    
        cutoffs = [0.5 * max_rmax] * len(atoms)
    
        nl = NeighborList(
            cutoffs=cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)
    
        positions = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())
    
        pair_clusters = {shell_name: [] for shell_name in self.shells}
    
        for i in atom_indices:
            neigh_indices, offsets = nl.get_neighbors(i)
    
            ri = positions[i]
    
            for j, S in zip(neigh_indices, offsets):
                j = int(j)
                S = tuple(map(int, S))
    
                rj = positions[j] + np.asarray(S) @ cell
                d = float(np.linalg.norm(rj - ri))
    
                for shell_name, (rmin, rmax) in self.shells.items():
                    if rmin <= d < rmax:
                        pair_clusters[shell_name].append((i, j, S, d))
    
        for shell_name in pair_clusters:
            pair_clusters[shell_name].sort(key=lambda x: (x[0], x[1], x[2]))
    
        return pair_clusters

    def generate_all_descriptors(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
    ):
        elements_list = self._validate_atoms(atoms, atom_indices)
    
        if atom_indices is None:
            clusters = self.build_clusters(atoms)
        else:
            clusters = self.build_clusters_local(atoms, atom_indices)
    
        descriptor, names = self.count_descriptors_atomic(
            elements_list=elements_list,
            clusters=clusters,
            atom_indices=atom_indices,
        )
    
        self.names = names
    
        if atom_indices is not None:
            descriptor = descriptor[atom_indices]
    
        return descriptor

    def __call__(
        self,
        atoms: ase.Atoms,
        atom_indices: list[int] | None = None,
    ):
        return self.generate_all_descriptors(
            atoms=atoms,
            atom_indices=atom_indices,
        )