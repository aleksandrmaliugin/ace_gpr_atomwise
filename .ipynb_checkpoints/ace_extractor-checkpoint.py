import numpy as np
import ase
from itertools import combinations, combinations_with_replacement
from collections import defaultdict
from ase.neighborlist import neighbor_list


class Cluster_Expansion:
    def __init__(
        self,
        atoms: ase.Atoms,
        shells: dict | None = None,
        max_order: int = 3,
        atom_indices: list[int] | None = None,
    ):
        self.atoms = atoms
        self.shells = shells or {}
        self.max_order = max_order
        self.atom_indices = None if atom_indices is None else sorted(set(atom_indices))

        self.elements_list = self.atoms.get_chemical_symbols()
        self.elements = sorted(np.unique(self.elements_list))

        self.clusters = None
        self.descriptor = None
        self.names = None

        self._validate_inputs()
        self.generate_all_descriptors()

    def _validate_inputs(self):
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

        if self.atom_indices is not None:
            n_atoms = len(self.atoms)
            for idx in self.atom_indices:
                if idx < 0 or idx >= n_atoms:
                    raise ValueError(
                        f"Atom index {idx} is out of bounds for {n_atoms} atoms."
                    )

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

        raise ValueError("order must be 1, 2, or 3")

    def _build_single_clusters(self):
        n_atoms = len(self.atoms)
        return {"singles": [[i] for i in range(n_atoms)]}

    def _build_pair_clusters(self):
        pair_clusters = {}

        for shell_name, (rmin, rmax) in self.shells.items():
            i_arr, j_arr, S_arr, d_arr = neighbor_list("ijSd", self.atoms, rmax)

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

    def image_distance(self, j, Sj, k, Sk):
        pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()

        Sj = np.asarray(Sj, dtype=float)
        Sk = np.asarray(Sk, dtype=float)

        rj = pos[j] + Sj @ cell
        rk = pos[k] + Sk @ cell

        return float(np.linalg.norm(rj - rk))

    def _build_triplet_clusters(self, pair_clusters):
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
                        d_jk = self.image_distance(j, Sj, k, Sk)

                        if base_rmin <= d_jk < base_rmax:
                            if (j, Sj) <= (k, Sk):
                                triplet = (center, j, Sj, k, Sk)
                            else:
                                triplet = (center, k, Sk, j, Sj)

                            triplets.append(triplet)

                triplet_clusters[triplet_name] = triplets

        return triplet_clusters

    def build_clusters(self):
        clusters = {}

        if self.max_order >= 1:
            clusters.update(self._build_single_clusters())

        pair_clusters = {}
        if self.max_order >= 2:
            pair_clusters = self._build_pair_clusters()
            clusters.update(pair_clusters)

        if self.max_order >= 3:
            triplet_clusters = self._build_triplet_clusters(pair_clusters)
            clusters.update(triplet_clusters)

        return clusters

    def count_descriptors_atomic(self):
        n_atoms = len(self.elements_list)

        ordered_geom_types = []

        if self.max_order >= 1:
            ordered_geom_types.append("singles")

        if self.max_order >= 2:
            for shell_name in self.shells.keys():
                ordered_geom_types.append(shell_name)

        if self.max_order >= 3:
            for hips_name in self.shells.keys():
                for base_name in self.shells.keys():
                    ordered_geom_types.append(f"trip_hips_{hips_name}_base_{base_name}")

        names = []
        blocks = []

        for geom_type in ordered_geom_types:
            if geom_type == "singles":
                order = 1
            elif geom_type.startswith("trip"):
                order = 3
            else:
                order = 2

            labels = self.chemical_labels_atomic(order)
            label_to_col = {lab: i for i, lab in enumerate(labels)}

            block = np.zeros((n_atoms, len(labels)), dtype=float)
            cluster_list = self.clusters.get(geom_type, [])

            for cluster in cluster_list:
                if geom_type == "singles":
                    center = cluster[0]
                    key = self.elements_list[center]

                elif geom_type.startswith("trip"):
                    center = cluster[0]
                    j = cluster[1]
                    k = cluster[3]
                    neigh = sorted([self.elements_list[j], self.elements_list[k]])
                    key = self.elements_list[center] + neigh[0] + neigh[1]

                else:
                    center = cluster[0]
                    j = cluster[1]
                    key = self.elements_list[center] + self.elements_list[j]

                if self.atom_indices is not None and center not in self.atom_indices:
                    continue

                block[center, label_to_col[key]] += 1.0

            blocks.append(block)
            names.extend([f"{geom_type}:{lab}" for lab in labels])

        descriptor = np.concatenate(blocks, axis=1)
        return descriptor, names

    def generate_all_descriptors(self):
        self.clusters = self.build_clusters()
        self.descriptor, self.names = self.count_descriptors_atomic()
        return 0