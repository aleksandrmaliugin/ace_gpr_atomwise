import torch
import gpytorch
from contextlib import contextmanager
from linear_operator import to_linear_operator


def pack_structures(structures, device, dtype):
    xs = [torch.as_tensor(x, device=device, dtype=dtype).contiguous() for x in structures]

    ptr = [0]
    for x in xs:
        ptr.append(ptr[-1] + x.shape[0])

    atoms = torch.cat(xs, dim=0)
    ptr = torch.tensor(ptr, device=device, dtype=torch.long)

    return atoms, ptr


def unpack_structure(atoms, ptr, idx):
    start = int(ptr[idx].item())
    end = int(ptr[idx + 1].item())
    return atoms[start:end]


def atomic_sum_kernel_dense(structures_1, structures_2, atom_kernel):
    device = structures_1[0].device
    dtype = structures_1[0].dtype

    K = torch.zeros(
        len(structures_1),
        len(structures_2),
        device=device,
        dtype=dtype,
    )

    for i, X1 in enumerate(structures_1):
        for j, X2 in enumerate(structures_2):
            K_atoms = atom_kernel(X1, X2).to_dense()
            K[i, j] = K_atoms.sum()

    return K

class AtomicSumLookupKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, atom_dim):
        super().__init__()

        self.atom_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=atom_dim)
        )

        self.register_buffer("train_atoms", torch.empty(0, atom_dim))
        self.register_buffer("train_ptr", torch.zeros(1, dtype=torch.long))

        self._query_atoms = None
        self._query_ptr = None

    @property
    def n_train_structures(self):
        return self.train_ptr.numel() - 1

    def set_train_structures(self, structures, device, dtype):
        atoms, ptr = pack_structures(structures, device=device, dtype=dtype)
        self.train_atoms = atoms
        self.train_ptr = ptr

    def _get_structure(self, idx):
        idx = int(idx)

        if idx < self.n_train_structures:
            return unpack_structure(self.train_atoms, self.train_ptr, idx)

        if self._query_atoms is None:
            raise RuntimeError("Query structures are not registered.")

        qidx = idx - self.n_train_structures
        return unpack_structure(self._query_atoms, self._query_ptr, qidx)

    def _structures_from_ids(self, ids):
        ids = ids.view(-1).long()
        return [self._get_structure(i) for i in ids]

    @contextmanager
    def register_query_structures(self, structures):
        device = self.train_atoms.device
        dtype = self.train_atoms.dtype

        self._query_atoms, self._query_ptr = pack_structures(
            structures,
            device=device,
            dtype=dtype,
        )

        start = self.n_train_structures
        test_ids = torch.arange(
            start,
            start + len(structures),
            device=device,
            dtype=torch.long,
        ).unsqueeze(-1)

        try:
            yield test_ids
        finally:
            self._query_atoms = None
            self._query_ptr = None

    def forward(self, x1, x2, diag=False, **params):
        structures_1 = self._structures_from_ids(x1)
        structures_2 = self._structures_from_ids(x2)

        K = atomic_sum_kernel_dense(
            structures_1,
            structures_2,
            self.atom_kernel,
        )

        if diag:
            return torch.diagonal(K)

        return to_linear_operator(K)


class AtomicCountMean(gpytorch.means.Mean):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.constant = torch.nn.Parameter(torch.zeros(()))

    def forward(self, structure_ids):
        ids = structure_ids.view(-1).long()
        counts = []

        for sid in ids:
            X = self.kernel._get_structure(int(sid))
            counts.append(X.shape[0])

        counts = torch.tensor(
            counts,
            dtype=self.constant.dtype,
            device=self.constant.device,
        )

        return counts * self.constant


class AtomicSumExactGP(gpytorch.models.ExactGP):
    def __init__(self, n_train, train_y, likelihood, atom_dim):
        train_ids = torch.arange(
            n_train,
            device=train_y.device,
            dtype=torch.long,
        ).unsqueeze(-1)

        super().__init__(train_ids, train_y, likelihood)

        #self.mean_module = gpytorch.means.ConstantMean()
        
        #self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = AtomicSumLookupKernel(atom_dim)
        self.mean_module = AtomicCountMean(self.covar_module)

    def forward(self, structure_ids):
        mean = self.mean_module(
            structure_ids.to(dtype=self.train_targets.dtype)
        ).squeeze(-1)

        covar = self.covar_module(structure_ids)

        return gpytorch.distributions.MultivariateNormal(mean, covar)