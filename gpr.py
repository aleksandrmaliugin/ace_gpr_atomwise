import torch
import torch.nn as nn
from tqdm import tqdm

from ace_config import ACEConfig

class SparseAtomicGPR(nn.Module):
    def __init__(
        self,
        x_train=None,
        model_path=None,
        config=None,
        M=100,
        div=0.001,
        init_lengthscale=1.0,
        init_sigma2=1e-4,
        init_outputscale=1.0,
        jitter=1e-8,
        device=None,
    ):
        super().__init__()

        self.device = device or "cpu"
        self.jitter = jitter

        self.register_buffer("y_train", None)
        
        self.register_buffer("x_M", None)
        self.register_buffer("K_NM_train", None)
        self.register_buffer("c", None)
        
        self.register_buffer("L_KMM", None)
        self.register_buffer("L_KSS", None)

        self.config = config

        if model_path is not None:
            state = torch.load(model_path, map_location=self.device, weights_only=False)

            self.y_train = state["y_train"].to(self.device) if state["y_train"] is not None else None
            
            self.x_M = state["x_M"].to(self.device)
            self.K_NM_train = state["K_NM_train"].to(self.device) if state["K_NM_train"] is not None else None
            self.c = state["c"].to(self.device) if state["c"] is not None else None
            
            self.log_sigma2 = nn.Parameter(state["log_sigma2"].to(self.device))
            
            self.L_KMM = state["L_KMM"].to(self.device) if state["L_KMM"] is not None else None
            self.L_KSS = state["L_KSS"].to(self.device) if state["L_KSS"] is not None else None

            self.log_lengthscale = nn.Parameter(
                state.get(
                    "log_lengthscale",
                    torch.ones(state["x_M"].shape[1], dtype=torch.float64, device=self.device).log()
                ).to(self.device)
            )
            
            self.log_outputscale = nn.Parameter(
                state.get(
                    "log_outputscale",
                    torch.tensor(1.0, dtype=torch.float64, device=self.device).log()
                ).to(self.device)
            )

            self.config = ACEConfig.from_dict(state["config"]) if state["config"] is not None else None
        
        else:
            if x_train is None:
                raise ValueError("Provide x_train if model_path is not given.")

            self.log_lengthscale = nn.Parameter(
                torch.full(
                    (x_train[0].shape[1],),
                    float(init_lengthscale),
                    dtype=torch.float64,
                    device=self.device,
                ).log()
            )

            self.log_sigma2 = nn.Parameter(
                torch.tensor(init_sigma2, dtype=torch.float64, device=self.device).log()
            )

            self.log_outputscale = nn.Parameter(
                torch.tensor(init_outputscale, dtype=torch.float64, device=self.device).log()
            )

            x_M = self.select_inducing_points(x_train, M=M, div=div)
            self.x_M = x_M.to(self.device)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale) + 1e-6

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2) + 1e-12

    @property
    def outputscale(self):
        return torch.exp(self.log_outputscale) + 1e-12

    def rbf_kernel(self, x1, x2):
        device = self.log_lengthscale.device

        x1 = torch.as_tensor(x1, dtype=torch.float64, device=device)
        x2 = torch.as_tensor(x2, dtype=torch.float64, device=device)

        diff = x1[:, None, :] - x2[None, :, :]
        diff = diff / self.lengthscale[None, None, :]

        dist2 = (diff ** 2).sum(dim=-1)
        K_rbf = torch.exp(-0.5 * dist2)

        return self.outputscale * K_rbf

    @torch.no_grad()
    def select_inducing_points(self, x_train, M=100, div=0.001):
        
        all_atoms = torch.cat(
            [torch.as_tensor(s, dtype=torch.float64, device=self.device) for s in x_train],
            dim=0,
        )

        x_M = [all_atoms[0]]

        for atom in tqdm(all_atoms[1:]):
            xm = torch.stack(x_M, dim=0)

            k = self.rbf_kernel(atom[None, :], xm).squeeze(0)

            k_xx = self.rbf_kernel(atom[None, :], atom[None, :]).squeeze()
            k_mm = torch.diag(self.rbf_kernel(xm, xm))
            sims = k / torch.sqrt(k_xx * k_mm)

            if sims.max() < div:
                x_M.append(atom)

            if len(x_M) == M:
                break

        print(f"Selected {len(x_M)} inducing points out of {M}")

        return torch.stack(x_M, dim=0)

    def build_K_NM(self, x_train):
        
        S = len(x_train)
        M = self.x_M.shape[0]

        K_NM = torch.zeros(
            S,
            M,
            dtype=torch.float64,
            device=self.x_M.device,
        )

        for s, x in enumerate(x_train):
            x = torch.as_tensor(x, dtype=torch.float64, device=self.x_M.device)

            K = self.rbf_kernel(x, self.x_M)

            K_NM[s] = K.sum(dim=0)

        return K_NM

    def safe_cholesky(self, A, jitter=None, max_tries=4):
        if jitter is None:
            jitter = self.jitter

        A = 0.5 * (A + A.T)

        for _ in range(max_tries):
            try:
                A_try = A.clone()
                A_try.diagonal().add_(jitter)
                return torch.linalg.cholesky(A_try)
            except torch.linalg.LinAlgError:
                jitter *= 10.0

        raise RuntimeError(f"Cholesky failed, final jitter={jitter:.2e}")

    def solve_c(self, K_NM, y):
        
        K_MM = self.rbf_kernel(self.x_M, self.x_M)
    
        # A = K_MM + (K_NM.T @ K_NM) / self.sigma2    
        A = K_MM + (K_NM.T @ K_NM) / self.sigma2
    
        b = (K_NM.T @ y) / self.sigma2
    
        L = self.safe_cholesky(A)
        c = torch.cholesky_solve(b[:, None], L).squeeze(-1)
    
        return c

    def training_loss(self, train_x, train_y):
        train_y = torch.as_tensor(train_y, dtype=torch.float64, device=self.x_M.device)

        K_NM = self.build_K_NM(train_x)

        c = self.solve_c(K_NM, train_y)

        y_pred = K_NM @ c
        loss = ((y_pred - train_y) ** 2).mean()

        return loss, y_pred, c

    def fit_c(self, train_x, train_y, build_uncertainty=False):
        train_y = torch.as_tensor(train_y, dtype=torch.float64, device=self.x_M.device)

        K_NM = self.build_K_NM(train_x)

        c = self.solve_c(K_NM, train_y)

        self.c = c.detach()
        self.K_NM_train = K_NM.detach()
        self.y_train = train_y.detach()
        
        K_MM = self.rbf_kernel(self.x_M, self.x_M).detach()
        self.L_KMM = self.safe_cholesky(K_MM)

        if build_uncertainty == True:

            K_MM_inv_K_NM_T = torch.cholesky_solve(
                self.K_NM_train.T,
                self.L_KMM,
            )
    
            K_SS = self.K_NM_train @ K_MM_inv_K_NM_T
            K_SS = K_SS + self.sigma2.detach() * torch.eye(
                K_SS.shape[0],
                dtype=torch.float64,
                device=self.x_M.device,
            )
    
            self.L_KSS = self.safe_cholesky(K_SS)

        return self.c

    def fit_c_no_grad(self, train_x, train_y, build_uncertainty=False):
        with torch.no_grad():
            return self.fit_c(train_x, train_y, build_uncertainty)

    def check_descriptor_dim(self, x):
        if isinstance(x, torch.Tensor):
            D = x.shape[1]
        else:
            D = x[0].shape[1]
    
        if D != self.x_M.shape[1]:
            raise ValueError(
                f"Descriptor dimension mismatch: got {D}, "
                f"expected {self.x_M.shape[1]}"
            )

    def forward(self, x):
        self.check_descriptor_dim(x)
    
        if self.c is None:
            raise RuntimeError("Call fit_c(x, y) before prediction.")
    
        K_NM = self.build_K_NM(x)
    
        return K_NM @ self.c

    def predict_uncertainty(self, x):
        if self.c is None:
            raise RuntimeError("Call fit_c(train_x, train_y) first.")

        if self.K_NM_train is None or self.L_KMM is None or self.L_KSS is None:
            raise RuntimeError("Uncertainty matrices are missing. Re-run fit_c().")

        K_NM_test = self.build_K_NM(x)

        mean = K_NM_test @ self.c

        K_MM_inv_K_NM_train_T = torch.cholesky_solve(
            self.K_NM_train.T,
            self.L_KMM,
        )

        K_star_S = K_NM_test @ K_MM_inv_K_NM_train_T

        K_MM_inv_K_NM_test_T = torch.cholesky_solve(
            K_NM_test.T,
            self.L_KMM,
        )

        K_star_star = K_NM_test @ K_MM_inv_K_NM_test_T

        tmp = torch.cholesky_solve(
            K_star_S.T,
            self.L_KSS,
        )

        cov = K_star_star - K_star_S @ tmp
        var = torch.clamp(cov.diagonal(), min=1e-12)
        std = torch.sqrt(var)

        return mean, std

    def save(self, path):
        torch.save(
            {
                "x_M": self.x_M.detach(),
                "log_lengthscale": self.log_lengthscale.detach(),
                "log_sigma2": self.log_sigma2.detach(),
                "log_outputscale": self.log_outputscale.detach(),
                "c": self.c.detach() if self.c is not None else None,
                "K_NM_train": self.K_NM_train.detach() if self.K_NM_train is not None else None,
                "y_train": self.y_train.detach() if self.y_train is not None else None,
                "L_KMM": self.L_KMM.detach() if self.L_KMM is not None else None,
                "L_KSS": self.L_KSS.detach() if self.L_KSS is not None else None,
                "config": self.config.to_dict() if self.config is not None else None,
            },
            path,
        )