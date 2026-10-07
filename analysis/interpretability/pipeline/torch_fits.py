"""PyTorch backend for the heavy fits of the full run (float64; CUDA on HoreKa, CPU for tests).

Same objectives as the CPU reference code, solved by a damped Newton method instead of scipy's L-BFGS-B:

* choice (conditional logit / top-k Plackett–Luce with position effects), ``page_readiness_ordering.fit_choice_model``:
  f(θ) = −(1/W) Σ_s w_s [u_chosen(s) − log Σ_{r∈s} exp u_r] + ½·ridge·‖θ‖², u_r = x_r·β + δ[position_r], δ[0] = 0;
* admission (Chamberlain conditional logit), ``funnel_models.fit_admission``:
  f(θ) = −(1/W) Σ_g w_g [Σ_{j∈S_g} η_j − log e_{m_g}(exp η_g)] + ½·ridge·‖θ‖², η = Xθ.

Both are strictly convex (ridge), so the optimum is unique and Newton converges to it more tightly than the CPU optimizer.
Gradients and Hessians are exact: closed form for the choice model; for the admission model the inclusion probabilities
and their covariance come from automatic differentiation of the max-scaled elementary-symmetric-polynomial recursion.
A problem object keeps the design resident on the device, so bootstrap draws (new weights), shuffles (only the
x-dependent columns change) and drop-one-block fits (``free`` mask: dropped coefficients fixed at 0, same objective as a
fit without those columns) reuse it. Failure semantics match the CPU fits: non-convergence raises RuntimeError.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

DTYPE = torch.float64
GRAD_TOL = 1e-9          # converged when max |gradient| of the mean objective is below this
STALL_GRAD_TOL = 1e-7    # ... or when the line search cannot decrease the loss any more and max |gradient| is below this
MAX_ITER = 200
CHUNK_ROWS = 2_000_000


def device_of(backend: str) -> torch.device:
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if backend == "torch-cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown torch backend {backend}")


def _result(theta, fun, success, message, iterations):
    return SimpleNamespace(x=theta.detach().cpu().numpy().astype(float), fun=float(fun), success=success, message=message,
                           nit=iterations)


def newton(loss_grad_hess, loss_only, theta: torch.Tensor, free: torch.Tensor, max_iter: int = MAX_ITER):
    """Damped Newton with Armijo backtracking on the free parameters (the others stay fixed)."""
    f, g, H = loss_grad_hess(theta)
    for it in range(max_iter):
        gf = g[free]
        gmax = float(gf.abs().max()) if gf.numel() else 0.0
        if gmax < GRAD_TOL:
            return _result(theta, f, True, "gradient below tolerance", it)
        Hf = H[free][:, free]
        try:
            step = torch.linalg.solve(Hf, gf)
        except RuntimeError:
            step = torch.linalg.lstsq(Hf, gf.unsqueeze(1)).solution.squeeze(1)
        slope = float(gf @ step)
        if not np.isfinite(slope) or slope <= 0:
            step, slope = gf, float(gf @ gf)             # fall back to steepest descent
        alpha = 1.0
        while True:
            trial = theta.clone()
            trial[free] = trial[free] - alpha * step
            f_new = loss_only(trial)
            if np.isfinite(f_new) and f_new <= f - 1e-4 * alpha * slope:
                break
            alpha *= 0.5
            if alpha < 1e-12:
                if gmax < STALL_GRAD_TOL:
                    return _result(theta, f, True, "line search stalled at a stationary point", it)
                raise RuntimeError(f"newton fit did not converge: line search failed (max |grad| {gmax:.3g})")
        theta = trial
        f, g, H = loss_grad_hess(theta)
    raise RuntimeError(f"newton fit did not converge in {max_iter} iterations")


class ChoiceProblem:
    """Conditional-logit choice sets (``data``: set, chosen, position, sets, levels) with a resident design."""

    def __init__(self, X: np.ndarray, data, device: torch.device, ridge: float = 1e-4):
        sets = np.asarray(data.set, np.int64)
        self.order = None
        if len(sets) and not np.all(np.diff(sets) >= 0):
            self.order = np.argsort(sets, kind="stable")
        take = (lambda a: a) if self.order is None else (lambda a: np.asarray(a)[self.order])
        self.device, self.ridge = device, ridge
        self.k = X.shape[1]
        self.levels = int(data.levels)
        self.S = int(data.sets)
        self.X = torch.as_tensor(np.ascontiguousarray(take(np.asarray(X, float))), dtype=DTYPE, device=device)
        self.set = torch.as_tensor(take(sets), device=device)
        self.chosen = torch.as_tensor(take(np.asarray(data.chosen, bool)), device=device).to(DTYPE)
        self.position = torch.as_tensor(take(np.asarray(data.position, np.int64)), device=device)
        # row chunks aligned to set boundaries
        n = len(sets)
        cuts = [0]
        set_sorted = take(sets)
        while cuts[-1] < n:
            end = min(cuts[-1] + CHUNK_ROWS, n)
            if end < n:
                end = int(np.searchsorted(set_sorted, set_sorted[end], side="left"))
                if end <= cuts[-1]:
                    end = int(np.searchsorted(set_sorted, set_sorted[cuts[-1]], side="right"))
            cuts.append(end)
        self.chunks = list(zip(cuts[:-1], cuts[1:]))
        self.p = self.k + self.levels - 1

    def set_columns(self, columns: list, values: np.ndarray) -> None:
        values = np.asarray(values, float)
        if self.order is not None:
            values = values[self.order]
        self.X[:, columns] = torch.as_tensor(values, dtype=DTYPE, device=self.device)

    def _pieces(self, theta, weight, need_hess):
        beta, delta = theta[:self.k], torch.cat([theta.new_zeros(1), theta[self.k:]])
        W = float(weight.sum())
        loss = theta.new_zeros(())
        grad = theta.new_zeros(self.p)
        hess = theta.new_zeros(self.p, self.p) if need_hess else None
        for lo, hi in self.chunks:
            X, s, ch, pos = self.X[lo:hi], self.set[lo:hi], self.chosen[lo:hi], self.position[lo:hi]
            s0 = int(s[0])
            ls = s - s0
            nS = int(s[-1]) - s0 + 1
            u = X @ beta + delta[pos]
            peak = torch.full((nS,), -torch.inf, dtype=DTYPE, device=self.device).scatter_reduce(0, ls, u, "amax")
            e = torch.exp(u - peak[ls])
            den = torch.zeros(nS, dtype=DTYPE, device=self.device).index_add_(0, ls, e)
            prob = e / den[ls]
            chosen_u = torch.zeros(nS, dtype=DTYPE, device=self.device).index_add_(0, ls, u * ch)
            w = weight[s0:s0 + nS]
            loss = loss - (w * (chosen_u - (peak + torch.log(den)))).sum() / W
            wrow = w[ls] / W
            r = (prob - ch) * wrow
            grad[:self.k] += X.T @ r
            grad[self.k:] += torch.zeros(self.levels, dtype=DTYPE, device=self.device).index_add_(0, pos, r)[1:]
            if need_hess:
                d = prob * wrow
                onehot = torch.nn.functional.one_hot(pos, self.levels).to(DTYPE)[:, 1:]
                Z = torch.cat([X, onehot], dim=1)
                hess += Z.T @ (Z * d[:, None])
                M = torch.zeros(nS, self.p, dtype=DTYPE, device=self.device).index_add_(0, ls, Z * prob[:, None])
                hess -= M.T @ (M * (w / W)[:, None])
        loss = loss + 0.5 * self.ridge * (theta @ theta)
        grad = grad + self.ridge * theta
        if need_hess:
            hess += self.ridge * torch.eye(self.p, dtype=DTYPE, device=self.device)
        return float(loss), grad, hess

    def fit(self, set_weight=None, start=None, free=None):
        weight = (torch.ones(self.S, dtype=DTYPE, device=self.device) if set_weight is None
                  else torch.as_tensor(np.asarray(set_weight, float), dtype=DTYPE, device=self.device))
        theta = (torch.zeros(self.p, dtype=DTYPE, device=self.device) if start is None
                 else torch.as_tensor(np.asarray(start, float), dtype=DTYPE, device=self.device).clone())
        mask = torch.ones(self.p, dtype=torch.bool, device=self.device)
        if free is not None:
            mask[:self.k] = torch.as_tensor(np.asarray(free, bool), device=self.device)
            theta[~mask] = 0.0
        with torch.no_grad():
            return newton(lambda t: self._pieces(t, weight, True), lambda t: self._pieces(t, weight, False)[0], theta, mask)


class AdmissionProblem:
    """Chamberlain conditional logit over padded groups (``data``: AdmissionData) with a resident design."""

    def __init__(self, X: np.ndarray, data, device: torch.device, ridge: float = 1e-4, group_chunk: int = 50_000):
        self.device, self.ridge = device, ridge
        self.k = X.shape[1]
        self.G, self.n = int(data.groups), int(data.width)
        self.group = torch.as_tensor(np.asarray(data.group, np.int64), device=device)
        self.slotx = torch.as_tensor(np.asarray(data.slot, np.int64), device=device)
        self.adm = torch.as_tensor(np.asarray(data.admitted, float), dtype=DTYPE, device=device)
        self.m = torch.as_tensor(np.asarray(data.m, np.int64), device=device)
        self.top = int(np.max(data.m)) if len(data.m) else 0
        self.X = torch.as_tensor(np.ascontiguousarray(np.asarray(X, float)), dtype=DTYPE, device=device)
        self.flat = self.group * self.n + self.slotx
        self.group_chunk = group_chunk
        self.p = self.k

    def set_columns(self, columns: list, values: np.ndarray) -> None:
        self.X[:, columns] = torch.as_tensor(np.asarray(values, float), dtype=DTYPE, device=self.device)

    def _log_esp(self, V, valid, m):
        """log e_m(exp V) per row of V (padding: valid False), max-scaled as funnel_models.admission_loss."""
        c = torch.where(valid, V, torch.full_like(V, -torch.inf)).max(dim=1, keepdim=True).values.detach()
        Wt = torch.where(valid, torch.exp(V - c), torch.zeros_like(V))
        E = [torch.ones(V.shape[0], dtype=DTYPE, device=V.device)] + [torch.zeros(V.shape[0], dtype=DTYPE, device=V.device)
                                                                      for _ in range(self.top)]
        for j in range(V.shape[1]):
            w = Wt[:, j]
            E = [E[0]] + [E[a] + w * E[a - 1] for a in range(1, self.top + 1)]
        Em = torch.stack(E, dim=1).gather(1, m[:, None]).squeeze(1)
        return torch.log(Em) + m.to(DTYPE) * c.squeeze(1)

    def _pieces(self, theta, weight, need_hess):
        W = float(weight.sum())
        eta_rows = self.X @ theta
        loss = theta.new_zeros(())
        grad = theta.new_zeros(self.p)
        hess = theta.new_zeros(self.p, self.p) if need_hess else None
        # rows are grouped by group (admission_data requires sorted answers), so chunk by group ranges
        bounds = torch.searchsorted(self.group, torch.arange(0, self.G + self.group_chunk, self.group_chunk, device=self.device)
                                    .clamp(max=self.G))
        for c in range(len(bounds) - 1):
            lo, hi = int(bounds[c]), int(bounds[c + 1])
            if hi <= lo:
                continue
            g0 = int(self.group[lo])
            gl = self.group[lo:hi] - g0
            nG = int(self.group[hi - 1]) - g0 + 1
            idx = gl * self.n + self.slotx[lo:hi]
            V = torch.zeros(nG * self.n, dtype=DTYPE, device=self.device)
            valid = torch.zeros(nG * self.n, dtype=torch.bool, device=self.device)
            V[idx] = eta_rows[lo:hi]
            valid[idx] = True
            V, valid = V.view(nG, self.n), valid.view(nG, self.n)
            m = self.m[g0:g0 + nG]
            w = weight[g0:g0 + nG]
            with torch.enable_grad():
                Vg = V.clone().requires_grad_(True)
                logZ = self._log_esp(Vg, valid, m)
                (pi,) = torch.autograd.grad(logZ.sum(), Vg, create_graph=need_hess)
                if need_hess:
                    C = torch.stack([torch.autograd.grad(pi[:, j].sum(), Vg, retain_graph=True)[0] for j in range(self.n)], dim=1)
            pi = pi.detach()
            adm_sum = torch.zeros(nG, dtype=DTYPE, device=self.device).index_add_(0, gl, eta_rows[lo:hi] * self.adm[lo:hi])
            loss = loss - (w * (adm_sum - logZ.detach())).sum() / W
            pi_rows = pi.reshape(-1)[idx]
            r = -(self.adm[lo:hi] - pi_rows) * (w[gl] / W)
            grad += self.X[lo:hi].T @ r
            if need_hess:
                Xp = torch.zeros(nG * self.n, self.k, dtype=DTYPE, device=self.device)
                Xp[idx] = self.X[lo:hi]
                Xp = Xp.view(nG, self.n, self.k)
                T = torch.einsum("gjl,glk->gjk", C.detach(), Xp)
                hess += torch.einsum("gjk,gjq,g->kq", Xp, T, w / W)
        loss = loss + 0.5 * self.ridge * (theta @ theta)
        grad = grad + self.ridge * theta
        if need_hess:
            hess += self.ridge * torch.eye(self.p, dtype=DTYPE, device=self.device)
        return float(loss), grad, hess

    def inclusion(self, theta: np.ndarray) -> np.ndarray:
        """Inclusion probabilities per row (for checks against funnel_models._inclusion)."""
        t = torch.as_tensor(np.asarray(theta, float), dtype=DTYPE, device=self.device)
        eta = self.X @ t
        V = torch.zeros(self.G * self.n, dtype=DTYPE, device=self.device)
        valid = torch.zeros(self.G * self.n, dtype=torch.bool, device=self.device)
        V[self.flat] = eta
        valid[self.flat] = True
        Vg = V.view(self.G, self.n).clone().requires_grad_(True)
        logZ = self._log_esp(Vg, valid.view(self.G, self.n), self.m)
        (pi,) = torch.autograd.grad(logZ.sum(), Vg)
        return pi.reshape(-1)[self.flat].cpu().numpy()

    def fit(self, weight=None, start=None, free=None):
        wt = (torch.ones(self.G, dtype=DTYPE, device=self.device) if weight is None
              else torch.as_tensor(np.asarray(weight, float), dtype=DTYPE, device=self.device))
        theta = (torch.zeros(self.p, dtype=DTYPE, device=self.device) if start is None
                 else torch.as_tensor(np.asarray(start, float), dtype=DTYPE, device=self.device).clone())
        mask = torch.ones(self.p, dtype=torch.bool, device=self.device)
        if free is not None:
            mask[:] = torch.as_tensor(np.asarray(free, bool), device=self.device)
            theta[~mask] = 0.0
        with torch.no_grad():
            return newton(lambda t: self._pieces(t, wt, True), lambda t: self._pieces(t, wt, False)[0], theta, mask)


def problem_for(kind: str, X: np.ndarray, data, backend: str, ridge: float = 1e-4):
    device = device_of(backend)
    return AdmissionProblem(X, data, device, ridge) if kind == "admission" else ChoiceProblem(X, data, device, ridge)


def fit(kind: str, X: np.ndarray, data, backend: str, weight=None, start=None, ridge: float = 1e-4):
    """One fit with the CPU fits' signature semantics (``funnel_models._fit``)."""
    problem = problem_for(kind, X, data, backend, ridge)
    if kind == "admission":
        return problem.fit(weight=weight, start=start)
    return problem.fit(set_weight=weight, start=start)


# ---------------------------------------------------------------- Monte Carlo (steelman generator), same random numbers

def sample_subsets(W: np.ndarray, L: np.ndarray, U: np.ndarray, device) -> torch.Tensor:
    """Torch twin of ``steelman.generator.sample_subsets`` (identical arithmetic; U supplied by the caller)."""
    Wt = torch.as_tensor(W, dtype=DTYPE, device=device)
    Wt = Wt / torch.clamp(Wt.max(dim=1, keepdim=True).values, min=1e-300)
    G, n = Wt.shape
    Ut = torch.as_tensor(U, dtype=DTYPE, device=device)
    M = Ut.shape[2]
    top = int(L.max()) if len(L) else 0
    S = torch.zeros(n + 1, G, top + 1, dtype=DTYPE, device=device)
    S[n, :, 0] = 1.0
    for j in range(n - 1, -1, -1):
        S[j] = S[j + 1]
        S[j, :, 1:] = S[j + 1, :, 1:] + Wt[:, j:j + 1] * S[j + 1, :, :-1]
    need = torch.as_tensor(np.repeat(np.asarray(L, np.int64)[:, None], M, axis=1), device=device)
    g = torch.arange(G, device=device)[:, None].expand(G, M)
    out = torch.zeros(G, n, M, dtype=torch.bool, device=device)
    for j in range(n):
        den = S[j][g, need]
        num = Wt[:, j][:, None] * S[j + 1][g, torch.clamp(need - 1, min=0)]
        prob = torch.where((need > 0) & (den > 0), num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(den))
        take = Ut[:, j, :] < prob
        out[:, j, :] = take
        need = need - take.to(need.dtype)
    return out


def expected_k(include: torch.Tensor, V: np.ndarray, u: np.ndarray, L: np.ndarray, gumbel: np.ndarray, device) -> np.ndarray:
    """Torch twin of ``steelman.generator.expected_k``."""
    from analysis.steelman.tables import rank_weights
    Vt = torch.as_tensor(V, dtype=DTYPE, device=device)
    ut = torch.as_tensor(u, dtype=DTYPE, device=device)
    Gm = torch.as_tensor(gumbel, dtype=DTYPE, device=device)
    score = torch.where(include, Vt[:, :, None] + Gm, torch.full_like(Gm, -torch.inf))
    order = torch.argsort(-score, dim=1, stable=True)
    u_sorted = torch.take_along_dim(ut[:, :, None].expand_as(score), order, dim=1)
    n = Vt.shape[1]
    w = torch.as_tensor(rank_weights(n), dtype=DTYPE, device=device)
    Lt = torch.as_tensor(np.asarray(L, np.int64), device=device)
    wmask = (torch.arange(n, device=device)[None, :] < Lt[:, None]).to(DTYPE) * w[None, :]
    num = torch.einsum("gn,gnm->gm", wmask, u_sorted)
    return (num / wmask.sum(dim=1, keepdim=True)).mean(dim=1).cpu().numpy()


# ---------------------------------------------------------------- two-way fixed effects, all bootstrap weights at once

def two_way_fe_batch(y: np.ndarray, X: np.ndarray, a: np.ndarray, r: np.ndarray, weights: np.ndarray, device,
                     tol: float = 1e-10, iters: int = 1000) -> np.ndarray:
    """Coefficients of the weighted two-way (a, r) fixed-effects regression of y on X for every weight vector in
    ``weights`` [B, n] at once (alternating projections as ``generator.demean_two_way``; zero weights drop rows)."""
    a = np.unique(a, return_inverse=True)[1]
    r = np.unique(r, return_inverse=True)[1]
    na, nr = int(a.max()) + 1, int(r.max()) + 1
    Wb = torch.as_tensor(np.asarray(weights, float), dtype=DTYPE, device=device)           # [B, n]
    V = torch.as_tensor(np.column_stack([y, X]), dtype=DTYPE, device=device)               # [n, q]
    B, (n, q) = Wb.shape[0], V.shape
    at = torch.as_tensor(a, device=device)
    rt = torch.as_tensor(r, device=device)
    out = V[None].expand(B, n, q).clone()                                                   # [B, n, q]
    wa = torch.zeros(B, na, dtype=DTYPE, device=device).index_add_(1, at, Wb).clamp(min=1e-300)
    wr = torch.zeros(B, nr, dtype=DTYPE, device=device).index_add_(1, rt, Wb).clamp(min=1e-300)
    active = torch.ones(B, dtype=torch.bool, device=device)
    for _ in range(iters):
        before = out.clone()
        wv = out * Wb[:, :, None]
        out -= (torch.zeros(B, na, q, dtype=DTYPE, device=device).index_add_(1, at, wv) / wa[:, :, None])[:, at]
        wv = out * Wb[:, :, None]
        out -= (torch.zeros(B, nr, q, dtype=DTYPE, device=device).index_add_(1, rt, wv) / wr[:, :, None])[:, rt]
        change = (out - before).abs().amax(dim=(1, 2))
        active = change >= tol
        if not bool(active.any()):
            break
    yt, Xt = out[:, :, 0], out[:, :, 1:]
    A = torch.einsum("bnp,bn,bnq->bpq", Xt, Wb, Xt)
    rhs = torch.einsum("bnp,bn,bn->bp", Xt, Wb, yt)
    # a singular system (e.g. a bootstrap draw without any row of one slot) fails only its own draw, as on the CPU
    coef, info = torch.linalg.solve_ex(A, rhs)
    coef[info != 0] = float("nan")
    return coef.cpu().numpy()
