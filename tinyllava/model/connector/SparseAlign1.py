# gcs_sparse_align_fp16.py
# FP16-friendly version of GCS sparse align module with mixed precision support.
# Key ideas:
# - compute numerically sensitive ops (pairwise similarity, topk, heat kernel, sinkhorn core) in fp32
# - allow other matrix ops and linear layers to run in fp16 via autocast
# - use GradScaler for stable training

import argparse
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------- Helpers -----------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def topk_cosine_similarity(A: torch.Tensor, B: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return row-wise top-k values & indices for A @ B^T.
    A: [m,d], B: [n,d]  (should be normalized)
    Performs core computation in FP32 for stability.
    """
    # compute in float32
    A32 = A.float()
    B32 = B.float()
    sim = A32 @ B32.t()  # [m, n] in float32
    vals, idx = torch.topk(sim, k=min(k, sim.size(1)), dim=1)
    return vals, idx


def build_knn_adj(Z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build symmetric kNN adjacency matrix from Z.
    Compute pairwise sim in FP32 to avoid fp16 overflow/precision issues.
    Returns A in same device and same dtype as Z (but adjacency values are 0/1 in Z.dtype).
    """
    N, d = Z.shape
    Zn = F.normalize(Z, p=2, dim=1)

    # compute similarity in float32 for stability
    S32 = Zn.float() @ Zn.float().t()  # [N,N]
    # mask diagonal (self) by -inf so topk excludes self
    S32.fill_diagonal_(float("-inf"))

    kk = min(k, max(N - 1, 1))
    vals, idx = torch.topk(S32, k=kk, dim=1)  # [N, kk], [N, kk]

    A = torch.zeros((N, N), device=Z.device, dtype=Z.dtype)
    rows = torch.arange(N, device=Z.device).unsqueeze(1).expand_as(idx)
    A[rows, idx] = 1.0
    # symmetrize
    A = torch.maximum(A, A.t())
    return A


def laplacian_from_adj(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1)
    D = torch.diag(deg)
    return D - A


def heat_kernel_approx(L: torch.Tensor, tau: float = 0.1, terms: int = 6) -> torch.Tensor:
    """
    Approximate exp(-tau * L) using truncated series.
    Execute in FP32 for numerical stability.
    """
    # cast to float32
    L32 = L.float()
    n = L32.size(0)
    I = torch.eye(n, device=L32.device, dtype=L32.dtype)
    K = torch.zeros_like(L32)
    term = I.clone()
    for m in range(0, terms):
        if m == 0:
            K = K + term
        else:
            term = (-tau / m) * (term @ L32)
            K = K + term
    return K.to(L.dtype)  # cast back to original dtype


def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, eps: float = 0.05, max_iter: int = 80) -> torch.Tensor:
    """
    Stable Sinkhorn, computing in log-domain / float32 for stability.
    a: [m], b: [n], C: [m,n] (float)
    Returns P [m,n] in same device (float dtype).
    """
    # Ensure float32
    C32 = C.float()
    m, n = C32.shape
    K = torch.exp(-C32 / eps)  # [m,n] float32
    # initialize u/v in float32
    u = torch.ones(m, device=C.device, dtype=torch.float32)
    v = torch.ones(n, device=C.device, dtype=torch.float32)
    a32 = (a.float() / (a.sum().float() + 1e-12))
    b32 = (b.float() / (b.sum().float() + 1e-12))
    for _ in range(max_iter):
        u = a32 / (K @ v + 1e-12)
        v = b32 / (K.t() @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)  # float32
    return P


# ----------------------- Model Components -----------------------

class GCSModule(nn.Module):
    """
    Compute Graph Cosine Similarity (GCS) and produce sparse alignment P.
    FP16-friendly: sensitive ops (pairwise sim, topk, kernel) computed in FP32.
    """

    def __init__(self, in_dim: int = 128, proj_dim: int = 128, k_intra: int = 16, topo_mode: str = 'mean'):
        super().__init__()
        self.k_intra = k_intra
        assert topo_mode in ('mean', 'heat')
        self.topo_mode = topo_mode
        # projection for both modalities (we assume input dims possibly equal to proj_dim)
        self.proj_x = nn.Linear(in_dim, proj_dim, bias=False)
        self.proj_t = nn.Linear(in_dim, proj_dim, bias=False)

        # scalar weights (kept in fp32 internally via parameters)
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))

    def forward(self, X: torch.Tensor, T: torch.Tensor,
                use_sinkhorn: bool = False, sinkhorn_eps: float = 0.05,
                topk_align: int = 8, coarse_candidate: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X: [B, D_in], T: [L, D_in]
        returns: GCS [B,L] (float dtype, on device), P [B,L] (same dtype), Ax [B,B] adjacency (same dtype as X)
        """

        device = X.device
        # Project & normalize: do projection inside autocast (can be fp16) for speed
        with torch.cuda.amp.autocast(enabled=True):
            Zx = l2norm(self.proj_x(X))  # may be fp16 or fp32 depending on autocast & model dtype
            Zt = l2norm(self.proj_t(T))

        # Build intra-graph adjacency in stable FP32 (build_knn_adj casts internally)
        Ax = build_knn_adj(Zx, k=self.k_intra)  # [B,B] dtype matches Zx dtype
        At = build_knn_adj(Zt, k=self.k_intra)

        # Laplacians or heat kernels (compute kernels in float32 for stability)
        Lx = laplacian_from_adj(Ax)
        Lt = laplacian_from_adj(At)
        if self.topo_mode == 'heat':
            Kx = heat_kernel_approx(Lx, tau=0.1, terms=6)
            Kt = heat_kernel_approx(Lt, tau=0.1, terms=6)
        else:
            Kx = Ax
            Kt = At

        # Base cosine similarity: compute in FP32 to avoid fp16 precision loss
        Zx32 = Zx.float()
        Zt32 = Zt.float()
        base = (Zx32 @ Zt32.t()).clamp(min=0.0)  # [B, L] float32

        # density correction (deg inverse), compute in float32
        deg_x = Ax.sum(dim=1).clamp(min=1.0).float()  # [B]
        deg_t = At.sum(dim=1).clamp(min=1.0).float()  # [L]
        cx = (1.0 / deg_x).unsqueeze(1)  # [B,1]
        ct = (1.0 / deg_t).unsqueeze(0)  # [1,L]
        density_correction = (cx * ct).clamp(min=1e-12)  # [B,L] float32

        # Topological consistency:
        if self.topo_mode == 'mean':
            # Nx_idx: neighborhood indicator in same dtype as Ax (0/1), but use float32 multiply with Zx32
            Nx_idx = (Ax > 0).to(torch.float32)  # [B,B]
            denom_x = Nx_idx.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
            # aggregate neighbors in float32
            z_x_bar = (Nx_idx @ Zx32) / denom_x  # [B, d]
            Nt_idx = (At > 0).to(torch.float32)
            denom_t = Nt_idx.sum(dim=1, keepdim=True).clamp(min=1.0)
            z_t_bar = (Nt_idx @ Zt32) / denom_t  # [L, d]
        else:
            # heat kernel aggregation (Kx/Kt are float types; ensure float32)
            z_x_bar = l2norm((Kx.float() @ Zx32))
            z_t_bar = l2norm((Kt.float() @ Zt32))

        topo = (z_x_bar @ z_t_bar.t()).clamp(min=0.0)  # [B,L] float32

        # Combine into GCS (all in float32)
        # Note: alpha,beta,gamma are registered as float32 parameters
        GCS = (base ** (self.alpha.float())) * (density_correction ** (self.beta.float())) * (topo ** (self.gamma.float()) + 1e-12)
        # GCS is float32 tensor

        # Sparsify to get P
        if use_sinkhorn:
            C = -GCS  # cost (float32)
            a = torch.ones(Zx32.size(0), device=device, dtype=torch.float32) / Zx32.size(0)
            b = torch.ones(Zt32.size(0), device=device, dtype=torch.float32) / Zt32.size(0)
            P = sinkhorn(a, b, C, eps=sinkhorn_eps, max_iter=80)  # P is float32
            # If user expects P in same dtype as GCS, keep float32 (stable)
        else:
            B = GCS.size(0)
            L = GCS.size(1)
            # row top-k
            if coarse_candidate is not None and coarse_candidate < L:
                # coarse candidate from inner product (do in fp32)
                vals_coarse, idx_coarse = topk_cosine_similarity(Zx32, Zt32, k=coarse_candidate)  # [B,M]
                vals, idx = torch.topk(vals_coarse, k=min(topk_align, vals_coarse.size(1)), dim=1)
                actual_idx = idx_coarse.gather(1, idx)  # [B, topk_align]
                P = torch.zeros_like(GCS, dtype=torch.float32)
                P_vals = torch.softmax(vals / 0.07, dim=1)  # float32
                rows = torch.arange(B, device=device).unsqueeze(1).expand(-1, actual_idx.size(1))
                P[rows, actual_idx] = P_vals
            else:
                # top-k on GCS (GCS is float32)
                vals, idx = torch.topk(GCS, k=min(topk_align, L), dim=1)
                P = torch.zeros_like(GCS, dtype=torch.float32)
                P_vals = torch.softmax(vals / 0.07, dim=1)  # float32
                rows = torch.arange(GCS.size(0), device=device).unsqueeze(1).expand(-1, idx.size(1))
                P[rows, idx] = P_vals

        return GCS, P, Ax


class BipartiteGNNLayer(nn.Module):
    """
    Bipartite layer. We allow linear layers to run in autocast (FP16) for speed.
    But when multiplying with P which is float32, cast P to match Ht dtype for matmul.
    """

    def __init__(self, d: int):
        super().__init__()
        self.W_x = nn.Linear(d, d, bias=False)
        self.W_t = nn.Linear(d, d, bias=False)
        self.act = nn.ReLU()

    def forward(self, Hx: torch.Tensor, Ht: torch.Tensor, P: torch.Tensor, update_text: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Hx: [B,d] (may be fp16 in autocast), Ht: [L,d], P: [B,L] float32
        # We'll perform P @ (W_t(Ht)) in float32 for stability but allow linear transforms in autocast
        device = Hx.device
        # compute self transforms in autocast
        with torch.cuda.amp.autocast(enabled=True):
            hx_self = self.W_x(Hx)  # may be fp16
            ht_trans = self.W_t(Ht)  # may be fp16
        # ensure ht_trans is float32 for matmul with P (P is float32)
        ht32 = ht_trans.float()
        xt_msg32 = P @ ht32  # [B, d] float32
        # cast xt_msg back to Hx dtype for addition
        xt_msg = xt_msg32.to(hx_self.dtype)

        Hx_new = self.act(hx_self + xt_msg)  # result in autocast dtype
        Ht_new = None
        if update_text:
            # update text side (less commonly used in our pipeline)
            with torch.cuda.amp.autocast(enabled=True):
                ht_self = self.W_t(Ht)
            tx_msg32 = P.t() @ hx_self.float()
            Ht_new = self.act(ht_self + tx_msg32.to(ht_self.dtype))
        return Hx_new, Ht_new


# ----------------------- Selector (vectorized NMS) -----------------------

def select_key_nodes(Hx: torch.Tensor, P: torch.Tensor, Ax: torch.Tensor,
                     K_ratio: float = 0.1, lambda_cent: float = 0.2, nms_thresh: float = 0.32) -> torch.Tensor:
    """
    Vectorized selection of key nodes.
    Hx: [B, d] (may be fp16)
    P: [B, L] float32
    Ax: [B, B] adjacency (0/1 in original dtype)
    Returns: keep_idx tensor (K,)
    """
    B = Hx.size(0)
    K = max(1, int(B * K_ratio))

    # compute score in stable dtype (float32)
    align_strength = P.max(dim=1).values  # [B] float32
    cent = Ax.sum(dim=1).float()  # [B] float32
    # normalize cent
    cent = (cent - cent.min()) / (cent.max() - cent.min() + 1e-12)
    score = align_strength + lambda_cent * cent  # [B] float32

    # pick top candidates (vectorized)
    num_candidates = min(B, K * 4)
    _, top_idx = torch.topk(score, k=num_candidates)  # [num_candidates]
    # gather feature vectors (cast to float32 for stable cosine)
    H_top = Hx[top_idx].float()  # [N, d] float32

    # pairwise cosine similarity (float32)
    # Use normalized vectors for cosine
    H_top_n = F.normalize(H_top, p=2, dim=1)
    sim_matrix = H_top_n @ H_top_n.t()  # [N, N] float32

    # keep only upper triangle comparisons (i < j)
    mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
    sim_upper = torch.where(mask, sim_matrix, torch.zeros_like(sim_matrix))

    # redundant if any previous (i < j) is > threshold => mark j as redundant
    redundant = (sim_upper > nms_thresh).any(dim=0)  # [N] bool
    keep_mask = ~redundant
    keep_idx = top_idx[keep_mask]  # indices into original Hx

    # ensure at least K selected; if not, fill from highest scores
    if keep_idx.numel() < K:
        extra = torch.topk(score, k=K).indices  # top-K by score
        combined = torch.cat([keep_idx, extra])
        # unique while preserving order: use torch.unique_consecutive after sorting by original order
        # simpler: use torch.unique (not guaranteed order), then slice
        uniq = torch.unique(combined)
        # if still less than K, pad by top entries
        if uniq.numel() < K:
            topK = torch.topk(score, k=K).indices
            uniq = torch.unique(torch.cat([uniq, topK]))
        keep_idx = uniq[:K]
    else:
        keep_idx = keep_idx[:K]

    return keep_idx.to(Hx.device)


# ----------------------- Losses -----------------------

def info_nce_loss(Hx: torch.Tensor, Ht: torch.Tensor, P: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Hx: [B, d] (autocast dtype)
    Ht: [L, d] (autocast dtype)
    P:  [B, L] float32
    Compute cross-entropy based on positive indices from P (float32).
    """
    device = Hx.device
    # normalize in autocast to preserve dtype
    Hx_n = l2norm(Hx)
    Ht_n = l2norm(Ht)

    pos_idx = P.argmax(dim=1).to(torch.long)  # [B]
    # compute logits in float32 for numerical stability
    logits = (Hx_n.float() @ Ht_n.float().t()) / float(temperature)  # [B,L] float32
    labels = pos_idx.to(device)
    loss = F.cross_entropy(logits, labels)
    return loss


def laplacian_alignment_loss(P: torch.Tensor, Ax: torch.Tensor, At: torch.Tensor) -> torch.Tensor:
    """
    L_x' = P^T L_x P  should approximate L_t
    Compute in float32.
    """
    Lx = laplacian_from_adj(Ax).float()
    Lt = laplacian_from_adj(At).float()
    P32 = P.float()
    PtpLp = P32.t() @ (Lx @ P32)
    loss = F.mse_loss(PtpLp, Lt)
    return loss


# ----------------------- Main wrapper -----------------------

class GCSAligner(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 128, k_intra: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.prepare_x = nn.Linear(input_dim, proj_dim) if input_dim != proj_dim else nn.Identity()
        self.prepare_t = nn.Linear(input_dim, proj_dim) if input_dim != proj_dim else nn.Identity()

        self.gcs = GCSModule(in_dim=input_dim, proj_dim=proj_dim, k_intra=k_intra, topo_mode='mean')
        self.gnn = BipartiteGNNLayer(d=proj_dim)

    def forward(self, X: torch.Tensor, T: torch.Tensor, use_sinkhorn: bool = False, topk_align: int = 8,
                coarse_candidate: Optional[int] = None) -> dict:
        # Project inputs (do inside autocast to allow fp16)
        with torch.cuda.amp.autocast(enabled=True):
            Xp = l2norm(self.prepare_x(X))
            Tp = l2norm(self.prepare_t(T))
        # GCS returns float32 tensors for GCS and P, and Ax (adjacency) in original dtype
        GCS, P, Ax = self.gcs(Xp, Tp, use_sinkhorn=use_sinkhorn, topk_align=topk_align, coarse_candidate=coarse_candidate)
        # Bipartite propagation: Hx will be in autocast dtype
        Hx, _ = self.gnn(Xp, Tp, P, update_text=False)
        key_idx = select_key_nodes(Hx, P, Ax, K_ratio=0.1)
        out = {
            'GCS': GCS,  # [B,L] float32
            'P': P,      # [B,L] float32
            'Ax': Ax,    # [B,B] adjacency
            'Hx': Hx,    # [B,d] autocast dtype
            'Tp': Tp,    # [L,d] autocast dtype
            'key_idx': key_idx
        }
        return out


# ----------------------- Training / Demo -----------------------

# def demo_train(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device:", device)
#
#     B = args.batch_size
#     L = args.L
#     input_dim = args.input_dim
#
#     # Synthetic data (replace with your encoder outputs)
#     X = torch.randn(B, input_dim, device=device)
#     T = torch.randn(L, input_dim, device=device)
#
#     model = GCSAligner(input_dim=input_dim, proj_dim=args.d, k_intra=args.k_intra).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scaler = torch.cuda.amp.GradScaler(init_scale=2.**10)
#
#     for epoch in range(args.epochs):
#         model.train()
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast(enabled=True):
#             out = model(X, T, use_sinkhorn=False, topk_align=args.topk_align, coarse_candidate=args.coarse_candidate)
#             GCS = out['GCS']  # float32
#             P = out['P']      # float32
#             Ax = out['Ax']
#             Hx = out['Hx']    # autocast dtype
#             Tp = out['Tp']
#
#             loss_nce = info_nce_loss(Hx, Tp, P, temperature=args.temp)
#             # For Lt in laplacian loss, we approximate At from Tp similarities for demo
#             At_approx = (Tp.float() @ Tp.float().t() > 0).float()
#             loss_lap = laplacian_alignment_loss(P, Ax, At_approx)
#             loss_sparsity = P.abs().sum() * args.lam_spars
#
#             loss = loss_nce + args.lam_lap * loss_lap + loss_sparsity
#
#         # backward with GradScaler
#         scaler.scale(loss).backward()
#         # optional gradient clipping
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#
#         if (epoch + 1) % args.log_every == 0:
#             with torch.no_grad():
#                 key_idx = out['key_idx']
#                 print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.6f} | NCE: {loss_nce.item():.6f} | Lap: {loss_lap.item():.6f} | #key: {len(key_idx)}")
#
#     # inference demo
#     model.eval()
#     with torch.no_grad():
#         out = model(X, T, use_sinkhorn=False, topk_align=args.topk_align)
#         print("Selected indices:", out['key_idx'].cpu().tolist())
#
#
# # ----------------------- CLI -----------------------
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--L', type=int, default=512)
#     parser.add_argument('--input_dim', type=int, default=256)
#     parser.add_argument('--d', type=int, default=128)
#     parser.add_argument('--k_intra', type=int, default=12)
#     parser.add_argument('--topk_align', type=int, default=8)
#     parser.add_argument('--coarse_candidate', type=int, default=None)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--temp', type=float, default=0.07)
#     parser.add_argument('--lam_lap', type=float, default=0.1)
#     parser.add_argument('--lam_spars', type=float, default=1e-4)
#     parser.add_argument('--log_every', type=int, default=1)
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     demo_train(args)