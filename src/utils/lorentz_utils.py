"""
Lorentzian (Hyperbolic) Geometry Utilities.

All operations on the Lorentz hyperboloid model H^n:
    H^n = { x in R^{n+1} : <x, x>_L = -1,  x_0 > 0 }

Lorentzian inner product:
    <x, y>_L = -x_0 * y_0 + sum_{i=1}^{n} x_i * y_i

Geodesic distance:
    d_L(x, y) = arcosh( -<x, y>_L )

Reference: Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018.
           Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Numerical stability clip
# ─────────────────────────────────────────────────────────────────────────────
_EPS   = 1e-8
_CLAMP = 1 - 1e-5   # for arcosh: argument must be >= 1


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Lorentzian inner product <x, y>_L = -x0*y0 + x1*y1 + ... + xn*yn.
    x, y : (..., n+1)
    returns: (...,)
    """
    # flip sign of time component
    sign = torch.ones_like(x)
    sign[..., 0] = -1.0
    return (sign * x * y).sum(dim=-1)


def lorentz_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Lorentzian norm sqrt(<x,x>_L).  For points on H^n this equals 1.
    Used for intermediate tensors (tangent vectors).
    """
    return torch.clamp(lorentz_inner(x, x), min=_EPS).sqrt()


def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """
    Project a Euclidean vector x = (x_1,...,x_n) onto H^n by computing
    the time component x_0 = sqrt(1 + ||x_{1:n}||^2).

    Input:  x of shape (..., n)   — spatial components only
    Output: y of shape (..., n+1) — full hyperboloid point
    """
    x0 = torch.sqrt(1.0 + (x ** 2).sum(dim=-1, keepdim=True) + _EPS)
    return torch.cat([x0, x], dim=-1)


def lorentz_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on H^n.
    d(x, y) = arcosh( -<x, y>_L )

    x, y : (..., n+1)  — points on the hyperboloid
    returns: (...,)
    """
    inner = -lorentz_inner(x, y)                      # should be >= 1
    inner = torch.clamp(inner, min=1.0 + _EPS)        # numerical safety
    return torch.acosh(inner)


def exp_map_origin(v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map at the origin o = (1, 0, ..., 0) of H^n.

    Maps a tangent vector v (with v_0 = 0) to a point on H^n:
        exp_o(v) = cosh(||v||) * o + sinh(||v||)/||v|| * v

    v : (..., n+1)  with v[..., 0] == 0
    returns: (..., n+1)  — point on H^n
    """
    v_norm = torch.clamp(lorentz_norm(v), min=_EPS)
    coeff  = torch.sinh(v_norm) / v_norm              # (...,)
    origin = torch.zeros_like(v)
    origin[..., 0] = 1.0
    return torch.cosh(v_norm).unsqueeze(-1) * origin + coeff.unsqueeze(-1) * v


def log_map_origin(x: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map at the origin: inverse of exp_map_origin.

    log_o(x) = d(o, x) / sinh(d(o, x)) * (x - cosh(d(o,x)) * o)

    x : (..., n+1)  — point on H^n
    returns: (..., n+1)  — tangent vector at o
    """
    origin     = torch.zeros_like(x)
    origin[..., 0] = 1.0
    d          = lorentz_dist(x, origin).unsqueeze(-1)         # (..., 1)
    coeff      = d / torch.clamp(torch.sinh(d), min=_EPS)
    return coeff * (x - torch.cosh(d) * origin)


def lorentz_centroid(x: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Fréchet mean on H^n — approximated via weighted mean in ambient space
    followed by re-projection onto the hyperboloid.

    x       : (N, n+1)
    weights : (N,) optional
    returns : (n+1)
    """
    if weights is None:
        weights = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    mean_ambient = (weights.unsqueeze(-1) * x).sum(dim=0)
    # Re-project: set x0 = sqrt(1 + ||x_{1:}||^2)
    spatial = mean_ambient[1:]
    x0 = torch.sqrt(1.0 + (spatial ** 2).sum() + _EPS)
    return torch.cat([x0.unsqueeze(0), spatial], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Lorentzian Linear Layer
# ─────────────────────────────────────────────────────────────────────────────

class LorentzLinear(nn.Module):
    """
    Linear transformation in the Lorentzian tangent space at the origin.

    Steps:
        1. log_map to move from H^n to tangent space T_o H^n
        2. Standard Euclidean linear transform
        3. exp_map back to H^{m}
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        in_features / out_features refer to the SPATIAL dimension n
        (i.e. the dimension of H^n, NOT the ambient n+1).
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, n+1) on H^n  →  (B, m+1) on H^m"""
        tangent  = log_map_origin(x)           # (B, n+1)
        tangent  = tangent[..., 1:]            # drop time component → (B, n)
        out_tan  = self.linear(tangent)        # (B, m)  [linear: n→m]
        # pad time component as zero → exp_map will compute correct x0
        out_full = torch.cat([torch.zeros_like(out_tan[..., :1]), out_tan], dim=-1)
        return exp_map_origin(out_full)        # (B, m+1)


class LorentzMLR(nn.Module):
    """
    Multinomial Logistic Regression on the Lorentz manifold.

    Computes class scores as Lorentzian distances to learned class prototypes.
    Each prototype p_k lives on H^n.

    Reference: Gao et al., "Curvature Generation in Curved Spaces for Few-Shot
               Learning", ICCV 2021.
    """

    def __init__(self, manifold_dim: int, num_classes: int):
        """
        manifold_dim : spatial dimension n (NOT ambient n+1).
                       Prototypes are stored as n-dim spatial vectors and
                       projected onto H^n via project_to_hyperboloid.
        """
        super().__init__()
        # Prototypes stored as spatial components (n,); time computed on forward
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, manifold_dim) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, n+1)  — points on H^n
        returns : (B, K)    — logits
        """
        # Project prototypes onto hyperboloid
        protos = project_to_hyperboloid(self.prototypes)  # (K, n+1)
        # Pairwise distances: (B, K)
        B = x.shape[0]
        K = protos.shape[0]
        x_exp    = x.unsqueeze(1).expand(B, K, -1)        # (B, K, n+1)
        p_exp    = protos.unsqueeze(0).expand(B, K, -1)   # (B, K, n+1)
        dists    = lorentz_dist(x_exp, p_exp)              # (B, K)
        return -dists                                       # negative dist = logit
