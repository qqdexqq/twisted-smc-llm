"""The Stage 0 toy target: a 2-D Gaussian-mixture likelihood tempered against
a Gaussian reference, with a closed-form normalizing constant at beta=1.

    mu(theta)        = N(theta; 0, sigma0^2 I_2)                (reference; Z_0 = 1)
    L(theta)         = sum_k w_k * N(theta; mu_k, tau^2 I_2)     (unnormalized "likelihood")
    pi_lambda(theta) ~ mu(theta) * L(theta)^lambda               (= mu * exp(-lambda*V), V = -log L)

Z_1 = integral of mu(theta)*L(theta) has a closed form via the Gaussian
product/convolution identity:

    Z_1 = sum_k w_k * c_k,   c_k = N(mu_k; 0, (sigma0^2 + tau^2) I_2)

Parameters below (sigma0^2=4.0, tau^2=0.25, 4 components ~12 std-devs apart,
asymmetric weights, an off-square mean) are chosen so the target is genuinely
multimodal and so that the component with the *largest* raw mixture weight
does NOT dominate Z_1 once the reference-density tilt is applied -- a
sampler with a subtly wrong reweighting formula is likely to get the
per-mode mass ratio wrong even if it visually "finds" every mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

D = 2


@dataclass(frozen=True)
class GMM2DTarget:
    sigma0_sq: float = 4.0
    tau_sq: float = 0.25
    means: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [4.0, 4.0]]
        )
    )
    weights: np.ndarray = field(
        default_factory=lambda: np.array([0.15, 0.20, 0.25, 0.40])
    )

    def __post_init__(self):
        assert self.means.shape == (len(self.weights), D)
        assert abs(self.weights.sum() - 1.0) < 1e-12

    # ---- reference measure mu(theta), Z_0 = 1 -----------------------------

    def log_mu(self, theta: np.ndarray) -> np.ndarray:
        """log N(theta; 0, sigma0^2 I_2), theta shape (N, 2)."""
        sq_norm = np.sum(theta**2, axis=-1)
        return -np.log(2 * np.pi * self.sigma0_sq) - sq_norm / (2 * self.sigma0_sq)

    def sample_reference(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact i.i.d. samples from mu (no burn-in needed at beta=0)."""
        return rng.normal(0.0, np.sqrt(self.sigma0_sq), size=(n, D))

    # ---- unnormalized "likelihood" L(theta) = exp(-V(theta)) --------------

    def log_L(self, theta: np.ndarray) -> np.ndarray:
        """log L(theta) = logsumexp_k [ log w_k + log N(theta; mu_k, tau^2 I) ].

        theta: (N, 2). Returns (N,). Always via logsumexp -- never raw
        log(sum(exp(...))) -- per the doc's numerical-hygiene rule.
        """
        diff = theta[:, None, :] - self.means[None, :, :]  # (N, K, 2)
        sq_norm = np.sum(diff**2, axis=-1)  # (N, K)
        log_comp = -np.log(2 * np.pi * self.tau_sq) - sq_norm / (2 * self.tau_sq)
        log_w = np.log(self.weights)
        return logsumexp(log_w[None, :] + log_comp, axis=-1)

    def log_G(self, theta: np.ndarray) -> np.ndarray:
        """log_G_i = -V(theta_i) = log L(theta_i); the rate solve_beta_step needs."""
        return self.log_L(theta)

    # ---- closed-form Z_1 ----------------------------------------------------

    @property
    def log_Z1(self) -> float:
        """log Z_1 = log sum_k w_k * c_k, c_k = N(mu_k; 0, (sigma0^2+tau^2) I_2)."""
        s = self.sigma0_sq + self.tau_sq
        sq_norm = np.sum(self.means**2, axis=-1)
        log_c = -np.log(2 * np.pi * s) - sq_norm / (2 * s)
        log_w = np.log(self.weights)
        return float(logsumexp(log_w + log_c))

    @property
    def Z1(self) -> float:
        return float(np.exp(self.log_Z1))

    def per_mode_mass_fraction(self) -> np.ndarray:
        """w_k * c_k / Z_1 for each mode -- the actual contribution to Z_1,
        as opposed to the raw mixture weight w_k. Useful for sanity-checking
        that a sampler recovers per-mode mass, not just mode *locations*.
        """
        s = self.sigma0_sq + self.tau_sq
        sq_norm = np.sum(self.means**2, axis=-1)
        log_c = -np.log(2 * np.pi * s) - sq_norm / (2 * s)
        log_w = np.log(self.weights)
        log_terms = log_w + log_c
        return np.exp(log_terms - logsumexp(log_terms))


DEFAULT_TARGET = GMM2DTarget()
