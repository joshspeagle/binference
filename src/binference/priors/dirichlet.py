"""Conjugate priors on the simplex: Dirichlet, Dirichlet-tree, generalised Dirichlet, Polya tree.

All of these remain conjugate given latent bin counts, so Gibbs sampling,
EM/MAP and mean-field VB are closed-form. Gradient-based engines use the
isometric log-ratio (ILR) parametrisation, in which the log-density of a
Dirichlet-tree is linear in the log masses of leaves and internal nodes:

    log p(rho) = const + sum_k (alpha_pa(k),k - 1) log rho_k
                       + sum_{c != root} (alpha_pa(c),c - A_c) log rho_c ,

with ``A_c`` the total concentration of node ``c``'s children.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import special

from .._simplex import helmert_basis, log_softmax, softmax
from .._utils import check_positive
from ..bins import Bins
from ..density import Density
from .base import (
    _TINY,
    BoundPrior,
    PopulationPrior,
    _log_gamma_variates,
    _per_item,
    _stick_break,
)

__all__ = ["Dirichlet", "DirichletTree", "GeneralizedDirichlet", "PolyaTree"]


def _dirichlet_draw(alpha: np.ndarray, rng: np.random.Generator, size: int | None = None):
    """Dirichlet draws without exact zeros, even for tiny concentrations (floored at 1e-300)."""
    alpha = np.asarray(alpha, dtype=float)
    shape = alpha.shape if size is None else (int(size), *alpha.shape)
    return np.maximum(softmax(_log_gamma_variates(alpha, rng, shape)), _TINY)


class _SimplexBound(BoundPrior):
    """Shared ILR machinery for simplex priors."""

    has_density = True
    has_prior_transform = True
    conjugate = True

    def __init__(self, prior: PopulationPrior, bins: Bins) -> None:
        super().__init__(prior, bins)
        self.dim = self.K - 1
        self.V = helmert_basis(self.K)
        self._log_jac_const = 0.5 * np.log(self.K)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, Any]:
        eta = self.V @ x
        return eta, eta

    def log_prior_and_grad(self, x, cache, value, d_eta):
        log_rho = log_softmax(cache)
        lp, g_eta = self._lp_eta(value, log_rho)
        return lp + self._log_jac_const, self.V.T @ (d_eta + g_eta)

    def init_from_value(self, value: np.ndarray) -> np.ndarray:
        r = np.asarray(value, dtype=float)
        r = np.maximum(r, 1e-12 * np.max(r))
        r = r / r.sum()
        return self.V.T @ np.log(r)

    def _lp_eta(self, rho, log_rho):  # pragma: no cover - abstract
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Dirichlet
# ----------------------------------------------------------------------------
class Dirichlet(PopulationPrior):
    r"""Dirichlet prior :math:`\rho \sim \mathrm{Dir}(\alpha)` on bin fractions.

    The default prior of binference (``alpha=1``, uniform on the simplex).

    Parameters
    ----------
    alpha : float, array_like (K,), or "reference"
        Concentration parameters (strictly positive). ``"reference"`` uses
        ``alpha = 1/K`` (Berger, Bernardo & Sun 2015), which avoids swamping
        sparse problems with many bins.

    Notes
    -----
    A symmetric ``alpha`` over bins of unequal volume is *not* consistent under
    merging bins; :meth:`from_base_measure` makes ``alpha`` proportional to a
    base measure (flat in the measure coordinate by default) instead.
    """

    def __init__(self, alpha: float | Sequence[float] | np.ndarray | str = 1.0) -> None:
        if isinstance(alpha, str):
            if alpha != "reference":
                raise ValueError("alpha must be positive numbers or 'reference'.")
        else:
            check_positive(alpha, "Dirichlet alpha")
        self.alpha = alpha
        self._base: tuple[float, Any] | None = None

    def _params(self):
        if self._base is not None:
            return {"concentration": self._base[0], "base": self._base[1]}
        return {"alpha": self.alpha}

    @classmethod
    def reference(cls, K: int | None = None) -> Dirichlet:
        """Reference prior ``alpha = 1/K``.

        Parameters
        ----------
        K : int, optional
            Number of bins; ``None`` takes it from the bins when the prior is bound.
        """
        return cls("reference") if K is None else cls(1.0 / K)

    @classmethod
    def from_counts(
        cls, counts: Sequence[float] | np.ndarray, strength: float = 1.0, base: float = 1.0
    ) -> Dirichlet:
        """Prior from a reference sample: ``alpha = base + strength * counts``.

        ``strength < 1`` down-weights a reference sample that may not be fully
        representative (for a coherent model of an unrepresentative reference
        use a reference-centred :class:`~binference.priors.LogisticNormal` or
        :class:`~binference.priors.LogisticRandomWalk` with ``mean=log(reference)``).

        Parameters
        ----------
        counts : array_like (K,)
            Counts of the reference sample in each bin.
        strength : float, optional
            Weight of the reference counts in ``alpha``.
        base : float, optional
            Concentration added to every bin.
        """
        c = np.asarray(counts, dtype=float)
        if np.any(c < 0):
            raise ValueError("counts must be non-negative.")
        return cls(base + strength * c)

    @classmethod
    def from_base_measure(cls, concentration: float = 1.0, base: Any = "flat") -> Dirichlet:
        """``alpha_k = concentration * B_k``, with ``B`` the normalised bin masses of ``base``.

        ``base`` is a :class:`~binference.Density` (or anything
        :meth:`Density.coerce <binference.Density.coerce>` accepts; an array is
        read as masses per bin). The default is flat in the measure coordinate,
        proportional to bin volumes on linear bins.

        Parameters
        ----------
        concentration : float, optional
            Total concentration, spread over the bins by the base measure.
        base : Density, str, callable or array_like, optional
            Base measure whose bin masses shape ``alpha``.
        """
        check_positive(concentration, "concentration")
        obj = cls(1.0)
        obj.alpha = None  # set at binding, from the base measure
        obj._base = (float(concentration), Density.coerce(base, "mass", name="base"))
        return obj

    def resolve_alpha(self, bins: Bins) -> np.ndarray:
        K = bins.K
        if self._base is not None:
            conc, base = self._base
            b = base.masses(bins)
            if np.any(b <= 0):
                raise ValueError("The base measure must give every bin positive mass.")
            return conc * b
        if isinstance(self.alpha, str):
            return np.full(K, 1.0 / K)
        return _per_item(self.alpha, K, "alpha").astype(float)

    def bind(self, bins: Bins) -> _BoundDirichlet:
        return _BoundDirichlet(self, bins, self.resolve_alpha(bins))


class _BoundDirichlet(_SimplexBound):
    def __init__(self, prior: PopulationPrior, bins: Bins, alpha: np.ndarray) -> None:
        super().__init__(prior, bins)
        self.alpha = alpha
        self.pseudocounts = alpha - 1.0
        self.A = float(alpha.sum())
        self._const = float(special.gammaln(self.A) - special.gammaln(alpha).sum())

    def _lp_eta(self, rho, log_rho):
        lp = float(np.dot(self.alpha, log_rho)) + self._const
        return lp, self.alpha - self.A * rho

    def sample(self, n, rng):
        return _dirichlet_draw(self.alpha, rng, size=n)

    def log_density(self, value):
        r = np.asarray(value, dtype=float)
        return np.sum(special.xlogy(self.alpha - 1.0, r), axis=-1) + self._const

    def grad_log_density(self, value):
        v = np.asarray(value, dtype=float)
        pos = v > 0
        return np.where(pos, (self.alpha - 1.0) / np.where(pos, v, 1.0), 0.0)

    def prior_transform(self, u):
        return _stick_break(self.alpha, np.asarray(u, dtype=float))

    def posterior_sample(self, counts, rng, exposure=None):
        return _dirichlet_draw(self.alpha + counts, rng)

    def unbounded_density(self):
        return bool(np.any(self.alpha < 1.0))

    def map_from_counts(self, counts, exposure=None):
        x = np.maximum(np.asarray(counts, float) + self.alpha - 1.0, 0.0)
        tot = x.sum()
        return x / tot if tot > 0 else np.full(self.K, 1.0 / self.K)

    def vb_update(self, counts, exposure=None):
        return self.alpha + counts

    def vb_expected_log(self, state):
        return special.digamma(state) - special.digamma(state.sum())

    def vb_sample(self, state, n, rng):
        return _dirichlet_draw(state, rng, size=n)

    def vb_mean(self, state):
        return state / state.sum()


# ----------------------------------------------------------------------------
# Dirichlet-tree family
# ----------------------------------------------------------------------------
class _TreeStructure:
    """Flattened representation of a Dirichlet-tree over ``K`` leaves.

    ``nodes[c]`` is a list of children, each ``("leaf", k)`` or ``("node", c')``;
    node 0 is the root and parents precede children.
    """

    def __init__(self, nodes: list[list[tuple[str, int]]], alphas: list[np.ndarray], K: int):
        # the priors validate the concentrations; the leaves need K
        C = len(nodes)
        seen_leaves: list[int] = []
        parent_slot_of_node = np.full(C, -1, dtype=np.int64)
        leaf_slot = np.full(K, -1, dtype=np.int64)
        slot_child: list[tuple[str, int]] = []
        slot_node: list[int] = []
        slot_alpha: list[float] = []
        ptr = [0]
        for c, (children, a) in enumerate(zip(nodes, alphas, strict=True)):
            for (kind, idx), aj in zip(children, a, strict=True):
                s = len(slot_child)
                slot_child.append((kind, idx))
                slot_node.append(c)
                slot_alpha.append(float(aj))
                if kind == "leaf":
                    if not 0 <= idx < K:
                        raise ValueError(f"Leaf index {idx} out of range for K={K}.")
                    seen_leaves.append(idx)
                    leaf_slot[idx] = s
                else:
                    if not c < idx < C:
                        raise ValueError("Child nodes must come after their parents.")
                    if parent_slot_of_node[idx] != -1:
                        raise ValueError(f"Node {idx} has more than one parent.")
                    parent_slot_of_node[idx] = s
            ptr.append(len(slot_child))
        if sorted(seen_leaves) != list(range(K)):
            raise ValueError("Tree leaves must be exactly 0..K-1, each appearing once.")
        if np.any(parent_slot_of_node[1:] < 0):
            raise ValueError("Every internal node except the root needs a parent.")
        self.K, self.C = K, C
        self.ptr = np.asarray(ptr)
        self.slot_node = np.asarray(slot_node)
        self.slot_alpha = np.asarray(slot_alpha)
        self.slot_child = slot_child
        S = len(slot_child)
        # leaf membership of every node and every child slot (bottom-up)
        node_mem = np.zeros((C, K))
        for c in range(C - 1, -1, -1):
            for kind, idx in nodes[c]:
                if kind == "leaf":
                    node_mem[c, idx] = 1.0
                else:
                    node_mem[c] += node_mem[idx]
        slot_mem = np.zeros((S, K))
        for s, (kind, idx) in enumerate(slot_child):
            if kind == "leaf":
                slot_mem[s, idx] = 1.0
            else:
                slot_mem[s] = node_mem[idx]
        self.node_mem = node_mem
        self.slot_mem = slot_mem
        A_node = np.add.reduceat(self.slot_alpha, self.ptr[:-1])
        self.A_node = A_node
        self.leaf_coef = self.slot_alpha[leaf_slot] - 1.0  # (K,)
        nr = np.arange(1, C)
        self.node_coef = self.slot_alpha[parent_slot_of_node[nr]] - A_node[nr]  # (C-1,)
        self.nonroot_mem = node_mem[1:]
        # slot_nodes[s, c'] = 1 if non-root internal node c' lies under (or is) slot s's child
        sizes = self.nonroot_mem.sum(axis=1)
        self.slot_nodes = ((slot_mem @ self.nonroot_mem.T) == sizes[None, :]).astype(float)
        self.const = float(
            np.sum(special.gammaln(A_node)) - np.sum(special.gammaln(self.slot_alpha))
        )

    # -- helpers --------------------------------------------------------------
    def compose(self, theta: np.ndarray) -> np.ndarray:
        """Leaf probabilities from branch probabilities (last axis = slots)."""
        lt = np.log(np.maximum(theta, _TINY))
        rho = np.exp(lt @ self.slot_mem)
        zero = (theta <= 0).astype(float) @ self.slot_mem > 0
        if np.any(zero):
            rho = np.where(zero, 0.0, rho)
        return rho

    def draw_theta(self, alpha_slots: np.ndarray, rng, size: int | None = None) -> np.ndarray:
        shape = (alpha_slots.size,) if size is None else (size, alpha_slots.size)
        theta = np.empty(shape)
        for c in range(self.C):
            sl = slice(self.ptr[c], self.ptr[c + 1])
            theta[..., sl] = _dirichlet_draw(alpha_slots[sl], rng, size=size)
        return theta

    def slot_counts(self, counts: np.ndarray) -> np.ndarray:
        return self.slot_mem @ counts


class _BoundTree(_SimplexBound):
    def __init__(self, prior: PopulationPrior, bins: Bins, tree: _TreeStructure) -> None:
        super().__init__(prior, bins)
        self.tree = tree
        self._a1 = tree.leaf_coef + 1.0
        self._sum_a1 = float(self._a1.sum())
        self._sum_d = float(tree.node_coef.sum())

    def _lp_eta(self, rho, log_rho):
        t = self.tree
        rc = np.maximum(t.nonroot_mem @ rho, _TINY)
        lp = float(np.dot(self._a1, log_rho) + np.dot(t.node_coef, np.log(rc)) + t.const)
        g = self._a1 - self._sum_a1 * rho + rho * (t.nonroot_mem.T @ (t.node_coef / rc))
        g = g - self._sum_d * rho
        return lp, g

    def sample(self, n, rng):
        theta = self.tree.draw_theta(self.tree.slot_alpha, rng, size=n)
        return np.maximum(self.tree.compose(theta), _TINY)

    def unbounded_density(self):
        # log-density coefficients of the branch probabilities: negative ones (e.g.
        # Polya trees, whose concentrations grow with depth) make it unbounded
        t = self.tree
        coef = t.slot_mem @ t.leaf_coef + t.slot_nodes @ t.node_coef
        return bool(np.any(coef < 0.0))

    def log_density(self, value):
        t = self.tree
        r = np.asarray(value, dtype=float)
        rc = r @ t.nonroot_mem.T
        leaf = np.sum(special.xlogy(t.leaf_coef, r), axis=-1)
        node = np.sum(special.xlogy(t.node_coef, rc), axis=-1)
        return leaf + node + t.const

    def grad_log_density(self, value):
        t = self.tree
        v = np.asarray(value, dtype=float)
        pos = v > 0
        g = np.where(pos, t.leaf_coef / np.where(pos, v, 1.0), 0.0)
        rc = t.nonroot_mem @ v
        ok = rc > 0
        g = g + t.nonroot_mem[ok].T @ (t.node_coef[ok] / rc[ok])
        return np.where(pos, g, 0.0)

    def prior_transform(self, u):
        t = self.tree
        u = np.asarray(u, dtype=float)
        theta = np.empty((*u.shape[:-1], t.slot_alpha.size))
        for c in range(t.C):  # one stick-breaking per internal node, of m - 1 uniforms
            lo, hi = t.ptr[c], t.ptr[c + 1]
            theta[..., lo:hi] = _stick_break(t.slot_alpha[lo:hi], u[..., lo - c : hi - c - 1])
        return t.compose(theta)

    def posterior_sample(self, counts, rng, exposure=None):
        t = self.tree
        return np.maximum(
            t.compose(t.draw_theta(t.slot_alpha + t.slot_counts(counts), rng)), _TINY
        )

    def map_from_counts(self, counts, exposure=None):
        # Mode of (prior density on the simplex) x prod rho^counts. The log-density is
        # linear in log leaf and node masses, i.e. linear in the log branch
        # probabilities, so each node's branches are proportional to their coefficients.
        t = self.tree
        x = t.slot_mem @ (np.asarray(counts, float) + t.leaf_coef) + t.slot_nodes @ t.node_coef
        x = np.maximum(x, 0.0)
        tot = np.add.reduceat(x, t.ptr[:-1])
        n_ch = np.diff(t.ptr)
        tot_rep = np.repeat(tot, n_ch)
        theta = np.where(
            tot_rep > 0, x / np.where(tot_rep > 0, tot_rep, 1.0), 1.0 / np.repeat(n_ch, n_ch)
        )
        return t.compose(theta)

    def vb_update(self, counts, exposure=None):
        return self.tree.slot_alpha + self.tree.slot_counts(counts)

    def vb_expected_log(self, state):
        t = self.tree
        tot = np.repeat(np.add.reduceat(state, t.ptr[:-1]), np.diff(t.ptr))
        return (special.digamma(state) - special.digamma(tot)) @ t.slot_mem

    def vb_sample(self, state, n, rng):
        return np.maximum(self.tree.compose(self.tree.draw_theta(state, rng, size=n)), _TINY)

    def vb_mean(self, state):
        t = self.tree
        tot = np.repeat(np.add.reduceat(state, t.ptr[:-1]), np.diff(t.ptr))
        return t.compose(state / tot)


def _parse_nested(tree: Any) -> list[list[tuple[str, int]]]:
    """Nested lists of leaf indices -> flattened node list (pre-order)."""
    nodes: list[list[tuple[str, int]]] = []

    def visit(sub: Any) -> int:
        if isinstance(sub, int | np.integer):
            raise ValueError("The tree root must be a list of children.")
        idx = len(nodes)
        children: list[tuple[str, int]] = []
        nodes.append(children)
        for ch in sub:
            if isinstance(ch, int | np.integer):
                children.append(("leaf", int(ch)))
            else:
                children.append(("node", visit(ch)))
        if len(children) < 2:
            raise ValueError("Every internal node of a Dirichlet-tree needs >= 2 children.")
        return idx

    visit(tree)
    return nodes


class _TreePrior(PopulationPrior):
    """A prior built as a Dirichlet-tree over the bins (subclasses give its structure)."""

    def _structure(self, bins: Bins) -> _TreeStructure:  # pragma: no cover - abstract
        raise NotImplementedError

    def bind(self, bins: Bins) -> _BoundTree:
        return _BoundTree(self, bins, self._structure(bins))


class DirichletTree(_TreePrior):
    """Dirichlet-tree prior: independent Dirichlet splits along a tree of bins.

    Generalises the Dirichlet (a single node) and the generalised Dirichlet (a
    "caterpillar" tree). Bins grouped under the same internal node can be
    positively correlated, and each split has its own concentration. Conjugate
    given latent counts.

    Parameters
    ----------
    tree : nested sequence of int
        Internal nodes are lists; leaves are bin indices ``0..K-1``, each
        appearing exactly once. E.g. ``[[0, 1], [2, [3, 4]]]``.
    alpha : float or sequence of arrays
        A single concentration for every branch, or one array per internal node
        (in pre-order, i.e. parents before children, left to right) with one
        value per child.

    Examples
    --------
    >>> import binference
    >>> prior = binference.priors.DirichletTree([[0, 1], [2, [3, 4]]], alpha=2.0)
    """

    def __init__(self, tree: Any, alpha: float | Sequence[Any] = 1.0) -> None:
        self.tree = tree
        self.alpha = alpha
        self._nodes = _parse_nested(tree)
        self._alphas = self._resolve_alphas(alpha)

    def _resolve_alphas(self, alpha) -> list[np.ndarray]:
        # per-node arrays may differ in length, so test the container before np.ndim
        if not isinstance(alpha, list | tuple) and np.ndim(alpha) == 0:
            a = float(check_positive(alpha, "alpha"))
            return [np.full(len(ch), a) for ch in self._nodes]
        alphas = [check_positive(a, "alpha").ravel() for a in alpha]
        if len(alphas) != len(self._nodes):
            raise ValueError(f"Need one alpha array per internal node ({len(self._nodes)}).")
        for c, (a, ch) in enumerate(zip(alphas, self._nodes, strict=True)):
            if a.size != len(ch):
                raise ValueError(f"alpha for node {c} needs one value per child ({len(ch)}).")
        return alphas

    def _structure(self, bins: Bins) -> _TreeStructure:
        return _TreeStructure(self._nodes, self._alphas, bins.K)


class GeneralizedDirichlet(_TreePrior):
    r"""Generalised Dirichlet (Connor & Mosimann 1969) via stick-breaking Betas.

    :math:`V_k \sim \mathrm{Beta}(a_k, b_k)` for ``k = 1..K-1`` and
    :math:`\rho_k = V_k \prod_{j<k} (1 - V_j)`, breaking sticks in bin order (or
    ``order``). It reduces to :math:`\mathrm{Dir}(\alpha_1, \dots, \alpha_K)`
    with :math:`a_k = \alpha_k` and :math:`b_k = \sum_{j>k}\alpha_j`, the sum
    including :math:`\alpha_K`; other choices give more flexible (including
    positive) correlations.

    Parameters
    ----------
    a : float or array_like (K-1,), optional
        First Beta parameter of each stick.
    b : float or array_like (K-1,), optional
        Second Beta parameter of each stick. ``None`` gives the
        Dirichlet-equivalent ``b_k = (K - 1 - k) * a`` for scalar ``a``.
    order : sequence of int, optional
        Permutation of the bins defining the stick-breaking order.
    """

    def __init__(
        self,
        a: float | Sequence[float] | np.ndarray = 1.0,
        b: float | Sequence[float] | np.ndarray | None = None,
        order: Sequence[int] | None = None,
    ) -> None:
        check_positive(a, "a")
        if b is None:
            if np.ndim(a) != 0:
                raise ValueError("Pass b explicitly when a is an array.")
        else:
            check_positive(b, "b")
        if order is not None:
            o = np.asarray(order)
            if o.ndim != 1 or sorted(o.tolist()) != list(range(o.size)):
                raise ValueError("order must be a permutation of 0..K-1.")
        self.a = a
        self.b = b
        self.order = order

    def _structure(self, bins: Bins) -> _TreeStructure:
        K = bins.K
        order = np.arange(K) if self.order is None else np.asarray(self.order, dtype=int)
        if order.size != K:
            raise ValueError(f"order must be a permutation of 0..K-1 with K={K}.")
        a = _per_item(self.a, K - 1, "a").astype(float)
        if self.b is None:
            b = (K - 1 - np.arange(K - 1)) * float(self.a)
        else:
            b = _per_item(self.b, K - 1, "b").astype(float)
        nodes: list[list[tuple[str, int]]] = []
        alphas: list[np.ndarray] = []
        for k in range(K - 1):
            right = ("leaf", int(order[K - 1])) if k == K - 2 else ("node", k + 1)
            nodes.append([("leaf", int(order[k])), right])
            alphas.append(np.array([a[k], b[k]]))
        return _TreeStructure(nodes, alphas, K)


class PolyaTree(_TreePrior):
    r"""Finite Polya-tree prior on an ordered 1-D grid (a balanced binary Dirichlet-tree).

    Bins are split recursively into halves; a split at level :math:`m`
    (root = 1) has Beta concentrations
    :math:`c\,m^{p}\cdot 2 G_0(\text{child})/G_0(\text{parent})`, so the prior
    mean is the base measure :math:`G_0` and fluctuations shrink at finer
    scales (``power=2`` gives absolutely continuous limits; Lavine 1992). A
    conjugate, multi-scale alternative to smoothness priors.

    Parameters
    ----------
    c : float
        Overall concentration (larger = closer to the base measure).
    power : float
        Growth of concentration with depth.
    base : Density, "flat", callable or array_like (K,)
        Base measure, read by :meth:`Density.coerce <binference.Density.coerce>`
        (an array is masses per bin). The default is flat in the measure
        coordinate.
    """

    def __init__(
        self,
        c: float = 1.0,
        power: float = 2.0,
        base: Any = "flat",
    ) -> None:
        check_positive(c, "c")
        if not (np.isfinite(power) and power >= 0):
            raise ValueError("power must be finite and non-negative.")
        self.c = float(c)
        self.power = float(power)
        self.base = Density.coerce(base, "mass", name="base")

    def _structure(self, bins: Bins) -> _TreeStructure:
        if bins.is_categorical or bins.ndim != 1:
            raise ValueError("PolyaTree is defined on ordered one-dimensional bins.")
        K = bins.K
        if K < 2:
            raise ValueError("PolyaTree needs at least two bins.")
        g0 = self.base.masses(bins)
        if np.any(g0 <= 0):
            raise ValueError("base must give a positive mass for every bin.")
        cum = np.concatenate(([0.0], np.cumsum(g0)))
        nodes: list[list[tuple[str, int]]] = []
        alphas: list[np.ndarray] = []

        def mass(lo: int, hi: int) -> float:
            return cum[hi] - cum[lo]

        def build(lo: int, hi: int, level: int) -> int:
            idx = len(nodes)
            children: list[tuple[str, int]] = []
            a = np.empty(2)
            nodes.append(children)
            alphas.append(a)
            mid = (lo + hi + 1) // 2
            for j, (clo, chi) in enumerate(((lo, mid), (mid, hi))):
                if chi - clo == 1:
                    children.append(("leaf", clo))
                else:
                    children.append(("node", build(clo, chi, level + 1)))
                with np.errstate(over="ignore"):  # inf is refused below
                    depth = np.power(float(level), self.power)
                a[j] = self.c * depth * 2.0 * mass(clo, chi) / mass(lo, hi)
            return idx

        build(0, K, 1)
        for a in alphas:  # a large power can overflow level**power
            check_positive(a, "PolyaTree concentrations (c * level**power)")
        return _TreeStructure(nodes, alphas, K)
