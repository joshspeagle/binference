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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import special

from .._simplex import helmert_basis, log_softmax, softmax
from .._utils import check_nonnegative, check_positive
from ..bins import Bins, _bin_labels
from ..density import Density
from .base import (
    _TINY,
    BoundPrior,
    PopulationPrior,
    _log_gamma_variates,
    _per_item,
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
    alpha : float or array_like (K,)
        Concentration parameters (strictly positive).

    Notes
    -----
    A symmetric ``alpha`` over bins of unequal volume is *not* consistent under
    merging bins; :meth:`from_base_measure` makes ``alpha`` proportional to a
    base measure (flat in the measure coordinate by default) instead.
    """

    def __init__(self, alpha: float | Sequence[float] | np.ndarray = 1.0) -> None:
        check_positive(alpha, "Dirichlet alpha")
        self.alpha = alpha
        self._base: tuple[float, Any] | None = None

    def _params(self):
        if self._base is not None:
            return {"concentration": self._base[0], "base": self._base[1]}
        return {"alpha": self.alpha}

    @classmethod
    def reference(cls) -> Dirichlet:
        """Reference prior ``alpha = 1/K`` (Berger, Bernardo & Sun 2015).

        ``K`` is the number of bins the prior is bound to. Adding one pseudo-object in all
        rather than ``K``, it avoids swamping sparse problems with many bins.
        """
        obj = cls()
        obj.alpha = "reference"  # 1/K, set at binding
        return obj

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
        c = check_nonnegative(counts, "counts")
        return cls(check_positive(base, "base") + check_nonnegative(strength, "strength") * c)

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

    def describe(self, bins: Bins, counts: Any = None) -> str:
        """The prior as a one-node tree, drawn as text (see :meth:`DirichletTree.describe`).

        Parameters
        ----------
        bins : Bins
            The bins it is drawn on.
        counts : array_like (K,), optional
            Counts per bin; the drawing then shows the posterior given them.
        """
        alpha = self.resolve_alpha(bins)
        tree = _TreeStructure([[("leaf", k) for k in range(bins.K)]], [alpha], bins.K)
        return _describe(self, bins, tree, counts)


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
        # the priors build valid trees and validate the concentrations
        if K < 2:
            raise ValueError("The simplex needs at least two bins (K >= 2).")
        C = len(nodes)
        parent_slot_of_node = np.full(C, -1, dtype=np.int64)
        leaf_slot = np.full(K, -1, dtype=np.int64)
        slot_child: list[tuple[str, int]] = []
        slot_node: list[int] = []
        slot_alpha: list[float] = []
        node_start = [0]  # node c's slots are node_start[c] to node_start[c + 1] - 1 (CSR)
        for c, (children, a) in enumerate(zip(nodes, alphas, strict=True)):
            for (kind, idx), aj in zip(children, a, strict=True):
                s = len(slot_child)
                slot_child.append((kind, idx))
                slot_node.append(c)
                slot_alpha.append(float(aj))
                if kind == "leaf":
                    leaf_slot[idx] = s
                else:
                    parent_slot_of_node[idx] = s
            node_start.append(len(slot_child))
        self.K, self.C = K, C
        self.node_start = np.asarray(node_start)
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
        # each branch's exponent on the simplex: its alpha less the bins below it
        self.slot_exponent = self.slot_alpha - slot_mem.sum(axis=1)
        A_node = np.add.reduceat(self.slot_alpha, self.node_start[:-1])
        self.leaf_coef = self.slot_alpha[leaf_slot] - 1.0  # (K,)
        nr = np.arange(1, C)
        self.node_coef = self.slot_alpha[parent_slot_of_node[nr]] - A_node[nr]  # (C-1,)
        self.nonroot_mem = node_mem[1:]
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
            sl = slice(self.node_start[c], self.node_start[c + 1])
            theta[..., sl] = _dirichlet_draw(alpha_slots[sl], rng, size=size)
        return theta

    def slot_counts(self, counts: np.ndarray) -> np.ndarray:
        return self.slot_mem @ counts

    def node_totals(self, x: np.ndarray) -> np.ndarray:
        """For each slot, the sum of ``x`` over the slots of its node."""
        return np.repeat(np.add.reduceat(x, self.node_start[:-1]), np.diff(self.node_start))


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
        # a branch with less concentration than bins below it (e.g. near the root of a
        # Polya tree with c below K/2) has a negative exponent, which makes the density
        # unbounded. Read from alpha, not summed from the log-density's coefficients,
        # whose rounding can put an exponent of zero below it.
        return bool(np.any(self.tree.slot_exponent < 0.0))

    def long_log_tails(self):
        # the Jacobian of the unconstrained coordinates adds one per bin below a
        # branch, so there its exponent is its alpha
        return bool(np.any(self.tree.slot_alpha < 1.0))

    def _kkt_violation(self, value, grad, include_prior=True):
        # In the branch probabilities theta, where the log-density is sum_j e_j log theta_j:
        # at each node with mass, (n_j + e_j) / theta_j for the counts n below branch j,
        # against lambda = sum (n_j + e_j) over its branches with mass (EM's M-step sets
        # theta_j to (n_j + e_j) / lambda). A branch of exponent zero adds exactly zero
        # here; in rho its terms of order 1/rho cancel only to round-off, which once a group
        # of bins empties (rho near 1e-22) is a gap of order 1e5.
        if not include_prior:
            return super()._kkt_violation(value, grad, False)
        t = self.tree
        vt = np.maximum(value, _TINY)  # as in the MAP objective
        mass, n, live = (t.slot_mem @ np.column_stack([vt, vt * grad, value])).T
        x = n + t.slot_exponent
        lam = t.node_totals(np.where(live > 0, x, 0.0))
        r = x * t.node_totals(mass) / mass - lam
        return float(np.max(r, where=t.node_totals(live) > 0, initial=-np.inf))

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

    def posterior_sample(self, counts, rng, exposure=None):
        t = self.tree
        return np.maximum(
            t.compose(t.draw_theta(t.slot_alpha + t.slot_counts(counts), rng)), _TINY
        )

    def map_from_counts(self, counts, exposure=None):
        # Mode of (prior density on the simplex) x prod rho^counts. The log-density is
        # linear in the log branch probabilities, so each node's branches are proportional
        # to their exponents plus the counts below them.
        t = self.tree
        x = np.maximum(t.slot_mem @ np.asarray(counts, float) + t.slot_exponent, 0.0)
        tot = t.node_totals(x)
        uniform = 1.0 / t.node_totals(np.ones_like(x))  # a node with no mass splits evenly
        return t.compose(np.where(tot > 0, x / np.where(tot > 0, tot, 1.0), uniform))

    def vb_update(self, counts, exposure=None):
        return self.tree.slot_alpha + self.tree.slot_counts(counts)

    def vb_expected_log(self, state):
        t = self.tree
        return (special.digamma(state) - special.digamma(t.node_totals(state))) @ t.slot_mem

    def vb_sample(self, state, n, rng):
        return np.maximum(self.tree.compose(self.tree.draw_theta(state, rng, size=n)), _TINY)

    def vb_mean(self, state):
        return self.tree.compose(state / self.tree.node_totals(state))


def _parse_tree(tree: Any) -> tuple[list[list[tuple[str, Any]]], list[list[list[Any]]]]:
    """A tree of bins -> its nodes in pre-order and the names of each node's branches.

    Groups are lists, or dicts from group names to members; a member is a group or a bin
    (an index, or a label resolved when the prior is bound). A named group of one member is
    that member. ``nodes[c]`` lists node ``c``'s children, each ``("leaf", bin)`` or
    ``("node", c')``; ``names[c][j]`` holds the group names of the branch to child ``j``.
    """
    nodes: list[list[tuple[str, Any]]] = []
    names: list[list[list[Any]]] = []
    seen: set[Any] = set()

    def is_group(member: Any) -> bool:
        return isinstance(member, Mapping | Sequence | np.ndarray) and not isinstance(
            member, str | bytes
        )

    def members(group: Any) -> list[tuple[Any, Any]]:
        if isinstance(group, Mapping):
            for name in group:
                if name in seen:
                    raise ValueError(f"tree: the group name {name!r} is used twice.")
                seen.add(name)
            return list(group.items())
        if not is_group(group):
            raise ValueError("tree: the root must be a list or dict of groups and bins.")
        return [(None, m) for m in group]

    def child(member: Any, name: Any) -> tuple[str, Any, list[Any]]:
        named = [] if name is None else [name]
        if is_group(member):
            items = members(member)
            # a group of one named member (a named group of one, or a dict of one group)
            # is that member
            if len(items) == 1 and (named or items[0][0] is not None):
                kind, value, inner = child(items[0][1], items[0][0])
                return kind, value, named + inner
            return "node", visit(items, name), named
        if isinstance(member, bool):
            raise ValueError(f"tree: {member!r} is not a bin.")
        try:
            hash(member)
        except TypeError:
            raise ValueError(
                f"tree: {member!r} is neither a bin nor a group (a list or a dict)."
            ) from None
        return "leaf", int(member) if isinstance(member, int | np.integer) else member, named

    def visit(items: list[tuple[Any, Any]], name: Any) -> int:
        idx = len(nodes)
        nodes.append([])
        names.append([])
        for member_name, member in items:
            kind, value, branch = child(member, member_name)
            nodes[idx].append((kind, value))
            names[idx].append(branch)
        if len(nodes[idx]) < 2:
            where = "the root" if idx == 0 else "a group" if name is None else f"group {name!r}"
            raise ValueError(f"tree: {where} needs at least two members.")
        return idx

    visit(members(tree), None)
    return nodes, names


def _bin_names(bins: Bins) -> list[str]:
    """Each bin as drawings and errors name it: its label, or its index and its cell."""
    cells = _bin_labels(bins)
    return list(cells) if bins.is_categorical else [f"bin {k} {c}" for k, c in enumerate(cells)]


def _span(bins: Bins, ks: np.ndarray, names: list[str]) -> str:
    if bins.is_categorical:
        shown = ", ".join(names[k] for k in ks[:6])
        return shown + (f", … ({ks.size} bins)" if ks.size > 6 else "")
    if np.all(np.diff(ks) == 1):
        return f"bins {ks[0]}–{ks[-1]}"
    return "bins " + ", ".join(map(str, ks[:8])) + (", …" if ks.size > 8 else "")


def _describe(
    prior: PopulationPrior,
    bins: Bins,
    t: _TreeStructure,
    counts: Any,
    names: list[list[Any]] | None = None,
) -> str:
    """The tree as text: each branch's concentration, mean share and mean mass."""
    alpha = t.slot_alpha
    if counts is not None:
        n = check_nonnegative(counts, "counts")
        if n.shape != (bins.K,):
            raise ValueError(
                f"counts must have one value per bin ({bins.K}); got shape {n.shape}."
            )
        alpha = alpha + t.slot_counts(n)
    share = alpha / t.node_totals(alpha)
    flat = [None] * len(t.slot_child) if names is None else [b for node in names for b in node]
    bin_names = _bin_names(bins)
    given = "posterior given the counts" if counts is not None else "prior"
    lines = [f"{type(prior).__name__}: {given}, on {bins.K} bins"]
    lines.append(f"root: {_span(bins, np.arange(bins.K), bin_names)}")
    # depth first, children in order, without recursion (a caterpillar is K - 1 deep); a
    # branch's mean mass is the product of the mean shares down to it, as the shares of
    # different nodes are independent
    stack = [(s, "", 1.0) for s in reversed(range(t.node_start[0], t.node_start[1]))]
    while stack:
        s, indent, above = stack.pop()
        kind, idx = t.slot_child[s]
        last = s == t.node_start[t.slot_node[s] + 1] - 1
        mass = above * share[s]
        if kind == "leaf":
            what = bin_names[idx]
        else:
            what = _span(bins, np.flatnonzero(t.node_mem[idx]), bin_names)
            inner = indent + ("    " if last else "│   ")
            stack += [
                (j, inner, mass) for j in reversed(range(t.node_start[idx], t.node_start[idx + 1]))
            ]
        label = f"{flat[s][0]}: {what}" if flat[s] else what
        lines.append(
            f"{indent}{'└── ' if last else '├── '}α={alpha[s]:.3g}  "
            f"share {share[s]:.2f}  mass {mass:.3f}  {label}"
        )
    return "\n".join(lines)


class _TreePrior(PopulationPrior):
    """A prior built as a Dirichlet-tree over the bins (subclasses give its structure)."""

    #: the group names on each node's branches (``DirichletTree`` with named groups)
    _names: list[list[list[Any]]] | None = None

    def _structure(self, bins: Bins) -> _TreeStructure:  # pragma: no cover - abstract
        raise NotImplementedError

    def bind(self, bins: Bins) -> _BoundTree:
        return _BoundTree(self, bins, self._structure(bins))

    def describe(self, bins: Bins, counts: Any = None) -> str:
        """The tree on ``bins``, drawn as text.

        Each line is a branch: its concentration ``α``, the share of its parent it takes
        on average, the mean fraction of the population below it, and the bins it leads
        to. With ``counts`` the drawing shows the posterior given them, each branch's
        concentration raised by the counts of the bins below it.

        Parameters
        ----------
        bins : Bins
            The bins it is drawn on.
        counts : array_like (K,), optional
            Counts per bin.

        Examples
        --------
        >>> import binference
        >>> bins = binference.Bins.linear(0.0, 1.0, 4)
        >>> print(binference.priors.PolyaTree().describe(bins))
        PolyaTree: prior, on 4 bins
        root: bins 0–3
        ├── α=1  share 0.50  mass 0.500  bins 0–1
        │   ├── α=4  share 0.50  mass 0.250  bin 0 [0, 0.25)
        │   └── α=4  share 0.50  mass 0.250  bin 1 [0.25, 0.5)
        └── α=1  share 0.50  mass 0.500  bins 2–3
            ├── α=4  share 0.50  mass 0.250  bin 2 [0.5, 0.75)
            └── α=4  share 0.50  mass 0.250  bin 3 [0.75, 1)
        """
        return _describe(self, bins, self._structure(bins), counts, self._names)


class DirichletTree(_TreePrior):
    """Dirichlet-tree prior: independent Dirichlet splits along a tree of bins.

    Generalises the Dirichlet (a single node) and the generalised Dirichlet (a
    "caterpillar" tree). Bins grouped under the same internal node can be
    positively correlated, and each split has its own concentration. Conjugate
    given latent counts.

    Parameters
    ----------
    tree : nested lists, or dicts of named groups
        The groups of bins. A group is a list of members, or a dict from group names
        to members; a member is a group or a bin. Bins are indices ``0..K-1`` or, on
        categorical bins, their labels, and each appears exactly once. A named group
        of one member is that member. E.g. ``[[0, 1], [2, [3, 4]]]``, or
        ``{"spiral": ["Sa", "Sb"], "early": {"elliptical": ["E0", "E3"], "lenticular": ["S0"]}}``.
    alpha : float, sequence of arrays, or dict
        A single concentration for every branch; one array per internal node (in
        pre-order, i.e. parents before children, left to right) with one value per
        member; or a dict from group names, and bins as the tree writes them, to the
        concentration of the branch leading into each, every other branch keeping 1.

    Examples
    --------
    >>> import binference
    >>> prior = binference.priors.DirichletTree([[0, 1], [2, [3, 4]]], alpha=2.0)
    >>> groups = {
    ...     "spiral": ["Sa", "Sb"],
    ...     "early": {"elliptical": ["E0", "E3"], "lenticular": ["S0"]},
    ... }
    >>> named = binference.priors.DirichletTree(groups, alpha={"spiral": 3.0})
    """

    describe = _TreePrior.describe  # documented on this class's API page

    def __init__(
        self, tree: Any, alpha: float | Sequence[Any] | Mapping[Any, float] = 1.0
    ) -> None:
        self.tree = tree
        self.alpha = alpha
        self._nodes, self._names = _parse_tree(tree)
        self._alphas = self._resolve_alphas(alpha)

    def _resolve_alphas(self, alpha) -> list[np.ndarray]:
        if isinstance(alpha, Mapping):  # keyed by group names and bins as the tree writes them
            out, used = [np.ones(len(ch)) for ch in self._nodes], set()
            for c, children in enumerate(self._nodes):
                for j, (kind, v) in enumerate(children):
                    names = self._names[c][j] + ([v] if kind == "leaf" else [])
                    keys = [key for key in names if key in alpha]
                    if len(keys) > 1:
                        raise ValueError(f"alpha: {keys[0]!r} and {keys[1]!r} name one branch.")
                    if not keys:
                        continue
                    if keys[0] in used:  # a group named like a bin
                        raise ValueError(f"alpha: {keys[0]!r} names more than one branch.")
                    a = check_positive(alpha[keys[0]], f"alpha[{keys[0]!r}]")
                    if a.ndim:
                        raise ValueError(f"alpha[{keys[0]!r}] must be one number.")
                    out[c][j] = float(a)
                    used.add(keys[0])
            unknown = [key for key in alpha if key not in used]
            if unknown:
                raise ValueError(
                    f"alpha: {', '.join(map(repr, unknown))} names no group or bin of the tree."
                )
            return out
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
        lookup = {} if bins.labels is None else {lab: k for k, lab in enumerate(bins.labels)}

        def index(leaf: Any) -> int:
            if isinstance(leaf, int):
                if not 0 <= leaf < bins.K:
                    raise ValueError(f"tree: bin index {leaf} is out of range for K={bins.K}.")
                return leaf
            if leaf not in lookup:
                if bins.labels is None:
                    raise ValueError(
                        f"tree: {leaf!r} is not a bin index (labels need categories)."
                    )
                raise ValueError(f"tree: {leaf!r} is not one of the bins' labels.")
            return lookup[leaf]

        nodes = [
            [(kind, index(v) if kind == "leaf" else v) for kind, v in ch] for ch in self._nodes
        ]
        n = np.bincount([v for ch in nodes for kind, v in ch if kind == "leaf"], minlength=bins.K)
        for bad, what in ((n > 1, "appear more than once"), (n == 0, "are in no group")):
            if bad.any():
                names = _bin_names(bins)
                raise ValueError(
                    f"tree: bins {what}: {', '.join(names[k] for k in np.flatnonzero(bad))}."
                )
        return _TreeStructure(nodes, self._alphas, bins.K)


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
        Dirichlet-equivalent ``b_k = (K - k) * a`` for scalar ``a``.
    order : sequence of int, optional
        Permutation of the bins defining the stick-breaking order.
    """

    describe = _TreePrior.describe  # documented on this class's API page

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

    describe = _TreePrior.describe  # documented on this class's API page

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
