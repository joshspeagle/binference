"""Saving, loading and converting results."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ._version import __version__
from .bins import Bins

if TYPE_CHECKING:  # pragma: no cover
    from .results import PopulationResult

__all__ = ["load_result", "result_to_dict", "save_result", "to_datatree"]

SCHEMA_VERSION = 1


def _bins_to_parts(bins: Bins) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if bins.is_categorical:
        labels = list(bins.labels)
        if all(isinstance(lab, int | np.integer) for lab in labels):
            return {"bins": "categories", "labels": [int(lab) for lab in labels]}, {}
        return {"bins": "categories", "labels": [str(lab) for lab in labels]}, {}
    arrays = {f"edges_{d}": np.asarray(e) for d, e in enumerate(bins.edges)}
    return {"bins": "grid", "ndim": bins.ndim}, arrays


def _bins_from_parts(meta: dict[str, Any], arrays: Any) -> Bins:
    if meta["bins"] == "categories":
        labels = meta["labels"]
        if all(isinstance(lab, int) for lab in labels) and labels == list(range(len(labels))):
            return Bins.categories(len(labels))
        return Bins.categories(labels)
    return Bins.grid(*[np.asarray(arrays[f"edges_{d}"]) for d in range(int(meta["ndim"]))])


def result_to_dict(result: PopulationResult) -> dict[str, Any]:
    """Arrays and metadata of a result as a plain dictionary."""
    meta, arrays = _bins_to_parts(result.bins)
    out: dict[str, Any] = {
        "draws": result._draws,
        "method": result.method,
        "kind": result.kind,
        "approximate": bool(result.approximate),
        "prior": result.prior_repr,
        **meta,
        **arrays,
    }
    if result.log_prob is not None:
        out["log_prob"] = result.log_prob
    if result.labels is not None:
        out["labels"] = result.labels
        if result.label_index is not None:
            out["label_index"] = result.label_index
    return out


def save_result(result: PopulationResult, path: str) -> None:
    """Save a result to a compressed ``.npz`` file (versioned schema)."""
    meta, arrays = _bins_to_parts(result.bins)
    meta.update(
        schema=SCHEMA_VERSION,
        binference_version=__version__,
        method=result.method,
        kind=result.kind,
        approximate=bool(result.approximate),
        prior=result.prior_repr,
    )
    scalars = {}
    for k, v in result.info.items():
        if isinstance(v, int | float | bool | str | np.integer | np.floating | np.bool_):
            scalars[k] = v.item() if hasattr(v, "item") else v
        elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            arrays[f"info_{k}"] = v
        elif k == "hyperparameters" and isinstance(v, dict):
            for name, draws in v.items():
                arrays[f"hyper_{name}"] = np.asarray(draws, dtype=float)
    meta["info"] = scalars
    payload: dict[str, Any] = {"draws": result._draws, "meta": np.array(json.dumps(meta))}
    payload.update(arrays)
    if result.log_prob is not None:
        payload["log_prob"] = np.asarray(result.log_prob)
    if result.labels is not None:
        payload["labels"] = np.asarray(result.labels)
        if result.label_index is not None:
            payload["label_index"] = np.asarray(result.label_index)
    np.savez_compressed(path, **payload)


def load_result(path: str) -> PopulationResult:
    """Load a result saved with :func:`save_result`.

    The returned result is detached from its likelihoods; call
    ``result.attach(likelihoods)`` to use per-object methods.
    """
    from .results import PopulationResult

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema", 0) > SCHEMA_VERSION:
            raise ValueError("This file was written by a newer version of binference.")
        bins = _bins_from_parts(meta, data)
        info: dict[str, Any] = dict(meta.get("info", {}))
        hyper = {}
        for key in data.files:
            if key.startswith("info_"):
                info[key[5:]] = np.array(data[key])
            elif key.startswith("hyper_"):
                hyper[key[6:]] = np.array(data[key])
        if hyper:
            info["hyperparameters"] = hyper
        return PopulationResult(
            None,
            np.array(data["draws"]),
            method=meta["method"],
            kind=meta["kind"],
            bins=bins,
            log_prob=np.array(data["log_prob"]) if "log_prob" in data.files else None,
            approximate=meta["approximate"],
            info=info,
            labels=np.array(data["labels"]) if "labels" in data.files else None,
            label_index=np.array(data["label_index"]) if "label_index" in data.files else None,
            prior_repr=meta.get("prior"),
        )


def to_datatree(result: PopulationResult):
    """Convert to an :class:`xarray.DataTree` with ``posterior`` and ``sample_stats`` groups.

    Follows the ArviZ (>= 1.0) layout: dimensions ``chain``, ``draw`` and
    ``bin``. Requires xarray >= 2024.10.
    """
    try:
        import xarray as xr
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError("to_datatree needs xarray: pip install 'binference[xarray]'.") from err
    C = result.chains
    n_chains, n_draws, _ = C.shape
    from .results import _bin_labels

    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_draws),
        "bin": np.asarray(_bin_labels(result.bins), dtype=str),
    }
    data_vars = {"rho": (("chain", "draw", "bin"), C)}
    if result.rates is not None:
        data_vars["rate"] = (("chain", "draw", "bin"), result._draws)
    post = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={"method": result.method, "approximate": int(result.approximate)},
    )
    groups = {"posterior": post}
    stats = {
        k: (("chain", "draw"), np.asarray(v))
        for k, v in result.info.items()
        if isinstance(v, np.ndarray) and v.shape == (n_chains, n_draws)
    }
    if result.log_prob is not None and np.shape(result.log_prob) == (n_chains, n_draws):
        stats["lp"] = (("chain", "draw"), np.asarray(result.log_prob))
    if stats:
        groups["sample_stats"] = xr.Dataset(
            stats, coords={k: coords[k] for k in ("chain", "draw")}
        )
    return xr.DataTree.from_dict(groups)
