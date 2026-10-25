"""Saving, loading and converting results."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ._version import __version__
from .basis import Basis
from .bins import Bins
from .measure import Measure
from .selection import Selection
from .selection import _rebuild as _rebuild_selection

if TYPE_CHECKING:  # pragma: no cover
    from .results import PopulationResult

__all__ = ["load_result", "result_to_dict", "save_result", "to_datatree"]

SCHEMA_VERSION = 1


def _bins_to_parts(bins: Bins) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if bins.is_categorical:
        # JSON keeps integer, float and string labels as they are (audit O-10)
        labels = [lab.item() if isinstance(lab, np.generic) else lab for lab in bins.labels]
        if not all(isinstance(lab, int | float | str) for lab in labels):
            raise ValueError("Category labels must be integers, floats or strings to be saved.")
        return {"bins": "categories", "labels": labels}, {}
    arrays = {f"edges_{d}": np.asarray(e) for d, e in enumerate(bins.edges)}
    names = []
    for d, (m, e) in enumerate(zip(bins.measures, bins.edges, strict=True)):
        names.append(m.name)
        if not m.is_named:  # other measures are saved as tables
            arrays[f"measure_z_{d}"], arrays[f"measure_u_{d}"] = m._tabulate(e)
    if bins.is_masked:
        arrays["mask"] = bins.mask
    return {"bins": "grid", "ndim": bins.ndim, "measures": names}, arrays


def _bins_from_parts(meta: dict[str, Any], arrays: Any) -> Bins:
    if meta["bins"] == "categories":
        labels = meta["labels"]
        if all(isinstance(lab, int) for lab in labels) and labels == list(range(len(labels))):
            return Bins.categories(len(labels))
        return Bins.categories(labels)
    nd = int(meta["ndim"])
    measures = []
    for d, name in enumerate(meta.get("measures", ["linear"] * nd)):
        if f"measure_z_{d}" not in arrays:
            measures.append(name)
        else:
            z, u = arrays[f"measure_z_{d}"], arrays[f"measure_u_{d}"]
            measures.append(Measure.tabulated(z, u, name=name))
    mask = np.asarray(arrays["mask"]) if "mask" in arrays else None
    edges = [np.asarray(arrays[f"edges_{d}"]) for d in range(nd)]
    return Bins.grid(*edges, measure=measures, mask=mask)


def _basis_meta(basis: Basis) -> dict[str, Any]:
    return {
        "order": basis.order,
        "resolution": list(basis.resolution),
        "oversample": basis.oversample,
    }


def _selection_parts(sel: Selection) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values, row, scale, _, injections = sel._parts()
    meta: dict[str, Any] = {
        "scale": scale,
        "row": row if row is None or isinstance(row, str) else "array",
    }
    arrays = {"selection_values": np.asarray(values)}
    if meta["row"] == "array":
        arrays["selection_row"] = np.asarray(row)
    if injections is not None:
        meta["injections"] = {"prior": list(injections.prior), "smooth": injections.smooth}
        arrays["selection_injected"] = np.asarray(injections.injected)
        arrays["selection_recovered"] = np.asarray(injections.recovered)
    return meta, arrays


def _selection_from_parts(meta: dict[str, Any], arrays: Any, basis: Basis | None) -> Selection:
    row = np.asarray(arrays["selection_row"]) if meta["row"] == "array" else meta["row"]
    inj = meta.get("injections")
    if inj is not None:
        inj = (
            np.asarray(arrays["selection_injected"]),
            np.asarray(arrays["selection_recovered"]),
            tuple(inj["prior"]),
            inj["smooth"],
        )
    return _rebuild_selection(
        np.asarray(arrays["selection_values"]), row, meta["scale"], basis, inj
    )


def result_to_dict(result: PopulationResult) -> dict[str, Any]:
    """Arrays and metadata of a result as a plain dictionary: what :func:`save_result`
    writes, with ``"meta"`` the JSON-compatible metadata and every other entry an array.

    Engine statistics that are neither numbers, strings nor numeric arrays (e.g. lists
    of NUTS metrics) are left out and named in ``meta["info_dropped"]``.
    """
    meta, arrays = _bins_to_parts(result.bins)
    meta.update(
        schema=SCHEMA_VERSION,
        binference_version=__version__,
        method=result.method,
        kind=result.kind,
        approximate=bool(result.approximate),
        prior=result.prior_repr,
    )
    scalars, dropped = {}, []
    for k, v in result.info.items():
        if isinstance(v, int | float | bool | str | np.integer | np.floating | np.bool_):
            scalars[k] = v.item() if hasattr(v, "item") else v
        elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            arrays[f"info_{k}"] = v
        elif k == "hyperparameters" and isinstance(v, dict):
            for name, draws in v.items():
                arrays[f"hyper_{name}"] = np.asarray(draws, dtype=float)
        else:
            dropped.append(k)
    meta["info"] = scalars
    if dropped:
        meta["info_dropped"] = dropped
    out: dict[str, Any] = {"draws": result._draws}
    if result.basis is not None:  # the draws per bin follow from the component draws
        meta["basis"] = _basis_meta(result.basis)
        out["component_draws"] = result.component_draws
    if result.selection is not None:  # per-object methods need it again (audit O-1)
        meta["selection"], sel_arrays = _selection_parts(result.selection)
        arrays.update(sel_arrays)
    total = result.total if result.kind == "fractions" else None
    if total is not None:  # kept so a result loaded without its likelihoods has rates
        out["total"] = total.reshape(result._draws.shape[:2])
    if result.log_density_unconstrained is not None:
        out["log_density_unconstrained"] = np.asarray(result.log_density_unconstrained)
    if result.labels is not None:  # draws of the objects' bins, not the bins' labels
        out["label_draws"] = np.asarray(result.labels)
        if result.label_index is not None:
            out["label_index"] = np.asarray(result.label_index)
    out.update(arrays)
    out["meta"] = meta
    return out


def save_result(result: PopulationResult, path: str) -> None:
    """Save a result to a compressed ``.npz`` file (versioned schema;
    :func:`result_to_dict`).

    A :meth:`Measure.transform <binference.Measure.transform>` axis is saved as
    a table of points, so it loads as a tabulated measure: the loaded bins
    compare unequal to the original ones.
    """
    payload = result_to_dict(result)
    payload["meta"] = np.array(json.dumps(payload["meta"]))
    np.savez_compressed(path, **payload)


def load_result(path: str) -> PopulationResult:
    """Load a result saved with :func:`save_result` (``.npz`` may be left off ``path``).

    The returned result is detached from its likelihoods; call
    ``result.attach(likelihoods)`` to use per-object methods.
    """
    from .results import PopulationResult

    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path += ".npz"  # np.savez_compressed adds the suffix (audit O-10)
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
        basis = Basis(bins, **meta["basis"]) if "basis" in meta else None
        selection = (
            _selection_from_parts(meta["selection"], data, basis) if "selection" in meta else None
        )
        # files written before C12a call the label draws "labels"
        label_key = next((k for k in ("label_draws", "labels") if k in data.files), None)
        return PopulationResult(
            None,
            np.array(data["component_draws" if basis is not None else "draws"]),
            method=meta["method"],
            kind=meta["kind"],
            bins=bins,
            log_density_unconstrained=next(
                (
                    np.array(data[k])
                    for k in ("log_density_unconstrained", "log_prob")
                    if k in data.files
                ),
                None,
            ),  # files written before DECISIONS #70 call it log_prob
            approximate=meta["approximate"],
            info=info,
            labels=None if label_key is None else np.array(data[label_key]),
            label_index=np.array(data["label_index"]) if "label_index" in data.files else None,
            prior_repr=meta.get("prior"),
            basis=basis,
            total=np.array(data["total"]) if "total" in data.files else None,
            selection=selection,
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
    rates = result.rates
    if rates is not None:
        data_vars["rate"] = (("chain", "draw", "bin"), rates.reshape(C.shape))
        data_vars["total"] = (("chain", "draw"), result.total.reshape(C.shape[:2]))
    if "selection_draws" in result.info:  # an uncertain selection, per draw
        data_vars["selection"] = (("chain", "draw", "bin"), result.info["selection_draws"])
    if "response_draws" in result.info:  # an inferred response, per draw
        dims = ("chain", "draw", "measured_bin", "bin")
        data_vars["response"] = (dims, result.info["response_draws"])
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
    lp = result.log_density_unconstrained
    if lp is not None and np.shape(lp) == (n_chains, n_draws):
        stats["lp"] = (("chain", "draw"), np.asarray(lp))
    if stats:
        groups["sample_stats"] = xr.Dataset(
            stats, coords={k: coords[k] for k in ("chain", "draw")}
        )
    return xr.DataTree.from_dict(groups)
