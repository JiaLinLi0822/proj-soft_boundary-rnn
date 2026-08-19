from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import matplotlib.pyplot as plt
import numpy as np

from neuralSPRT_BEH import neuralSPRT_BEH
from neuralSPRT_PHY import neuralSPRT_PHY


def _load_mat73(file_path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with h5py.File(str(file_path), "r") as f:
        def load_group(group: h5py.Group) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for key in group.keys():
                if key == "#refs#":
                    continue
                obj = group[key]
                if isinstance(obj, h5py.Dataset):
                    out[key] = np.array(obj)
                elif isinstance(obj, h5py.Group):
                    out[key] = load_group(obj)
            return out

        for key in f.keys():
            if key == "#refs#":
                continue
            if isinstance(f[key], h5py.Group):
                data[key] = load_group(f[key])
            else:
                data[key] = np.array(f[key])
    return data


def _configure_matlab_like_plot_defaults() -> None:
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def _as_int_list(values: Iterable[int] | None) -> List[int]:
    if values is None:
        return []
    return [int(v) for v in values]


def neuralSPRT(
    monkey_id: int,
    fig_switch_beh: Iterable[int] | None,
    fig_switch_phy: Iterable[int] | None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Python reproduction of MATLAB neuralSPRT.m.

    monkey_id: 1 (E) or 2 (J)
    fig_switch_beh: behavior figure switches (1..6)
    fig_switch_phy: physiology figure switches (1..4)
    """
    plt.close("all")
    np.random.seed(0)
    _configure_matlab_like_plot_defaults()

    fig_switch_beh = _as_int_list(fig_switch_beh)
    fig_switch_phy = _as_int_list(fig_switch_phy)

    if monkey_id == 1:
        monkey = "E"
        tau_s = 200
    elif monkey_id == 2:
        monkey = "J"
        tau_s = 130
    else:
        raise ValueError("monkey_id must be 1 or 2")

    here = Path(__file__).resolve().parent
    project_root = here.parent

    if data_dir is None:
        data_dir = project_root / "neuralSPRT" / "data"

    mat_path = data_dir / f"data{monkey}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    raw = _load_mat73(mat_path)
    if "info" not in raw:
        raise KeyError(f"MAT file missing 'info' key: {mat_path}")

    info = raw["info"]

    # Remove fields as in MATLAB neuralSPRT.m (if present)
    info.pop("FM", None)
    info.pop("SpM", None)
    info.pop("adMtrx", None)

    # Transpose arrays saved in MATLAB column-major style
    if "TM" in info:
        info["TM"] = np.array(info["TM"]).T
    if "KM" in info:
        info["KM"] = np.array(info["KM"]).T
    if "popID" in info:
        pop = np.asarray(info["popID"]).reshape(-1)
        if "TM" in info and pop.size == info["TM"].shape[0]:
            info["popID"] = pop
            info["num_cell"] = int(np.unique(pop).size)
        else:
            # Keep behavior identical to MATLAB path where popID is optional.
            info.pop("popID", None)
            info.pop("num_cell", None)

    info["tau_s"] = tau_s
    info["fig"] = 1

    if "TM" not in info:
        raise KeyError("info.TM not found in loaded data")

    print(f"TM size: {info['TM'].shape[0]} x {info['TM'].shape[1]}\n")

    if fig_switch_beh:
        info = neuralSPRT_BEH(info, monkey_id, fig_switch_beh)
    if fig_switch_phy:
        if "KM" not in info:
            raise KeyError("Physiology figures requested but info.KM missing")
        if "FRgain" not in info:
            info["FRgain"] = 100.0
        if "delay" not in info:
            info["delay"] = 0.0
        info = neuralSPRT_PHY(info, monkey_id, fig_switch_phy)

    if out_dir is None:
        out_dir = here / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, int(info.get("fig", 1))):
        fig = plt.figure(i)
        png_path = out_dir / f"neuralSPRT_monkey{monkey}_fig{i}.png"
        svg_path = out_dir / f"neuralSPRT_monkey{monkey}_fig{i}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")

    return info


def _parse_int_list(values: List[str] | None) -> List[int]:
    if not values:
        return []
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Python reproduction of neuralSPRT.m")
    parser.add_argument("--id", type=int, default=2, choices=[1, 2], help="1=Monkey E, 2=Monkey J")
    parser.add_argument("--beh", nargs="*", default=["1", "2", "3", "4", "5", "6"], help="Behavior figure switches")
    parser.add_argument("--phy", nargs="*", default=["1", "2", "3", "4"], help="Physiology figure switches")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataE.mat/dataJ.mat")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    neuralSPRT(
        monkey_id=args.id,
        fig_switch_beh=_parse_int_list(args.beh),
        fig_switch_phy=_parse_int_list(args.phy),
        data_dir=data_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
