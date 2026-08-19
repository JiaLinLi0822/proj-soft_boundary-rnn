from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuralSPRT import neuralSPRT


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures" / "fig2e_paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run ONLY behavior switch 5 to match paper Fig.2E conditions.
    info = neuralSPRT(
        monkey_id=1,
        fig_switch_beh=[5],
        fig_switch_phy=[],
        out_dir=out_dir,
    )

    # The produced figure index is 1 in this dedicated run.
    fig = plt.figure(1)
    fig.savefig(out_dir / "Fig2E_monkeyE.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "Fig2E_monkeyE.svg", bbox_inches="tight")

    m = info["meanCumLLR"]
    print("k=2, choice1 meanCumLLR:", float(m[1, 1, 0]))
    print("k=2, choice2 meanCumLLR:", float(m[1, 1, 1]))


if __name__ == "__main__":
    main()
