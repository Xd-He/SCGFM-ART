from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def save_bases_heatmap(
    bases: torch.Tensor,
    path: str | Path,
    epoch: int,
) -> Path:
    """Save every geometric basis in one figure with a shared [0, 1] scale."""
    values = bases.detach().float().cpu().numpy()
    if values.ndim != 3:
        raise ValueError(
            f"Expected bases with shape [K, M, M], got {values.shape}."
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    num_bases = values.shape[0]
    columns = max(1, math.ceil(math.sqrt(num_bases)))
    rows = math.ceil(num_bases / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 3.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for index, axis in enumerate(axes.flat):
        if index >= num_bases:
            axis.set_visible(False)
            continue
        image = axis.imshow(
            values[index],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"Base {index + 1:02d}")
        axis.set_xticks([])
        axis.set_yticks([])

    if image is not None:
        figure.colorbar(
            image,
            ax=[axis for axis in axes.flat if axis.get_visible()],
            shrink=0.85,
            label="normalized basis value",
        )
    figure.suptitle(f"Geometric bases at epoch {epoch}")
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


