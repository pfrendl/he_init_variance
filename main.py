import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(gain * torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(input=x, weight=self.weight, bias=self.bias)


def fill_axes(
    data_arr: list[Tensor],
    ax: plt.Axes,
) -> None:
    data = torch.stack(data_arr, dim=0)
    means = data.mean(dim=1)
    stds = data.std(dim=1)

    ax.plot(means.tolist(), color="orangered", label="Mean")
    ax.fill_between(
        x=np.arange(len(means)),
        y1=(means - stds / 2).detach().numpy(),
        y2=(means + stds / 2).detach().numpy(),
        facecolor="dodgerblue",
        alpha=0.1,
        edgecolor="dodgerblue",
        label="One standard deviation",
    )
    ax.grid(True, which="major", linestyle="--")
    ax.legend()


def test(
    input: Tensor,
    linear_layers: list[Linear],
    nonlinearity_ctr: Callable[[], nn.Module],
    save_dir: Path,
) -> None:
    layers: list[nn.Module] = [linear_layers[0]]
    for layer in linear_layers[1:]:
        layers.append(nn.Sequential(nonlinearity_ctr(), layer))

    for dim in [0, 1]:
        x = input
        means_arr = [x.mean(dim=dim)]
        stds_arr = [x.std(dim=dim)]
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
                means_arr.append(x.mean(dim=dim))
                stds_arr.append(x.std(dim=dim))

        # output layer statistics histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

        fig.suptitle(
            f"Output layer statistics, reduced dim={dim}, nonlinearity={nonlinearity_ctr.__name__}"
        )

        ax1.set_xlabel("Mean")
        ax1.set_ylabel("Number of features")
        ax1.hist(
            x=means_arr[-1].tolist(),
            bins=100,
            color="dodgerblue",
        )
        ax1.grid(True, linestyle="--")

        ax2.set_xlabel("Standard deviation")
        ax2.hist(
            x=stds_arr[-1].tolist(),
            bins=100,
            color="dodgerblue",
        )
        ax2.grid(True, linestyle="--")

        plt.savefig(
            save_dir / f"histogram_dim{dim}_{nonlinearity_ctr.__name__}",
            bbox_inches="tight",
        )
        plt.close(fig)

        # statistics by network depth line plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        fig.suptitle(
            f"Statistics by network depth, reduced dim={dim}, nonlinearity={nonlinearity_ctr.__name__}"
        )

        ax1.set_xlabel("Layers applied")
        ax1.set_ylabel("Distribution of means")
        fill_axes(data_arr=means_arr, ax=ax1)

        ax2.set_xlabel("Layers applied")
        ax2.set_ylabel("Distribution of standard deviations")
        fill_axes(data_arr=stds_arr, ax=ax2)

        plt.savefig(
            save_dir / f"plot_dim{dim}_{nonlinearity_ctr.__name__}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    batch_size = 1024
    num_features = 1024
    depth = 1000
    save_dir = Path("outputs/")
    save_dir.mkdir(parents=True, exist_ok=True)

    nonlinearity_ctrs = [
        nn.Identity,
        ReLU,
        nn.SELU,
    ]

    input = torch.randn((batch_size, num_features))
    linear_layers = [Linear(num_features, num_features) for _ in range(depth)]

    for nonlinearity_ctr in nonlinearity_ctrs:
        test(
            input=input,
            linear_layers=linear_layers,
            nonlinearity_ctr=nonlinearity_ctr,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
