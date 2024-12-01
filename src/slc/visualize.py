"""Module for visualizing raster data, creating plots and saving visualizations as images.

Typical usage example:

    import matplotlib.pyplot as plt
    import pandas as pd
    from ltm.visualize import plot_report

    # Create a DataFrame
    df = pd.DataFrame({
        "A": [1, 2, 4],
        "B": [4, 6, 7],
    })

    # Plot the DataFrame
    plot_report(df, title="My Plot", xlabel="X-axis", ylabel="Y-axis", figsize=(10, 5), marker="o")

    plt.show()
"""

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from typeguard import typechecked

from slc.data import combine_band_name, split_band_name


def _rgb_bands2indices(
    raster: np.ndarray | None = None,
    bands: tuple[str, ...] | None = None,
    rgb_bands: list[str | None] | None = None,
) -> list[str]:
    # Check if bands or raster is provided
    if bands is None and raster is None:
        msg = "Either bands or raster must be provided"
        raise ValueError(msg)

    # Check whether bands exists if rgb_bands is not None
    if rgb_bands is not None and bands is None:
        msg = "bands must not be None if rgb_bands is not None"
        raise ValueError(msg)

    # Check if bands is equal to the number of bands in raster
    if bands is not None and len(bands) != raster.shape[0]:
        msg = f"bands must have same length as number of bands in raster: len(bands)={len(bands)} != raster.shape[0]={raster.shape[0]}"
        raise ValueError(msg)

    # Default bands to tuple of integers if None
    bands = (
        bands
        if bands is not None
        else tuple(str(number) for number in range(raster.shape[0]))
    )

    # Check if rgb_bands has length of three or is None
    if rgb_bands is not None and len(rgb_bands) != 3:  # noqa: PLR2004
        msg = "rgb_bands must be a list of length 3 or None"
        raise ValueError(msg)

    # Fill rgb_bands with bands and maybe None if it is None
    if rgb_bands is None:
        if len(bands) == 1:
            rgb_bands = [bands[0]] * 3
        elif len(bands) == 2:  # noqa: PLR2004
            rgb_bands = [bands[0], bands[1], None]
        else:
            rgb_bands = list(bands[:3])

    # Get the indices of the bands
    rgb_indices = []
    for rgb_band in rgb_bands:
        if rgb_band is not None:
            rgb_indices.append(bands.index(rgb_band))
        else:
            rgb_indices.append(None)

    return rgb_indices


@typechecked
def _raster2rgb(
    raster: np.ndarray,
    bands: tuple[str, ...] | None = None,
    rgb_bands: list[str | None] | None = None,
    *,
    mask_nan: bool = True,
) -> np.ndarray:
    """Create an RGB image from a rasterio raster.

    Args:
        raster:
            Rasterio raster to convert to RGB.
        bands:
            Tuple of band names. Must not be None if rgb_bands is not None. Defaults to None.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.
        mask_nan:
            A boolean whether to mask NaN values. If any band has a NaN value, the whole pixel will be masked. Defaults to True.

    Returns:
        A NumPy array RGB image.

    """
    # Get RGB band indices
    rgb_indices = _rgb_bands2indices(raster, bands, rgb_bands)

    # Create RGB image bands
    rgb_plot = []
    for rgb_idx in rgb_indices:
        if rgb_idx is not None:
            rgb_plot.append(raster[rgb_idx])
        else:
            rgb_plot.append(np.zeros_like(raster[0, :, :]))

    # Stack RGB image bands
    rgb_plot = np.dstack(rgb_plot)

    # Mask NaN values
    if mask_nan:
        nan_mask = np.any(np.isnan(rgb_plot), axis=2)
        rgb_plot[nan_mask] = np.nan

    return rgb_plot


@typechecked
def _plot_timeseries(
    rgb_rasters: list[np.ndarray],
    reducer_title: str,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    # Get num_composites
    num_composites = len(rgb_rasters)

    # Get min and max values for normalization
    min_value = np.nanmin(np.array(rgb_rasters))
    max_value = np.nanmax(np.array(rgb_rasters))

    # Plot the rasters below each other
    fig, axs = plt.subplots(nrows=num_composites, figsize=(10, 10))
    if num_composites == 1:
        axs = np.array([axs])
    fig.tight_layout()
    for i, (ax, rgb_raster) in enumerate(zip(axs, rgb_rasters, strict=False), start=1):
        # normalize values and apply gamma correction
        display_raster = rgb_raster - min_value
        display_raster /= max_value - min_value
        display_raster **= 1 / 2.2
        display_raster[np.isnan(display_raster)] = 0
        ax.imshow(display_raster)
        ax.set_title(f"{i} {reducer_title}")
        ax.axis("off")

    plt.show()

    return fig, axs


@typechecked
def show_timeseries(
    raster_path: str,
    reducer: str,
    rgb_bands: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Show a timeseries of composites.

    Args:
        raster_path:
            A string representing the file path to the raster.
        reducer:
            A string representing the reducer to use when creating the composite.
        rgb_bands:
            A list of strings representing the bands to use for the RGB composite. If None the first bands are used. Defaults to None.

    Returns:
        A tuple of the matplotlib figure and axes.

    """
    # Read raster and get band names
    with rasterio.open(raster_path) as src:
        raster = src.read()
        bands = src.descriptions

    # Find the reducer title and the number of composites
    num_composites = 0
    for band in bands:
        composite_idx, _, curr_reducer, _ = split_band_name(band)
        if curr_reducer.lower() == reducer.lower():
            reducer = curr_reducer
            num_composites = max(num_composites, composite_idx)
    bands = [band.lower() for band in bands]

    if num_composites == 0:
        msg = f"No bands found with reducer {reducer}"
        raise ValueError(msg)

    # Get the rasters for the rgb bands
    rgb_rasters = []
    for i in range(1, num_composites + 1):
        rgb_raster = []
        if rgb_bands is not None:
            curr_rgb_bands = []
            for band in rgb_bands:
                if band is None:
                    curr_rgb_bands.append(None)
                else:
                    band_lower = combine_band_name(
                        i, band=band, reducer=reducer
                    ).lower()
                    curr_rgb_bands.append(band_lower)
        else:
            curr_rgb_bands = None

        rgb_raster = _raster2rgb(raster, tuple(bands), curr_rgb_bands)
        rgb_rasters.append(rgb_raster)

    # Plot the rasters
    return _plot_timeseries(rgb_rasters, reducer)


@typechecked
def plot_report(  # noqa: PLR0913
    df: pd.DataFrame,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label_rotation: int = 0,
    replace_labels: dict | None = None,
    *,
    categorical_x: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> plt.Axes:
    """Plot a DataFrame with a title, x and y label and legend.

    The index of the DataFrame is used for the x-axis and the columns for the lines in the plot. The legend is placed to the right of the plot.

    Args:
        df:
            DataFrame to plot.
        title:
            Title of the plot. Defaults to None.
        xlabel:
            Label for the x-axis. Defaults to None.
        ylabel:
            Label for the y-axis. Defaults to None.
        label_rotation:
            Rotation of the x-axis labels. Defaults to 0.
        replace_labels:
            Dictionary to replace labels in the legend. Works also for replacing a subset of the labels. Defaults to None.
        categorical_x:
            Whether the x-axis is categorial and thus each index is an xtick. Not recommended for very long DataFrames. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to df.plot(), e.g. figsize=(10, 5) or marker="o".

    Returns:
        Axes of the plot.

    """
    # Create empty dictionary if replace is None
    if replace_labels is None:
        replace_labels = {}

    # Plot
    ax = df.plot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set xticks
    if categorical_x:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=label_rotation)
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)

    # Replace labels
    labels = ax.get_legend_handles_labels()[1]
    for i, label in enumerate(labels):
        if label in replace_labels:
            labels[i] = replace_labels[label]

    # Set legend
    golden_ratio = (1 + 5**0.5) / 2
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 1 / golden_ratio))

    return ax


@typechecked
def label_subplots(
    axs: np.ndarray | np.flatiter, y_offset: float | list[float] = 0
) -> None:
    """Label subplots with the alphabet.

    Args:
        axs:
            All Axes to label.
        y_offset:
            Label offset along the y-axis. Optionally per axes. Defaults to 0.

    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(axs):
        label = f"({alphabet[i]})"

        offset = -y_offset[i] if isinstance(y_offset, list) else -y_offset
        ax.text(0.5, offset, label, transform=ax.transAxes, va="top", ha="center")


@typechecked
def fig2array(fig: plt.Figure | None = None) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array.

    Args:
        fig:
            Matplotlib figure to convert. Defaults to None, which will use the current figure.

    Returns:
        Numpy array of the figure.

    """
    with io.BytesIO() as buff:
        if fig is None:
            fig = plt.gcf()
        fig.savefig(buff, format="png")
        buff.seek(0)
        return plt.imread(buff)


@typechecked
def save_fig(
    fig: plt.Figure,
    file_path: str | Path,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Save a matplotlib figure to a file.

    Args:
        fig:
            Matplotlib figure to save.
        file_path:
            File path to save the figure to.
        **kwargs:
            Additional keyword arguments to pass to fig.savefig(). Defaults to dpi=300 and transparent=True.

    """
    defaults = {"dpi": 300, "transparent": True}
    kwargs = {**defaults, **kwargs}

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, **kwargs)
    plt.savefig(file_path)


@typechecked
def plot_rasterio(
    ds_reader: rasterio.io.DatasetReader,
    title: str,
    ignore_mask: pd.Series | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Plot a raster with a legend for the leaf type.

    Args:
        ds_reader:
            The rasterio dataset reader.
        title:
            The title of the plot.
        ignore_mask:
            A Series with the flattened raster of boolean values indicating whether a pixel should be ignored.
        ax:
            The matplotlib axes to plot on. Defaults to None.

    """
    content = ds_reader.read()
    if ignore_mask is not None:
        larix_raster = ignore_mask.to_numpy().reshape(content.shape[1:])
        content[:, larix_raster] = np.nan

    if ax is None:
        fig, ax = plt.subplots()
    ax = show(content, interpolation="nearest", ax=ax)
    ax.set(xticks=[], yticks=[])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xlabel(title)
