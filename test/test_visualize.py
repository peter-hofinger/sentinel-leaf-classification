import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from slc.visualize import fig2array, plot_report, show_timeseries


def test_fig2array():
    # Create a sample figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

    # Convert the figure to an array
    arr = fig2array(fig)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (fig.canvas.get_width_height()[::-1] + (4,))
    assert np.allclose(
        arr[0, 0],
        [1, 1, 1, 1],  # pylint: disable=unsubscriptable-object
    )
    assert np.allclose(arr[..., 3], 1)  # pylint: disable=unsubscriptable-object


def test_plot_report():
    dummy = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [4, 3, 2, 1],
        }
    )

    ax = plot_report(dummy, title="Test Plot", xlabel="X-axis", ylabel="Y-axis")

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Plot"
    assert ax.get_xlabel() == "X-axis"
    assert ax.get_ylabel() == "Y-axis"
    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[0].get_ydata(), [1, 2, 3, 4])
    assert np.allclose(ax.lines[1].get_ydata(), [4, 3, 2, 1])


def test_show_timeseries(tmp_path):
    num_composites = 3
    reducer = "mean"
    rgb_bands = ["B4", "B3", "B2"]

    band_names = []
    for i in range(1, num_composites + 1):
        band_names += [f"{i} {band} {reducer}" for band in rgb_bands]

    # Create a sample raster file
    raster_path = tmp_path / "raster.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=len(band_names),
        dtype="float32",
    ) as dst:
        dst.write(np.random.default_rng().random((len(band_names), 10, 10)))
        dst.descriptions = band_names

    mpl.use("Agg")
    fig, axes = show_timeseries(str(raster_path), reducer, rgb_bands)

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (num_composites,)
    assert len(fig.axes) == num_composites
    assert fig.axes == list(axes)
