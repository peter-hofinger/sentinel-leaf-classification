import numpy as np
import pandas as pd
import pytest
import rasterio
from slc.features import (
    interpolate_data,
    load_raster,
    np2pd_like,
    save_raster,
    to_float32,
)


@pytest.fixture(name="data_path")
def fixture_data_path(tmp_path):
    data = np.random.default_rng().random((2, 10, 20))
    data_file = tmp_path / "data.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(f"Mean B{i+1}" for i in range(data.shape[0]))
    return str(data_file)


@pytest.fixture(name="target_path")
def fixture_target_path(tmp_path):
    target = np.random.default_rng().random((10, 20)).round()
    target[0, 0] = np.nan
    target_file = tmp_path / "target.tif"
    with rasterio.open(
        target_file,
        "w",
        driver="GTiff",
        height=target.shape[0],
        width=target.shape[1],
        count=1,
        dtype=target.dtype,
    ) as dst:
        dst.write(target, 1)

    return str(target_file)


def test_np2pd_like():
    dummy = pd.DataFrame(
        {
            "B1": [0],
            "B2": [0],
            "B3": [0],
        }
    )

    data = np.zeros((20, 3))
    data_pd = np2pd_like(data, dummy)
    assert isinstance(data_pd, pd.DataFrame)
    assert data_pd.shape == (20, 3)
    assert data_pd.columns[0] == "B1"
    assert data_pd.columns[1] == "B2"
    assert data_pd.columns[2] == "B3"


def test_load_data(data_path):
    data = load_raster(data_path)
    band_names = list(data.columns)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (200, 2)
    assert len(band_names) == 2
    assert band_names[0] == "Mean B1"


def test_load_target(target_path):
    target = load_raster(target_path)
    assert isinstance(target, pd.Series)
    assert len(target) == 200


def test_save_raster(target_path, tmp_path):
    data = np.zeros((10, 20)).flatten()
    columns = ["B1 mean"]
    data_df = pd.DataFrame(data, columns=columns)

    source_path = target_path
    destination_path = str(tmp_path / "data.tif")
    save_raster(data_df, source_path, destination_path)

    with rasterio.open(destination_path) as src:
        assert src.width == 20
        assert src.height == 10
        assert src.count == 1
        assert src.read(1)[0, 0] == 0


def test_interpolate_data():
    data = pd.DataFrame(
        {
            "1 B1 mean": [0],
            "1 B2 mean": [0],
            "1 B3 mean": [0],
            "2 B1 mean": [2],
            "2 B2 mean": [2],
            "2 B3 mean": [2],
            "3 B1 mean": [np.nan],
            "3 B2 mean": [np.nan],
            "3 B3 mean": [np.nan],
        }
    )
    data_interpolated = interpolate_data(data)
    assert data_interpolated["3 B1 mean"][0] == 1


def test_to_float32():
    float32_max = np.finfo(np.float32).max
    float64_max = np.finfo(np.float64).max

    # Check with string
    data = pd.DataFrame(
        {
            "1 B1 mean": [float64_max],
            "1 B2 mean": ["abc"],
            "1 B3 mean": [0],
        }
    )
    with pytest.raises(
        ValueError, match="Expected numeric data, found object instead."
    ):
        to_float32(data)

    data["1 B2 mean"] = [1]
    data_float32 = to_float32(data)

    assert data_float32["1 B1 mean"].dtype == np.float32
    assert data_float32["1 B1 mean"][0] == float32_max
    assert data_float32["1 B2 mean"].dtype == np.float32
    assert data_float32["1 B3 mean"].dtype == np.float32
