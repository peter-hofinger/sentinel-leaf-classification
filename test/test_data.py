import numpy as np
import pytest
import rasterio
from slc.data import combine_band_name, download_dlt_2018, split_band_name


@pytest.fixture(name="reference_path")
def fixture_reference_path(tmp_path):
    reference = np.zeros((10, 20))
    reference_file = tmp_path / "reference.tif"
    with rasterio.open(
        reference_file,
        "w",
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype=reference.dtype,
        crs="EPSG:32633",
        transform=rasterio.transform.from_origin(320000, 5300000, 100, 100),
    ) as dst:
        dst.write(reference, 1)

    return str(reference_file)


def test_combine_band_name():
    composite_idx = 1
    band_label = "TCI_G"
    reducer = "kendallsCorrelation"
    reducer_band = "p-value"
    combined = combine_band_name(composite_idx, band_label, reducer, reducer_band)
    assert combined == "1 TCI_G kendallsCorrelation p-value"


def test_split_band_name():
    band_name = "1 TCI_G kendallsCorrelation p-value"
    a, b, c, d = split_band_name(band_name)
    assert a == 1
    assert b == "TCI_G"
    assert c == "kendallsCorrelation"
    assert d == "p-value"


def test_download_dlt_2018(tmp_path, reference_path):
    destination_path = tmp_path / "dlt.tif"
    download_dlt_2018(reference_path, str(destination_path))

    # Check if all values are 0, 1 and NaN
    assert destination_path.exists()
    with rasterio.open(destination_path) as src:
        data = src.read()
        assert np.any(data == 0)
        assert np.any(data == 1)
        assert np.any(data)
        assert np.unique(data).shape == (3,)

        # Check if the metadata is correct
        assert src.crs.to_epsg() == 32633
        assert src.count == 1
        assert src.width == 20
        assert src.height == 10
