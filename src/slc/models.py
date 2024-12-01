"""Performs hyperparameter search using Optuna and offers cross-validated inference on raster data.

Typical usage example:

    from ltm.features import load_raster
    from ltm.models import hyperparam_search
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import make_scorer, root_mean_squared_error

    def suggest_categorical(*args, **kwargs):
        return "suggest_categorical", args, kwargs

    model = RandomForestRegressor()
    search_space = [suggest_categorical("n_estimators", [100, 200, 300])]
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    data = load_raster("data_interpolated.tif")
    target = load_raster("target.tif")
    data, target = data[target.notna()], target[target.notna()]

    best_model, study = hyperparam_search(
        model,
        search_space,
        data,
        target,
        scorer,
    )
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any
from zoneinfo import ZoneInfo

import dill
import numpy as np
import optuna
import pandas as pd
import rasterio
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from typeguard import typechecked

from slc.data import (
    BROADLEAF_AREA,
    CONIFER_AREA,
    list_bands,
    list_indices,
    sentinel_composite,
)
from slc.features import (
    interpolate_data,
    load_raster,
    np2pd_like,
    save_raster,
    to_float32,
)


@typechecked
def _target2raster(
    target: pd.Series | pd.DataFrame,
    indices: np.ndarray,
    plot_shape: tuple[int, int, int],
    *,
    area2mixture: bool = False,
) -> np.ndarray:
    # Create target values array of shape (n_samples, n_features)
    target_values = target.to_numpy()
    if len(target_values.shape) == 1:
        target_values = np.expand_dims(target_values, axis=1)

    # Create raster from target with the help of an indices array
    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = target_values
    raster = raster.reshape(plot_shape[1], plot_shape[2], plot_shape[0]).transpose(
        2, 0, 1
    )

    # Use indices of BROADLEAF_AREA and CONIFER_AREA to compute mixture = broadleaf / (broadleaf + conifer)
    if area2mixture:
        columns = list(target.columns)
        broadleaf_index = columns.index(BROADLEAF_AREA)
        conifer_index = columns.index(CONIFER_AREA)
        broadleaf = raster[broadleaf_index, :, :]
        conifer = raster[conifer_index, :, :]
        raster = broadleaf / (broadleaf + conifer)
        raster = np.expand_dims(raster, axis=0)

    return raster


@typechecked
def _has_nan_error(
    estimator: BaseEstimator,
) -> bool:
    # Checks if estimator raises a ValueError when predicting on NaN
    try:
        estimator.fit([[0]], [0])
        estimator.predict([[np.nan]])
    except ValueError:
        return True
    else:
        return False


@typechecked
def _build_pipeline(
    model: BaseEstimator,
    n_components: int | None,
    model_params: dict[str, Any],
    *,
    do_standardize: bool,
    do_pca: bool,
) -> Pipeline:
    # Set params first, as it can change _has_nan_error
    model = model.set_params(**model_params)

    steps = []
    if _has_nan_error(model) or do_pca:
        steps.append(("imputer", KNNImputer()))
    if do_standardize:
        steps.append(("scaler", StandardScaler()))
    if do_pca:
        if n_components is None:
            msg = "n_components must be set if do_pca is True."
            raise ValueError(msg)
        steps.append(("pca", PCA(n_components=n_components)))

    steps.append(("model", model))

    return Pipeline(steps=steps)


@typechecked
def _study2model(
    study: optuna.study.Study,
    model: BaseEstimator,
    data: pd.DataFrame,
    target: pd.Series,
) -> Pipeline:
    # Define preprocessing steps for best model
    model_params = {
        param: value
        for param, value in study.best_params.items()
        if param not in ["do_standardize", "do_pca", "n_components"]
    }

    n_components = None
    if study.best_params["do_pca"]:
        n_components = study.best_params["n_components"]

    best_model = _build_pipeline(
        model,
        n_components,
        model_params,
        do_standardize=study.best_params["do_standardize"],
        do_pca=study.best_params["do_pca"],
    )

    # Fit best model
    logger.debug("Fitting best model on complete dataset...")
    best_model.fit(data, target)

    return best_model


@typechecked
def _create_paths(
    model: BaseEstimator,
    save_folder: str,
) -> tuple[Path, Path, Path]:
    save_path_obj = Path(save_folder)
    if not save_path_obj.parent.exists():
        msg = f"Directory of save_path does not exist: {save_path_obj.parent}"
        raise ValueError(msg)

    # Check if files already exist
    model_name = model.__class__.__name__
    cache_path = save_path_obj / f"{model_name}_cache.pkl"
    study_path = save_path_obj / f"{model_name}_study.pkl"
    model_path = save_path_obj / f"{model_name}.pkl"

    return study_path, model_path, cache_path


@typechecked
def _check_save_folder(
    model: ClassifierMixin,
    data: pd.DataFrame,
    target: pd.Series,
    save_folder: str | None,
    *,
    use_caching: bool,
) -> tuple[Pipeline, optuna.study.Study] | None:
    # Check for valid save_path
    if save_folder is not None:
        study_path, model_path, _ = _create_paths(model, save_folder)

        if study_path.exists() and model_path.exists():
            logger.info(
                f"Files already exist, skipping search: {study_path}, {model_path}"
            )

            # Load best model and study
            with model_path.open("rb") as file:
                best_model = dill.load(file)  # noqa: S301
            with study_path.open("rb") as file:
                study = dill.load(file)  # noqa: S301

            return best_model, study

        if study_path.exists():
            # Inform user
            logger.info("Creating model from study file...")

            # Load the study and create the best model
            with study_path.open("rb") as file:
                study = dill.load(file)  # noqa: S301
            best_model = _study2model(study, model, data, target)

            # Save best model
            with model_path.open("wb") as file:
                dill.dump(best_model, file)

            return best_model, study

        if model_path.exists():
            # Raise error if model file exists but study file is missing
            msg = f"Study file is missing, please delete the model file manually and rerun the script: {model_path}"
            raise ValueError(msg)
    elif use_caching:
        logger.info("use_caching=True but save_folder=None, caching is disabled.")

    return None


@typechecked
def _save_study_model(
    study: optuna.study.Study,
    best_model: Pipeline,
    study_path: Path,
    model_path: Path,
) -> None:
    # Save study
    with study_path.open("wb") as file:
        dill.dump(study, file)

    # Save best model
    with model_path.open("wb") as file:
        dill.dump(best_model, file)


@typechecked
def _get_composite_params() -> dict[str, int]:
    compositing_path = "../reports/compositing.csv"

    try:
        compositing = pd.read_csv(compositing_path)
    except FileNotFoundError as exc:
        msg = f"File {compositing_path} not found. Please run the compositing notebook first."
        raise FileNotFoundError(msg) from exc

    metric = "F1 Score"

    optimal_idx = compositing.groupby("Reducer")[metric].idxmax()
    optimal_df = compositing.loc[optimal_idx]
    optimal_df = optimal_df.set_index("Reducer")

    composite_dict = optimal_df["Composites"].to_dict()

    # Sort alphabetically
    return dict(sorted(composite_dict.items(), key=lambda x: x[0]))


@typechecked
def _create_composites(
    year: int,
    data_folder: Path,
    target_path: Path,
    batch_size: int | None = None,
    top_n: int | None = None,
) -> None:
    # Get selected bands and indices
    importance_path = "../reports/band_importance.csv"
    if Path(importance_path).exists():
        sentinel_bands, indices = bands_from_importance(importance_path, top_n=top_n)
    else:
        msg = f"File {importance_path} not found. Please run the band importance notebook first."
        raise FileNotFoundError(msg)

    # Get composites by most composites first
    composite_dict = _get_composite_params()
    composite_dict = dict(
        sorted(composite_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # Create one composite for each reducer
    tz = ZoneInfo("UTC")
    iterable = tqdm(composite_dict.items(), desc=f"Downloading Composites for {year}")
    for reducer, num_composites in iterable:
        composite_path = str(
            Path(data_folder) / f"{year}/data_{reducer}_{num_composites}.tif"
        )

        sleep_time = 60
        while not Path(composite_path).exists():
            try:
                sentinel_composite(
                    target_path,
                    composite_path,
                    time_window=(
                        datetime(year, 1, 1, tzinfo=tz),
                        datetime(year + 1, 1, 1, tzinfo=tz),
                    ),
                    num_composites=num_composites,
                    temporal_reducers=[reducer],
                    indices=indices,
                    sentinel_bands=sentinel_bands,
                    batch_size=batch_size,
                )
            except BaseException as e:  # noqa: BLE001, PERF203
                logger.debug(str(e))
                sleep(sleep_time)
                sleep_time *= 2


@typechecked
def bands_from_importance(
    band_importance_path: str,
    top_n: int | None = None,
    *,
    level_2a: bool = True,
) -> tuple[list[str], list[str]]:
    """Extract the band names of Sentinel-2 bands and indices from the band importance file.

    The best/top_n bands are read from the band importance file. Those bands are then divided into Sentinel-2 bands and indices.

    Args:
        band_importance_path:
            Path to the file with band names and their scores. The bands are expected to be in reverse order of when they were removed by RFE. Only the band names and their order are used.
        top_n:
            An optional integer representing the number of bands to keep. Chooses the number of bands leading to the max score if None. Defaults to None.
        level_2a:
            Whether the band importance file is from a level 2A dataset. This is necessary for distinguishing Sentinel-2 bands from indices. Defaults to True.

    Returns:
        Tuple of lists of Sentinel-2 band names and index names as strings.

    """
    # Check path
    if not Path(band_importance_path).exists():
        msg = f"File does not exist: {band_importance_path}"
        raise ValueError(msg)

    # Read band importance file
    band_importance = pd.read_csv(band_importance_path, index_col=0)
    band_names = band_importance.index
    band_importance = band_importance.reset_index(drop=True)

    # Divide bands into Sentinel-2 bands and indices
    if top_n is None:
        logger.info(
            f"Using maximum value in column {band_importance.columns[0]} to determine top_n."
        )
        top_n = band_importance[band_importance.columns[0]].idxmax() + 1

    best_bands = list(band_names[:top_n])
    valid_sentinel_bands = list_bands(level_2a=level_2a)
    valid_index_bands = list_indices()
    sentinel_bands = [band for band in valid_sentinel_bands if band in best_bands]
    index_bands = [band for band in valid_index_bands if band in best_bands]

    # Sanity check
    if len(sentinel_bands) + len(index_bands) != len(best_bands):
        msg = "The sum of Sentinel-2 bands and index bands does not equal the number of best bands. This should not happen..."
        raise ValueError(msg)

    return sentinel_bands, index_bands


@typechecked
def area2mixture_scorer(scorer: _BaseScorer) -> _BaseScorer:
    """Modify the score function of a scorer to use the leaf type mixture calculated from leaf type areas.

    Args:
        scorer:
            A _BaseScorer scorer, e.g. returned by make_scorer().

    Returns:
        Scorer with modified score function.

    """
    score_func = scorer._score_func  # noqa: SLF001

    def _mixture_score_func(
        target_true: pd.Series,
        target_pred: pd.Series,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable:
        # Convert to np.ndarray
        target_true = np.array(target_true)
        target_pred = np.array(target_pred)

        # broadleaf is 0, conifer is 1
        target_true = target_true[:, 0] / (target_true[:, 0] + target_true[:, 1])
        target_pred = target_pred[:, 0] / (target_pred[:, 0] + target_pred[:, 1])

        return score_func(target_true, target_pred, *args, **kwargs)

    scorer._score_func = _mixture_score_func  # noqa: SLF001

    return scorer


@typechecked
def hyperparam_search(  # noqa: C901, PLR0913
    model: ClassifierMixin,
    search_space: list[tuple[str, tuple, dict[str, Any]]],
    data: pd.DataFrame,
    target: pd.Series,
    scorer: _BaseScorer,
    cv: int | BaseCrossValidator = 5,
    n_trials: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    save_folder: str | None = None,
    *,
    use_caching: bool = True,
    always_standardize: bool = False,
) -> tuple[Pipeline, optuna.study.Study]:
    """Perform hyperparameter search for a model using Optuna.

    The search space will be explored using Optuna's TPE sampler, together with standardization and PCA for preprocessing. The best pipeline object will be returned along with the Optuna study. A KNNImputer is used for estimators not supporting NaN values.

    Args:
        model:
            Model to perform hyperparameter search for.
        search_space:
            List of tuples with the name of the method to suggest, the arguments and the keyword arguments. For example [("suggest_float", ("alpha", 1e-10, 1e-1), {"log": True})].
        data:
            Features to use for hyperparameter search.
        target:
            Labels to use for hyperparameter search.
        scorer:
            Scorer to use for hyperparameter search. Greater score is better. Please make sure to set greater_is_better=False when using make_scorer if you want to minimize a metric.
        cv:
            Number of folds or BaseCrossValidator instance to use for cross validation. Defaults to 5.
        n_trials:
            Number of trials to perform hyperparameter search. Defaults to 100.
        n_jobs:
            Number of jobs to use for hyperparameter search. Set it to -1 to maximize parallelization.  Defaults to 1, as otherwise Optuna becomes non-deterministic.
        random_state:
            Integer to be used as random state for reproducible results. Defaults to None.
        save_folder:
            Folder to save the study PKL and model PKL to. Uses model.__class__.__name__ to name the files. Skips search if files already exist. Defaults to None.
        use_caching:
            Whether to use caching for the search. Saves a [model]_cache.pkl for each step and resumes if a cache exists. The cache is deleted after the final study is saved. Defaults to True.
        always_standardize:
            If true the data is always standardized. Recommended for SVM based estimators. Defaults to False.

    Returns:
        Tuple of the best pipeline and the Optuna study.

    """
    result = _check_save_folder(
        model,
        data,
        target,
        save_folder,
        use_caching=use_caching,
    )
    if result is not None:
        return result

    # Create paths
    if save_folder is not None:
        study_path, model_path, cache_path = _create_paths(model, save_folder)

    def _callback(study: optuna.study.Study, _: optuna.trial.FrozenTrial) -> None:
        # Save intermediate study
        if use_caching and save_folder is not None:
            with cache_path.open("wb") as file:
                dill.dump(study, file)

    def _objective(trial: optuna.trial.Trial) -> float:
        # Choose whether to standardize and apply PCA
        standardize_options = [True, False]
        if always_standardize:
            standardize_options = [True]
        do_standardize = trial.suggest_categorical(
            "do_standardize", standardize_options
        )
        do_pca = trial.suggest_categorical("do_pca", [True, False])

        # Build pipeline
        n_components = None
        if do_pca:
            n_splits = cv if isinstance(cv, int) else cv.get_n_splits()
            max_components = min(data.shape) - np.ceil(
                min(data.shape) / n_splits
            ).astype(int)
            n_components = trial.suggest_int("n_components", 1, max_components)

        params = {
            args[0]: getattr(trial, name)(*args, **kwargs)
            for name, args, kwargs in search_space
        }

        nonlocal model
        pipe = _build_pipeline(
            model,
            n_components,
            params,
            do_standardize=do_standardize,
            do_pca=do_pca,
        )

        # Cross validate pipeline
        try:
            k_fold = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
            cv_scores = cross_val_score(
                pipe, data, target, cv=k_fold, scoring=scorer, n_jobs=-1
            )
            return cv_scores.mean()
        # Catch case that all fits fail
        except ValueError:
            logger.info("All fits failed, returning NaN.")
            return np.nan

    if use_caching and save_folder is not None and Path(cache_path).exists():
        # Resume search from cache
        with cache_path.open("rb") as file:
            study = dill.load(file)  # noqa: S301
        n_trials -= len(study.trials)

        logger.info(f"Resuming search from cache at trial {len(study.trials)}.")
    else:
        # Start new search
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_state),
            direction="maximize",
            study_name=model.__class__.__name__,
        )

    # Optimize study
    study.optimize(_objective, callbacks=[_callback], n_trials=n_trials, n_jobs=n_jobs)

    best_model = _study2model(study, model, data, target)

    if save_folder is not None:
        _save_study_model(study, best_model, study_path, model_path)

        # Delete cache
        if use_caching:
            Path(cache_path).unlink()

    return best_model, study


@typechecked
def cv_predict(
    model: BaseEstimator,
    data_path: str | Path,
    target_path: str | Path,
    cv: int | BaseCrossValidator | None = None,
) -> np.ndarray:
    """Predicts on rasters using cross_val_predict.

    Args:
        model:
            Regressor to use for prediction.
        data_path:
            A string or Path with path to the data in GeoTIFF format.
        target_path:
            A string or Path with path to the target data in GeoTIFF format.
        cv:
            An integer for the number of folds or a BaseCrossValidator object for performing cross validation. Will be passed to sklearn.model_selection.cross_val_predict(). Defaults to None.

    Returns:
        A numpy raster of the prediction in the format of read() from rasterio.

    """
    # Make paths Path objects
    data_path = Path(data_path)
    target_path = Path(target_path)

    # Load data and plot shape
    data = load_raster(data_path)
    target = load_raster(target_path)

    with rasterio.open(target_path) as src:
        shape = src.read().shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    mask = target.notna()
    data, target, indices_array = data[mask], target[mask], indices_array[mask]

    # Predict using cross_val_predict
    target_pred = cross_val_predict(model, data, target, cv=cv, n_jobs=-1)
    target_pred = np2pd_like(target_pred, target)

    return _target2raster(target_pred, indices_array, shape)


@typechecked
def create_data(
    year: int,
    target_path: str | Path,
    data_folder: str | Path,
    batch_size: int | None = None,
    top_n: int | None = None,
) -> None:
    """Create a data raster for a given year ready to be used for inference using the Earth Engine API.

    The files band_importance.csv and compositing.csv are expected to be located in "../reports/". The data raster is saved in the data_folder/year folder with the final data raster being saved as data_folder/year/data.tif.

    Args:
        year:
            Year to create data for.
        target_path:
            Path to the target data in GeoTIFF format.
        data_folder:
            Path to the folder to save the data and composite rasters in.
        batch_size:
            Number of samples to download at a time. Defaults to None.
        top_n:
            Number of most important bands to use for the composites. Uses number of bands that lead to max score if None. Defaults to None.

    """
    # Make paths Path objects
    target_path = Path(target_path)
    data_folder = Path(data_folder)

    # Skip if data already exists
    data_path = Path(data_folder) / f"{year}/data.tif"
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Create composites
    _create_composites(year, data_folder, target_path, batch_size, top_n=top_n)

    # Combine into one raster
    total_data = pd.DataFrame()
    composite_dict = _get_composite_params()
    for reducer, num_composites in tqdm(
        composite_dict.items(), desc=f"Combining Composites for {year}"
    ):
        composite_path = str(
            Path(data_folder) / f"{year}/data_{reducer}_{num_composites}.tif"
        )

        data = load_raster(composite_path)
        data = interpolate_data(data)
        data = to_float32(data)
        total_data = pd.concat([total_data, data], axis=1)

    # Save the concatenated data
    save_raster(total_data, target_path, str(data_path))


@typechecked
def rolling_window(
    x: pd.Series,
    data: pd.DataFrame,
    window_width: float,
    function: Callable,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Compute a rolling window over a pandas DataFrame.

    Args:
        x:
            The x-axis values of each row in the DataFrame.
        data:
            The DataFrame containing the data to be windowed.
        window_width:
            The width of the window in the x-axis units.
        function:
            The function to be applied to each window. Must return a dictionary like {"Point Estimate": float, "Lower Bound": float, "Upper Bound": float}.
        n_samples:
            The number of samples to be taken from the x-axis. The number of rows in the resulting DataFrame.

    Returns:
        A DataFrame containing the results of the function applied to each window. The index are the x-axis values of the windows.

    """
    min_sample_size = 2
    if n_samples < min_sample_size:
        msg = "n_samples must be at least 2"
        raise ValueError(msg)

    x = x.reset_index(drop=True)
    data = data.reset_index(drop=True)

    idcs = x.argsort()
    sorted_x = x[idcs]
    data = data.loc[idcs]

    x_values = []
    results = []
    for x_value in np.linspace(sorted_x.min(), sorted_x.max(), n_samples):
        mask = sorted_x.between(x_value - window_width / 2, x_value + window_width / 2)
        data_window = data[mask]
        result = function(data_window)
        x_values.append(x_value)
        results.append(result)

    return pd.DataFrame(results, index=x_values)
