import unittest

import numpy as np
import pandas as pd
from numpy.random import default_rng
from optuna import Study
from skelm import ELMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from slc.models import cv_predict, hyperparam_search

from .test_features import fixture_data_path, fixture_target_path  # noqa: F401


class TestHyperparamSearch(unittest.TestCase):
    def setUp(self):
        classification = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        self.data, self.target = classification
        self.data = pd.DataFrame(self.data)
        self.target = pd.Series(self.target)

        def _suggest_float(*args, **kwargs):
            return "suggest_float", args, kwargs

        def _suggest_categorical(*args, **kwargs):
            return "suggest_categorical", args, kwargs

        self.model = ELMClassifier(random_state=42)
        self.search_space = [
            _suggest_float("alpha", 1e-8, 1e5, log=True),
            _suggest_categorical("include_original_features", [True, False]),
            _suggest_float("n_neurons", 1, 1000),
            _suggest_categorical("ufunc", ["tanh", "sigm", "relu", "lin"]),
            _suggest_float("density", 0.01, 0.99),
        ]
        self.scorer = make_scorer(f1_score)

    def test_hyperparam_search(self):
        elm_model, elm_study = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        assert isinstance(elm_model, Pipeline)
        assert isinstance(elm_study, Study)
        assert isinstance(elm_study.best_params, dict)
        assert elm_study.best_value > 0.5

    def test_reproducible_seed(self):
        elm_model1, elm_study1 = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        elm_model2, elm_study2 = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )
        assert elm_study1.best_params == elm_study2.best_params
        assert elm_model1[-1].get_params() == elm_model2[-1].get_params()

    def test_reproducible_score(self):
        elm_model, elm_study = hyperparam_search(
            self.model,
            self.search_space,
            self.data,
            self.target,
            self.scorer,
            n_trials=1,
            random_state=42,
        )

        k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            elm_model, self.data, self.target, cv=k_fold, scoring=self.scorer, n_jobs=-1
        )

        assert np.isclose(elm_study.best_value, np.mean(cv_scores))


def test_cv_predict(data_path, target_path):
    rng = default_rng(42)
    data, target = rng.random((10 * 20, 2)), rng.random(10 * 20).round().astype(int)
    data = pd.DataFrame(data, columns=["Mean B1", "Mean B2"])

    model = ELMClassifier(random_state=42)
    model.fit(data, target)

    # Test whether same pixels are masked in raster than in target
    prediction = cv_predict(model, data_path, target_path, cv=2)

    assert np.isnan(prediction[0, 0, 0])
    assert prediction.shape == (1, 10, 20)
