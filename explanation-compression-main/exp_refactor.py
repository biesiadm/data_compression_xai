import pandas as pd
import shap
import dalex as dx
import xgboost as xgb
from sklearn.datasets import load_diabetes

from exp_utils import DataProcessor, Experiment

VERBOSE = False


def experiment_1(no_tests=10, save_path='./results/exp1_diabetes.parquet'):
    X, y = load_diabetes(return_X_y=True)
    data_processor = DataProcessor(X=X, y=y)

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': xgb.XGBRegressor,
        'model_params': {'max_depth': 4, 'subsample': 0.9 ** 3, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9,
                         'colsample_bynode': 0.9, 'alpha': 0.1},
        'shap_class': shap.explainers.Tree,
        'shap_params': {'model_output': "raw"},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_polynomial, save_path=save_path, test_size=4**3)

    return result


def experiment_3(no_tests=10, save_path='./results/exp3_bank.parquet'):
    data = pd.read_csv("data/bank-clean.csv")  # .sample(frac=0.5)
    data_processor = DataProcessor(df=data, target='y_yes')

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': xgb.XGBClassifier,
        'model_params': {'n_estimators': 500, 'max_depth': 5, 'subsample': 0.8**3,
                         'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'colsample_bynode': 0.8,
                         'alpha': 0.1, 'eval_metric': 'logloss', 'use_label_encoder': False},
        'shap_class': shap.explainers.Tree,
        'shap_params': {'model_output': "probability", 'check_additivity': False},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'type': "ratio", 'N': None, 'verbose': VERBOSE},
        'pdp_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_polynomial, save_path=save_path, test_size=4**7)

    return result


experiments = [
    # experiment_1,
    experiment_3
]

for exp in experiments:
    exp()
