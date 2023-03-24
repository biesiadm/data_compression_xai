import pandas as pd
import shap
import dalex as dx

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from exp_utils import DataProcessor, Experiment
VERBOSE = False

def experiment_5(no_tests=30, save_path='./explanation-compression-main/results/exp_bank_knn_nocat_check.parquet', model_metric='accuracy'):
    data = pd.read_csv("./explanation-compression-main/data/bank-clean-nocat.csv")
    data_processor = DataProcessor(df=data, target='y_yes')

    experiment_settings = {
        'data_processor': data_processor,
        'model_class': KNeighborsClassifier,
        'model_params': {},
        'shap_class': shap.Explainer,
        'is_tree': False,
        'shap_params': {},
        'dalex_class': dx.Explainer,
        'dalex_params': {'verbose': VERBOSE},
        'pvi_params': {'N': None, 'verbose': VERBOSE},
        'pdp_params': {'N': None, 'verbose': VERBOSE},
        'ale_params': {'type': "accumulated", 'center': False, 'N': None, 'verbose': VERBOSE}
    }

    experiment = Experiment(**experiment_settings)
    result = experiment.run(no_tests, Experiment.kernel_gaussian, save_path=save_path,
                            test_size=4 ** 7, model_metric=model_metric)

    return result