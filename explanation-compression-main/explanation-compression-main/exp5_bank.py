import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from goodpoints import kt, compress
from scipy.stats import wasserstein_distance

from sklearn.model_selection import train_test_split
import xgboost
import shap
import dalex as dx

from tqdm import tqdm

import utils, tictoc

t = tictoc.TicToc(print_toc=False)

## get data

data = pd.read_csv("data/bank-clean-nocat.csv")
X_raw, y_raw = data.drop("y_yes", axis=1).values, data.y_yes.values

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.40, random_state=123)

n = 4**7
X, y = X_test[0:n, :], y_test[0:n]

## train model

model = xgboost.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    subsample=0.8**3, 
    colsample_bytree=0.8, colsample_bylevel=0.8, colsample_bynode=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)

## shap

shap_exp = shap.explainers.Tree(model, data=X, model_output="probability", check_additivity=False)

t.tic()
shap_sv = shap_exp(X).values
t.toc()
time_sv = t.elapsed

shap_svi = np.absolute(shap_sv).mean(axis=0) # 1d shap importance

## dalex

exp = dx.Explainer(model, X, y, verbose=False)

t.tic()
pvi_ = exp.model_parts(type="ratio", N=None, B=25)
t.toc()
time_pvi = t.elapsed

pvi = pvi_.result.iloc[1:X.shape[1], :].sort_values('variable').dropout_loss # 1d permutational variable importance

most_important_variable = pvi_.result.variable[(X.shape[1])]
variable_splits = {most_important_variable: np.linspace(X[:, int(most_important_variable)].min(), X[:, int(most_important_variable)].max(), num=51)}

t.tic()
pdp_ = exp.model_profile(
    type="partial", center=False,
    N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False)
t.toc()
time_pdp = t.elapsed

pdp = pdp_.result[['_yhat_']].to_numpy()

t.tic()
ale_ = exp.model_profile(
    type="accumulated", center=False,
    N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False)
t.toc()
time_ale = t.elapsed

ale = ale_.result[['_yhat_']].to_numpy()


## experiments

exp_results_summary = pd.DataFrame()

for kernel_name, kernel in {'gaussian': utils.kernel_gaussian}.items():
    for kernel_param in [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]:
        exp_results_details = pd.DataFrame()

        for seed in tqdm(range(50)):

            ## compress

            f_halve = lambda x: kt.thin(
                X=x, 
                m=1, 
                split_kernel=lambda x, y: kernel(x, y, kernel_param), 
                swap_kernel=lambda x, y: kernel(x, y, kernel_param), 
                store_K=True,
                seed=seed
            )

            t.tic()
            id_compressed = compress.compress(X, halve=f_halve, g=2)
            t.toc()
            time_kt = t.elapsed

            np.random.seed(seed)
            id_random = np.random.choice(X.shape[0], size=len(id_compressed))

            X_compressed, y_compressed = X[id_compressed], y[id_compressed]
            X_random, y_random = X[id_random], y[id_random]

            ## data

            wd_compressed = np.sum([wasserstein_distance(X[:, i], X_compressed[:, i]) for i in range(X.shape[1])])
            wd_random = np.sum([wasserstein_distance(X[:, i], X_random[:, i]) for i in range(X.shape[1])])

            mmd_linear_compressed = utils.mmd_linear(X, X_compressed)
            mmd_linear_random = utils.mmd_linear(X, X_random)

            mmd_gaussian_compressed = utils.mmd_gaussian(X, X_compressed, kernel_param)
            mmd_gaussian_random = utils.mmd_gaussian(X, X_random, kernel_param)
            
            ## shap

            shap_exp_compressed = shap.explainers.Tree(model, data=X_compressed, model_output="probability", check_additivity=False)
            t.tic()
            shap_sv_compressed = shap_exp_compressed(X_compressed).values
            t.toc()
            time_sv_compressed = t.elapsed
            shap_svi_compressed = np.absolute(shap_sv_compressed).mean(axis=0) # 1d shap importance compressed

            shap_exp_random = shap.explainers.Tree(model, data=X_random, model_output="probability", check_additivity=False)
            shap_sv_random = shap_exp_random(X_random).values
            shap_svi_random = np.absolute(shap_sv_random).mean(axis=0) # 1d shap importance random

            sv_wd_compressed = np.sum([wasserstein_distance(shap_sv[:, i], shap_sv_compressed[:, i]) for i in range(X.shape[1])])
            sv_wd_random = np.sum([wasserstein_distance(shap_sv[:, i], shap_sv_random[:, i]) for i in range(X.shape[1])])

            ## dalex

            exp_compressed = dx.Explainer(model, X_compressed, y_compressed, verbose=False)
            exp_random = dx.Explainer(model, X_random, y_random, verbose=False)

            t.tic()
            pvi_compressed_ = exp_compressed.model_parts(type="ratio", N=None, B=25)
            t.toc()
            time_pvi_compressed = t.elapsed

            pvi_compressed = pvi_compressed_.result.iloc[1:X.shape[1], :].sort_values('variable').dropout_loss
            pvi_random = exp_random.model_parts(type="ratio", N=None, B=25).result.iloc[1:X.shape[1], :].sort_values('variable').dropout_loss

            t.tic()
            pdp_compressed_ = exp_compressed.model_profile(
                type="partial", center=False,
                N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False)
            t.toc()
            time_pdp_compressed = t.elapsed

            pdp_compressed = pdp_compressed_.result[['_yhat_']].to_numpy()
            pdp_random = exp_random.model_profile(
                type="partial", center=False,
                N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False).result[['_yhat_']].to_numpy()

            t.tic()
            ale_compressed_ = exp_compressed.model_profile(
                type="accumulated", center=False,
                N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False)
            t.toc()
            time_ale_compressed = t.elapsed

            ale_compressed = ale_compressed_.result[['_yhat_']].to_numpy()
            ale_random = exp_random.model_profile(
                type="accumulated", center=False,
                N=None, variables=most_important_variable, variable_splits=variable_splits, verbose=False).result[['_yhat_']].to_numpy()

            ## results

            exp_results_details = pd.concat([exp_results_details, pd.DataFrame({
                'model_performance_random': exp_random.model_performance().result.auc.values[0],
                'model_performance_compressed': exp_compressed.model_performance().result.auc.values[0],
                'wd_random': wd_random,
                'wd_compressed': wd_compressed,
                'wd_diff': wd_random - wd_compressed,
                'mmd_linear_random': mmd_linear_random,
                'mmd_linear_compressed': mmd_linear_compressed,
                'mmd_linear_diff': mmd_linear_random - mmd_linear_compressed,
                'mmd_gaussian_random': mmd_gaussian_random,
                'mmd_gaussian_compressed': mmd_gaussian_compressed,
                'mmd_gaussian_diff': mmd_gaussian_random - mmd_gaussian_compressed,
                'svi_random': np.sum(np.abs(shap_svi / shap_svi.max() - shap_svi_random / shap_svi_random.max())),
                'svi_compressed': np.sum(np.abs(shap_svi / shap_svi.max() - shap_svi_compressed / shap_svi_compressed.max())),
                'svi_diff': np.sum(np.abs(shap_svi / shap_svi.max() - shap_svi_random / shap_svi_random.max())) -\
                     np.sum(np.abs(shap_svi / shap_svi.max() - shap_svi_compressed /  shap_svi_compressed.max())),
                'sv_wd_random': sv_wd_random,
                'sv_wd_compressed': sv_wd_compressed,
                'sv_wd_diff': sv_wd_random - sv_wd_compressed,
                'pvi_random': np.sum(np.abs(pvi - pvi_random)),
                'pvi_compressed': np.sum(np.abs(pvi - pvi_compressed)),
                'pvi_diff': np.sum(np.abs(pvi - pvi_random)) - np.sum(np.abs(pvi - pvi_compressed)),
                'pdp_random': np.sum(np.abs(pdp - pdp_random)),
                'pdp_compressed': np.sum(np.abs(pdp - pdp_compressed)),
                'pdp_diff': np.sum(np.abs(pdp - pdp_random)) - np.sum(np.abs(pdp - pdp_compressed)),
                'ale_random': np.sum(np.abs(ale - ale_random)),
                'ale_compressed': np.sum(np.abs(ale - ale_compressed)),
                'ale_diff': np.sum(np.abs(ale - ale_random)) - np.sum(np.abs(ale - ale_compressed)),
                'time_kt': time_kt,
                'time_sv_diff': time_sv - time_sv_compressed,
                'time_pvi_diff': time_pvi - time_pvi_compressed,
                'time_pdp_diff': time_pdp - time_pdp_compressed,
                'time_ale_diff': time_ale - time_ale_compressed
            }, index=[seed])])

        exp_results_details.to_csv(f'results/exp5_bank_{kernel_name}_{kernel_param}.csv', index=False)

        exp_results_summary = pd.concat([
            exp_results_summary, 
            pd.DataFrame(exp_results_details.mean(axis=0), columns=[f'{kernel_name}_{kernel_param}'])
        ], axis=1)

exp_results_summary.to_csv(f'results/exp5_bank_summary.csv')