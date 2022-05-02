
import numpy as np
import pandas as pd
from predict_mlr import filter_rows_by_year_lag, filter_rows_by_trait, filter_columns_by_metric, filter_rows_by_nan, filter_rows_by_lc, gc_orig_to_merge
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dists(data, data_ctrl, ax, vr, color, lbl):

    data = data[(np.isfinite(data)) & (np.isfinite(data_ctrl))]
    data_ctrl = data_ctrl[(np.isfinite(data)) & (np.isfinite(data_ctrl))]

    sns.kdeplot(data=np.unique(data), shade=True, color=color, label=str(lbl)+'-year lag', ax=ax[0])

    if lbl!=5:
        sns.kdeplot(data=np.unique(data-data_ctrl), shade=True, color=color, label=str(lbl)+' minus 5', ax=ax[1])

    ax[0].set_xlabel(vr)
    ax[0].set_ylabel('density')
    ax[0].legend(loc='best', prop={'size': 7})
    ax[1].set_xlabel(vr+'\ndifference')
    ax[1].set_ylabel('density')
    ax[1].legend(loc='best', prop={'size': 7})

    return

def predict_from_coefs(X, coefs):

    y_pred = np.ones(len(X))*np.nan

    for row in range(X.shape[0]):

        y_pred[row] = np.nansum(X[row,:] * coefs[:-1]) + coefs[-1]

    return y_pred


def main():

    experiment_dir = '../data/experiments/'
    version = '_v1.4'

    year_lags = [5, 10, 15, 20, 25, 30, 35, 40]
    climate_vars = pd.read_csv(experiment_dir + 'exp' + str(year_lags[0]) + version + '.csv').columns

    '''for vr in climate_vars:

        fig, ax = plt.subplots(1, 2, figsize=(6,3))

        colors = plt.cm.rainbow(np.linspace(0,1,len(year_lags)))
        for year_lag in year_lags:

            data_ctrl = filter_rows_by_year_lag(pd.read_csv(experiment_dir + 'exp' + str(year_lags[0]) + version + '.csv'), year_lags[0])[vr]

            data = filter_rows_by_year_lag(pd.read_csv(experiment_dir + 'exp' + str(year_lag) + version + '.csv'), year_lag)[vr]

            plot_dists(data, data_ctrl, ax, vr, color=colors[year_lags.index(year_lag)], lbl=year_lag)

        plt.tight_layout()
        plt.savefig('../plots/dists/'+vr+'.pdf')
        plt.close()'''

    #################################

    traits = [3115, 186, 14, 4]

    metrics = ['a', 'b']

    output = pd.read_pickle('../data/outputs/coefs'+version+'.pkl')

    for trait in traits:

        for metric in metrics:

            fig, ax = plt.subplots(1, 2, figsize=(6,3))

            colors = plt.cm.rainbow(np.linspace(0,1,len(year_lags)))

            for year_lag in year_lags:

                coefs = output[output['trait']==trait]['coefs_exp5'+metric].values[0]

                print('.....')
                print(trait, metric, year_lag)

                data = filter_rows_by_year_lag(pd.read_csv(experiment_dir + 'exp' + str(year_lag) + version + '.csv'), year_lag)
                data['globcover_merged'] = [gc_orig_to_merge(orig) for orig in data['GLOBCOVER_L4_200901_200912_V2.3_0.1deg']]
                data = filter_rows_by_lc(data, 200)
                data_for_trait = filter_rows_by_trait(data, trait)

                X = filter_columns_by_metric(data_for_trait, metric, [])
                y = data_for_trait['StdValue']

                X, y, _ = filter_rows_by_nan(X, y)

                y_pred = predict_from_coefs(X, coefs)

                plot_dists(y_pred, y, ax, str(trait)+'_'+metric, color=colors[year_lags.index(year_lag)], lbl=year_lag)

            plt.tight_layout()
            plt.savefig('../plots/dists/'+str(trait)+'_'+metric+'.pdf')
            plt.close()

    return

if __name__=='__main__':
    main()
