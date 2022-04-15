
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def filter_rows_by_year_lag(df, year_lag):
    # remove rows with nonsensical year_lag
    sub_df = df[(df['SamplingDateStr'].apply(pd.to_numeric)- 40 > 1950) & (df['StdValue']<400)]
    return sub_df

def filter_rows_by_trait(df, trait):
    #
    sub_df = df[df['TraitID']==trait]
    return sub_df

def filter_columns_by_metric(df, metric):

    if metric=='a':

        substrs = ['mean', '10km']#, 'Simard']

    elif metric=='b':

        substrs = ['mean', 'std', 'p5', 'p95', '10km']#, 'Simard']

    mask = [col for col in df.columns if any(s in col for s in substrs) and 'd2m' not in col]
    sub_df = df[mask]

    return sub_df

def scale(X):
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    return scaled_X

def filter_rows_by_nan(X, y, sample_weights):
    bad_inds = X.isnull().any(axis=1) | y.isnull()

    X = scale(X[~bad_inds])
    y = y[~bad_inds]
    sample_weights = sample_weights[~bad_inds]

    return X, y, sample_weights

def id_to_str(trait):
    switcher = {
        3115: 'SLA, petiole excl.',
        3116: 'SLA, petiole incl.',
        3117: 'SLA, petiole undef.',
        186: 'Vcmax',
        14: 'Leaf N content'
    }
    return switcher.get(trait)

def plot_scatter(y_true, y_pred, ax, ax_row, ax_col, year_lag, metric, trait):
    # scatter

    if metric=='a':
        color = 'cornflowerblue'

    elif metric=='b':
        color = 'tomato'
        ax[ax_row, ax_col].plot([0,1],[0,1], transform=ax[ax_row, ax_col].transAxes, c='k', linewidth=0.75)
        ax[ax_row, ax_col].set_xlabel(id_to_str(trait) + '\n(observed)')
        ax[ax_row, ax_col].set_ylabel(id_to_str(trait) + '\n(predicted)')

        if ax_row==0:
            ax[ax_row, ax_col].set_title(str(year_lag) + '-year lag',fontweight='bold')

    ax[ax_row, ax_col].scatter(y_true, y_pred, s=45, facecolor=color, edgecolor='k',  linewidth=0.5, alpha=0.95)

    return

def plot_coefs(df, year_lags, metric, saveloc, savename):

    for index, row in df.iterrows():

        coefs = []
        for i in year_lags:

            coefs.append(row['coefs_exp'+str(i)+metric])

        vm = max([abs(np.max(np.array(coefs))), abs(np.min(np.array(coefs)))])

        plt.figure(figsize=(int(len(coefs[0])/2),2))
        plt.imshow(coefs, cmap='Spectral', vmax=0.9*vm, vmin=-0.9*vm)
        plt.yticks(range(len(year_lags)), [str(year_lag)+'-year lag' for year_lag in year_lags])
        plt.xticks(range(len(coefs[0])), ['f'+str(c) for c in range(len(coefs[0]))])
        bar = plt.colorbar()

        plt.tight_layout()
        plt.savefig(saveloc + str(row['trait']) + '_exp'+metric + savename + '.pdf')
        plt.close()

    return

def plot_scores(df, year_lags, metrics, saveloc, savename):

    for index, row in df.iterrows():

        scores = []
        for i in year_lags:

            for m in metrics:

                scores.append(row['r2_exp'+str(i)+m])

        plt.figure(figsize=(3.5,3.5))
        x = np.arange(len(year_lags))
        plt.bar(x-0.2, np.array(scores[::2]), width=0.3, color='cornflowerblue', edgecolor='k')
        plt.bar(x+0.2, np.array(scores[1::2]), width=0.3, color='tomato', edgecolor='k')
        plt.tight_layout()
        plt.savefig(saveloc + str(row['trait']) + savename + '.pdf')
        plt.close()


def main():

    experiment_dir = '../data/experiments/'
    version = '_v1.3'

    traits = [3115, 186, 14] #[3117, 3116,
    year_lags = [10, 20, 30, 40]#, 50]
    metrics = ['a', 'b']

    results = pd.DataFrame(columns = ['trait'] + ['r2_exp'+str(year_lag)+metric for year_lag in year_lags for metric in metrics] + ['coefs_exp'+str(year_lag)+metric for year_lag in year_lags for metric in metrics])

    figs, axs = plt.subplots(len(traits), len(year_lags), figsize=(11,8))

    for year_lag in year_lags:

        data = filter_rows_by_year_lag(pd.read_csv(experiment_dir + 'exp' + str(year_lag) + version + '.csv'), year_lag)

        for trait in traits:

            data_for_trait = filter_rows_by_trait(data, trait)
            results.loc[traits.index(trait), 'trait'] = trait

            for metric in metrics:

                X = filter_columns_by_metric(data_for_trait, metric)
                y = data_for_trait['StdValue']
                sample_weights = np.sqrt(data_for_trait['NumMeasurements'].values)

                X, y, sample_weights = filter_rows_by_nan(X, y, sample_weights)

                mlr = linear_model.LinearRegression()
                mlr.fit(X, y, sample_weights)

                print(year_lag, trait, metric)

                plot_scatter(y, mlr.predict(X), ax=axs, ax_row=traits.index(trait), ax_col=year_lags.index(year_lag), year_lag=year_lag, metric=metric, trait=trait)

                results.loc[traits.index(trait), 'r2_exp'+str(year_lag)+metric] = mlr.score(X, y)
                results.loc[traits.index(trait), 'coefs_exp'+str(year_lag)+metric] = mlr.coef_.reshape(1,-1)[0]

    print(results)

    figs.tight_layout()
    figs.savefig('../plots/scatters/scatter_by_exp'+version+'.pdf')
    plt.close()

    for metric in metrics:
        plot_coefs(results, year_lags, metric, '../plots/coefs/', version)

    plot_scores(results, year_lags, metrics, '../plots/scores/', version)

    return

if __name__=='__main__':
    main()
