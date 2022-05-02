
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
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

def filter_rows_by_lc(df, lc_na_val):
    sub_df = df[df['globcover_merged']!=lc_na_val]
    return sub_df

def filter_columns_by_metric(df, metric, predictor_strs):

    if metric=='a':

        substrs = ['mean', '10km']#, 'Simard']

    elif metric=='b':

        substrs = ['mean', 'std', 'p5', 'p95', '10km']#, 'Simard']

    mask = [col for col in df.columns if any(s in col for s in substrs) and 'd2m' not in col]
    sub_df = df[mask]
    if len(predictor_strs) <= 2:
        predictor_strs.append([col.split('_')[0] if len(col.split('_'))>2 else col for col in sub_df.columns] + ['intercept'])

    return sub_df

def scale(X):
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    return scaled_X

def filter_rows_by_nan(X, y, sample_weights=None):
    bad_inds = X.isnull().any(axis=1) | y.isnull()

    X = scale(X[~bad_inds])
    y = y[~bad_inds]

    if sample_weights is not None:
        sample_weights = sample_weights[~bad_inds]

    return X, y, sample_weights

def id_to_str(trait):
    switcher = {
        3115: 'SLA, petiole excl.',
        3116: 'SLA, petiole incl.',
        3117: 'SLA, petiole undef.',
        186: 'V$_{c,max}$',
        14: 'Leaf N content',
        4: 'Wood density'
    }
    return switcher.get(trait)

def gc_orig_to_merge(lc_val):
    switcher = {
        11: (11, 'cropland'),
        14: (11, 'cropland'),
        20: (20, 'shrubland'),
        30: (30, 'grassland'),
        40: (40, 'ebf'),
        50: (50, 'dbf'),
        60: (50, 'dbf'),
        70: (70, 'enf'),
        90: (70, 'enf'),
        100: (100, 'mf'),
        110: (20, 'shrubland'),
        120: (30, 'grassland'),
        130: (20, 'shrubland'),
        140: (140, 'savanna'),
        150: (200, 'na'),
        160: (160, 'wetland'),
        170: (160, 'wetland'),
        180: (160, 'wetland'),
        190: (200, 'na'),
        200: (200, 'na'),
        210: (200, 'na'),
        220: (200, 'na'),
        230: (200, 'na')
    }
    return switcher.get(lc_val)[0]

def feature_selection_by_corr(X, y, thresh):

    df = pd.concat([X, y], axis=1)
    corr = df.corr()['StdValue'].sort_values(ascending=False)[1:]
    abs_corr = abs(corr)
    relevant_features = abs_corr[abs_corr>thresh]
    inds = [df.columns.get_loc(c) for c in relevant_features.index if c in df]

    return X[relevant_features.index], inds

def plot_scatter(y_true, y_pred, ax, ax_row, ax_col, year_lag, metric, trait, colorby=None):
    # scatter

    if metric=='a':
        color = 'cornflowerblue'
        marker = 'o'

    elif metric=='b':
        color = 'tomato'
        marker = '^'
        ax[ax_row, ax_col].plot([0,1],[0,1], transform=ax[ax_row, ax_col].transAxes, c='k', linewidth=0.75)
        ax[ax_row, ax_col].set_xlabel(id_to_str(trait) + '\n(observed)')
        ax[ax_row, ax_col].set_ylabel(id_to_str(trait) + '\n(predicted)')

        if ax_row==0:
            ax[ax_row, ax_col].set_title(str(year_lag) + '-year lag',fontweight='bold')


    if colorby is not None:
        sc = ax[ax_row, ax_col].scatter(y_true, y_pred, s=45, marker=marker, c=colorby, edgecolor='k',  linewidth=0.5, alpha=0.95, cmap=plt.cm.gist_ncar)

        if metric=='b':
            bar = plt.colorbar(sc, ax=ax[ax_row, ax_col])

    else:
        ax[ax_row, ax_col].scatter(y_true, y_pred, s=45, facecolor=color, edgecolor='k',  linewidth=0.5, alpha=0.95)

    return

def plot_errors_by_year(y_true, y_pred, year, ax, ax_row, ax_col, year_lag, metric, trait):
    # scatter

    if metric=='a':
        color = 'cornflowerblue'
        offset = -0.2

    elif metric=='b':
        color = 'tomato'
        ax[ax_row, ax_col].set_xlabel('Sampling year')
        ax[ax_row, ax_col].set_ylabel(id_to_str(trait) + '\nprediction error')
        offset = 0.2

        if ax_row==0:
            ax[ax_row, ax_col].set_title(str(year_lag) + '-year lag',fontweight='bold')

    errs = np.array([abs(yt - yp) for yt, yp in zip(y_true, y_pred)])

    err_bar = []
    for yr in np.unique(year):
        err_bar.append(np.median(errs[year==yr]))

    ax[ax_row, ax_col].scatter(year, errs, s=20, facecolor=color, edgecolor='w', alpha=0.3, zorder=0)
    ax[ax_row, ax_col].scatter(np.unique(year), err_bar, s=40, facecolor=color, edgecolor='k',  linewidth=0.5, alpha=0.95, zorder=1)
    #ax[ax_row, ax_col].set_yscale('log')

    return

def plot_coefs(df, year_lags, metric, predictor_strs, saveloc, savename):

    for index, row in df.iterrows():

        coefs = []
        for i in year_lags:

            coefs.append(row['coefs_exp'+str(i)+metric])

        vm = max([abs(np.nanmax(np.array(coefs))), abs(np.nanmin(np.array(coefs)))])

        plt.figure(figsize=(int(len(coefs[0])/2),len(year_lags)/2))
        plt.imshow(coefs, cmap='RdYlBu', vmax=0.9*vm, vmin=-0.9*vm)
        plt.yticks(range(len(year_lags)), [str(year_lag)+'-year lag' for year_lag in year_lags])
        plt.xticks(range(len(coefs[0])), predictor_strs, rotation=90)#['f'+str(c) for c in range(len(coefs[0]))])
        bar = plt.colorbar()
        bar.set_label('Regression coefficient value')

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

        plt.figure(figsize=(3.5,len(year_lags)/2))
        x = np.arange(len(year_lags))
        plt.bar(x-0.2, np.array(scores[::2]), width=0.3, color='cornflowerblue', edgecolor='k')
        plt.bar(x+0.2, np.array(scores[1::2]), width=0.3, color='tomato', edgecolor='k')
        plt.ylabel('R$^2$')
        plt.xticks(range(len(year_lags)), [str(year_lag)+'-year lag' for year_lag in year_lags], rotation=90)
        plt.tight_layout()
        plt.savefig(saveloc + str(row['trait']) + savename + '.pdf')
        plt.close()


def main():

    experiment_dir = '../data/experiments/'
    version = '_v1.4'
    do_rfe = True

    traits = [3115, 186, 14, 4] #[3117, 3116,
    year_lags = [40,35,30,25,20,15,10,5]#[5, 10, 15, 20, 25, 30, 35, 40]#, 50]
    metrics = ['a', 'b']

    results = pd.DataFrame(columns = ['trait'] + ['r2_exp'+str(year_lag)+metric for year_lag in year_lags for metric in metrics] + ['coefs_exp'+str(year_lag)+metric for year_lag in year_lags for metric in metrics])

    figs, axs = plt.subplots(len(traits), len(year_lags), figsize=(len(year_lags)*2.5,len(traits)*2.5))

    fige, axe = plt.subplots(len(traits), len(year_lags), figsize=(len(year_lags)*2.75,len(traits)*2.5))

    predictor_strs, inds_fs_lst = [], []

    for year_lag in year_lags:

        data = filter_rows_by_year_lag(pd.read_csv(experiment_dir + 'exp' + str(year_lag) + version + '.csv'), year_lag)
        data['globcover_merged'] = [gc_orig_to_merge(orig) for orig in data['GLOBCOVER_L4_200901_200912_V2.3_0.1deg']]
        data = filter_rows_by_lc(data, 200)

        t_m_count = 0

        for trait in traits:

            data_for_trait = filter_rows_by_trait(data, trait)

            results.loc[traits.index(trait), 'trait'] = trait

            for metric in metrics:

                X = filter_columns_by_metric(data_for_trait, metric, predictor_strs)
                coefs = np.ones(X.shape[1])*np.nan
                y = data_for_trait['StdValue']

                sample_weights = np.sqrt(data_for_trait['NumMeasurements'].values)# / np.sqrt(data_for_trait['NumMeasurements'].values)
                X, y, sample_weights = filter_rows_by_nan(X, y, sample_weights)
                print(year_lag, trait, metric, len(y), X.shape)
                print(X, y)

                mlr = linear_model.LinearRegression()

                if do_rfe:

                    if year_lag==40:

                        n_features = 6 if metric=='a' else 12
                        mlr = RFE(mlr, n_features_to_select=n_features)
                        mlr.fit(X, y, sample_weight=sample_weights)
                        inds_fs = np.where(mlr.support_)[0]
                        inds_fs_lst.append(inds_fs)

                        coefs[inds_fs] = mlr.estimator_.coef_.reshape(1,-1)[0]
                        results.loc[traits.index(trait), 'coefs_exp'+str(year_lag)+metric] = np.concatenate((coefs, [mlr.estimator_.intercept_]))

                    else:

                        inds_fs = inds_fs_lst[t_m_count]
                        t_m_count += 1
                        print(t_m_count, inds_fs)
                        X = X[:,inds_fs]
                        mlr.fit(X, y, sample_weight=sample_weights)
                        coefs[inds_fs] = mlr.coef_.reshape(1,-1)[0]
                        results.loc[traits.index(trait), 'coefs_exp'+str(year_lag)+metric] = np.concatenate((coefs, [mlr.intercept_]))

                else:

                    inds_fs = np.arange(X.shape[1])
                    mlr.fit(X, y, sample_weight=sample_weights)
                    coefs[inds_fs] = mlr.coef_.reshape(1,-1)[0]
                    results.loc[traits.index(trait), 'coefs_exp'+str(year_lag)+metric] = np.concatenate((coefs, [mlr.intercept_]))

                plot_scatter(y, mlr.predict(X), ax=axs, ax_row=traits.index(trait), ax_col=year_lags.index(year_lag), year_lag=year_lag, metric=metric, trait=trait)#, colorby=data_for_trait['SamplingDateStr'].to_numpy())

                plot_errors_by_year(y, mlr.predict(X), data_for_trait['SamplingDateStr'].to_numpy(), ax=axe, ax_row=traits.index(trait), ax_col=year_lags.index(year_lag), year_lag=year_lag, metric=metric, trait=trait)

                results.loc[traits.index(trait), 'r2_exp'+str(year_lag)+metric] = mlr.score(X, y)


    print(results)

    figs.tight_layout()
    figs.savefig('../plots/scatters/scatter_by_exp'+version+'_rfe.pdf')
    plt.close()

    fige.tight_layout()
    fige.savefig('../plots/scatters/scatter_errs_by_exp'+version+'_rfe.pdf')
    plt.close()

    for metric in metrics:
        plot_coefs(results, year_lags, metric, predictor_strs[metrics.index(metric)], '../plots/coefs/', version+'_rfe')

    plot_scores(results, year_lags, metrics, '../plots/scores/', version+'_rfe')

    results.to_pickle('../data/outputs/coefs'+version+'.pkl')

    return

if __name__=='__main__':
    main()
