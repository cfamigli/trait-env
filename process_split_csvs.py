
import pandas as pd
import numpy as np
import os
import glob
import re
from align_try_covs import my_ceil, my_floor, reset_lon, standardize_year, loc_year

def has_head(i):
    # index of header for read_csv
    if i<1:
        return 0
    else:
        return None

def rows_to_keep_str(df, column_name, keywords):
    # adapted from https://stackoverflow.com/questions/48541444/pandas-filtering-for-multiple-substrings-in-series

    subset_lst = [re.escape(keyword) for keyword in keywords]
    pattern = '|'.join(subset_lst)
    reduced_df = df[df[column_name].str.contains(pattern, case=False)].reset_index(drop=True)

    return reduced_df

def rows_to_keep_num(df, column_name, keynums):
    # adapted from https://stackoverflow.com/questions/48541444/pandas-filtering-for-multiple-substrings-in-series

    reduced_df = df[df[column_name].isin(keynums)].reset_index(drop=True)
    return reduced_df

def read_metadata_csvs(metadata_dir):
    # read all metadata csvs beforehand so they don't have to be opened on every loop iteration

    metadata_files = glob.glob(metadata_dir + '*original.csv')
    metadata_files.sort()
    ids = [metadata_file.split('-')[1] for metadata_file in metadata_files]
    data = [pd.read_csv(metadata_file, header=0, encoding="ISO-8859-1") for metadata_file in metadata_files]
    return ids, data


def community_means(df, trait_lst, resolution=0.1):

    community_mean_df = pd.DataFrame(columns = list(df.columns) + ['NumMeasurements'] + ['NumCommunities'])

    df['RLat'] = [my_ceil(la, precision=1) for la in df['Lat'].values]
    df['RLon'] = [my_floor(lo, precision=1) for lo in df['Lon'].values]
    df['SYear'] = [float(standardize_year(yr)) for yr in df['SamplingDateStr'].values]

    lat_lst, lon_lst = reset_lon(nlat=int(180/resolution), nlon=int(360/resolution), breakpoint=None, type='community_means')

    i = 0
    for lat in lat_lst:
        print(lat)

        for lon in lon_lst:

            plot_records = df[(lat==df['RLat']) & (lon==df['RLon']) & (df['SYear']>1990.)]

            if len(plot_records)>0:

                for trait in trait_lst:

                    trait_plot_records = plot_records[plot_records['TraitID']==trait]

                    if len(trait_plot_records)>0:
                        print(trait_plot_records)

                        for yr_grp_name, yr_grp in trait_plot_records.groupby('SYear'):

                            sp_mean = []
                            for sp_grp_name, sp_grp in yr_grp.groupby('AccSpeciesID'):
                                sp_mean.append(sp_grp['StdValue'].mean())

                            community_mean_df.loc[i] = [np.nan for c in community_mean_df.columns]
                            community_mean_df.loc[i]['TraitID'] = trait
                            community_mean_df.loc[i]['StdValue'] = np.mean(sp_mean)#grp['StdValue'].mean()
                            community_mean_df.loc[i]['Lat'] = lat
                            community_mean_df.loc[i]['Lon'] = lon
                            community_mean_df.loc[i]['SamplingDateStr'] = yr_grp['SYear'].mean() # COME BACK TO THIS
                            community_mean_df.loc[i]['NumMeasurements'] = len(yr_grp)
                            community_mean_df.loc[i]['NumCommunities'] = len(sp_mean)
                            print(community_mean_df.loc[i])

                        i += 1

    return community_mean_df


def main():

    version = '_v1.4'

    data_dir = '../data/19963_24032022083038/'
    data_files = glob.glob(data_dir + '*.csv')
    data_files.sort()

    trait_lst = [3117, 3116, 3115, 186, 14, 4]
    boonman_lst = [1,12,20,33,34,45,50,52,55,65,67,73,75,77,84,87,88,91,94,100,102,105,112,113,114,115,123,131,152,156,159,161,163,165,170,180,181,183,190,193,194,200,205,206,207,209,215,218,226,227,228,230,234,236,239,249,255,256,262,263,265,267,269,270,274,329]
    # list of dataset ids that pass boonman criteria

    appended_data = []
    for i, file in enumerate(data_files):

        print(file)

        data = pd.read_csv(file, header=has_head(i), encoding="ISO-8859-1", usecols=list(range(27)))

        if i==0:
            columns = data.columns # if it's the first csv, save the column names
            print(columns)
        else:
            data.columns = columns # apply those column names to all the other csvs

        rows_filtered = rows_to_keep_num(rows_to_keep_num(data, 'TraitID', trait_lst), 'DatasetID', boonman_lst)

        appended_data.append(rows_filtered)#'DataName', [' SLA','LMA','Vcmax'])) # get rid of extraneous rows without trait values

    appended_data = pd.concat(appended_data).reset_index(drop=True) # convert back to pandas dataframe
    print(appended_data)

    #################################################################################

    ids, metadata = read_metadata_csvs('../metadata/')

    output_df = pd.DataFrame(index=appended_data.index, columns=['DatasetID','Genus','AccSpeciesID','ObsDataID','TraitID','OrigValueStr','OrigUnitStr','StdValue','UnitName','Lon','Lat','SamplingDateStr'])

    for index, row in appended_data.iterrows():

        if np.mod(index,100)==0: print(index)

        trait_id = str(int(row['TraitID']))

        metadata_row = metadata[ids.index(trait_id)][metadata[ids.index(trait_id)]['ObsdataID']==row['ObsDataID']]

        output_df.loc[index] = [row['DatasetID'], row['AccSpeciesName'].split(' ')[0], row['AccSpeciesID'], row['ObsDataID'], row['TraitID'], row['OrigValueStr'], row['OrigUnitStr'], row['StdValue'], row['UnitName'],
            float(metadata_row['Lon'].values[0]), float(metadata_row['Lat'].values[0]), metadata_row['SamplingDate'].values[0]]

        print(output_df.loc[index])

    output_df.to_csv('../data/try-processed/aligned'+version+'.csv')

    output_df = loc_year(pd.read_csv('../data/try-processed/aligned'+version+'.csv', encoding="ISO-8859-1"))
    community_mean_df = community_means(output_df, trait_lst)
    print(community_mean_df)

    community_mean_df.to_csv('../data/try-processed/community_mean_df'+version+'.csv')

    return

if __name__=='__main__':
    main()
