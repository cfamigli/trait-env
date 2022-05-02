
import numpy as np
from netCDF4 import Dataset, num2date
import gdal
import glob
import pandas as pd
from datetime import date, timedelta
import covplotlib as cpl

def loc_year(df):
    # drop rows without location or year
    df = df[(df['Lon'].notna()) & (df['Lat'].notna()) & (df['SamplingDateStr'].notna())]
    return df


def reset_lon(nlat, nlon, breakpoint, type='ERA5'):
    # bisect longitude and swap so indices go from 0:180, -180:0 instead of 0:360

    if type=='ERA5':
        lat_lst = [round(la,1) for la in np.linspace(90, -90, nlat)]
        lon_lst = [round(lo,1) for lo in np.linspace(0, 359.9, nlon)]

        for i in range(len(lon_lst)):
            if i >= breakpoint:
                lon_lst[i] = round(lon_lst[i]-360.,1)

    elif type=='SoilGrids':
        lat_lst = [round(la,1) for la in np.linspace(90, -89.9, nlat)]
        lon_lst = [round(lo,1) for lo in np.linspace(-180, 179.9, nlon)]

    elif type=='Simard':
        lat_lst = [round(la,3) for la in np.linspace(90, -89.9, nlat)]
        lon_lst = [round(lo,3) for lo in np.linspace(-180, 179.9, nlon)]

    elif type=='Globcover':
        lat_lst = [round(la,1) for la in np.linspace(90, -89.9, nlat)]
        lon_lst = [round(lo,1) for lo in np.linspace(-180, 179.9, nlon)]

    elif type=='community_means':
        lat_lst = [round(la,1) for la in np.linspace(90, -89.9, nlat)]
        lon_lst = [round(lo,1) for lo in np.linspace(-180, 179.9, nlon)]

    return lat_lst, lon_lst


def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def latlon_to_idx(lat_lst, lon_lst, lat, lon):
    # find indices of trait lat lon in nc row col

    lat_idx = (np.abs(np.asarray(lat_lst) - my_ceil(float(lat), precision=1))).argmin()
    lon_idx = (np.abs(np.asarray(lon_lst) - my_floor(float(lon), precision=1))).argmin()

    return lat_idx, lon_idx


def calc_vpd(TaK, TdK):

    TdC = TdK - 273.15
    TaC = TaK - 273.15

    ea = 6.1078 * np.exp( (17.269 * TdC) / (237.3 + TdC))
    es = 6.1078 * np.exp( (17.269 * TaC) / (237.3 + TaC))

    return es - ea # VPD in mb

def extract_nc_obs(nc_dir, lat_idx, lon_idx):
    # extract time series at lat idx, lon idx

    nc_files = glob.glob(nc_dir + '*.nc')
    nc_files.sort()

    varnames = [file.split('/')[-1].split('.nc')[0] for file in nc_files]

    df = pd.DataFrame(columns = ['hours', 'date'] + varnames + ['vpd'])
    for i, (file, varname) in enumerate(zip(nc_files, varnames)):

        data = Dataset(file, 'r')

        x = np.copy(data[varname][:,lat_idx,lon_idx])
        x[x==-32767] = np.nan

        df[varname] = x

        if i==0:

            hours = data.variables['time'][:]
            df['hours'] = hours

            base = date(1900, 1, 1)
            for hour in hours: base + timedelta(hours = int(hour))
            df['date'] = pd.to_datetime(df['hours'].apply(lambda x: base + timedelta(hours=int(x))))

        data.close()

    df['vpd'] = calc_vpd(df['t2m'], df['d2m'])

    return df

def standardize_year(SamplingDateStr):

    years = [str(year) for year in np.arange(1950, 2022).tolist()]

    for year in years:
        if year in SamplingDateStr:
            return year

        elif year[2:]==SamplingDateStr.split('/')[-1]:
            return year

    return str(np.nan)


def get_preceding_climate(climate_df, obs_year, year_lag):

    include = climate_df[(climate_df['date'].dt.year <= int(obs_year)) & (climate_df['date'].dt.year >= int(obs_year) - year_lag)]
    #cpl.plot_ts(include, ['pev', 'ssrd', 'swvl1', 't2m', 'tp', 'vpd'])
    include = include.drop(['hours', 'date'], axis=1)

    metrics = pd.DataFrame(columns=include.columns)

    metrics.loc['mean'] = include.mean()
    metrics.loc['stddev'] = include.std()
    metrics.loc['p5'] = include.quantile(0.05)
    metrics.loc['p95'] = include.quantile(0.95)

    return metrics

def set_column_names_for_output(nc_dir):

    nc_files = glob.glob(nc_dir + '*.nc')
    nc_files.sort()

    varnames = [file.split('/')[-1].split('.nc')[0] for file in nc_files] + ['vpd']

    varnames_metrics = [[varname+'_mean', varname+'_std', varname+'_p5', varname+'_p95'] for varname in varnames]
    varnames_metrics = [item for sublist in varnames_metrics for item in sublist]

    return varnames_metrics

def main():

    version = '_v1.4'

    trait_data_processed = loc_year(pd.read_csv('../data/try-processed/community_mean_df'+version+'.csv', encoding="ISO-8859-1"))
    #trait_data_processed = community_means(trait_data_processed, [3117, 3116, 3115, 186, 14])
    print(trait_data_processed)

    climate_dir = '../../ERA5/unprocessed/'
    climate_lat_lst, climate_lon_lst = reset_lon(nlat=1801, nlon=3600, breakpoint=1800, type='ERA5')

    soil_dir = '../../soilgrids/sg_orig/'
    soil_data = pd.read_csv(soil_dir + 'sg_processed.csv')
    soil_vars = soil_data.columns[2:]

    canopy_dir = '../../soilgrids/ch/'
    canopy_data = pd.read_csv(canopy_dir + 'ch_processed.csv')
    canopy_vars = canopy_data.columns[2:]

    globcover_dir = '../../Globcover2009_V2/trait-env/'
    globcover_data = pd.read_csv(globcover_dir + 'globcover_processed.csv')
    globcover_vars = globcover_data.columns[2:]

    year_lags = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for year_lag in year_lags:

        xy_by_experiment = pd.DataFrame(columns = list(trait_data_processed.columns) + set_column_names_for_output(climate_dir) + list(soil_vars) + list(canopy_vars) + list(globcover_vars))

        for i, (index, row) in enumerate(trait_data_processed.iterrows()):

            if np.mod(i,100)==0: print(i)

            obs_year = standardize_year(str(row['SamplingDateStr']))

            if (obs_year=='nan'): continue

            climate_lat_idx, climate_lon_idx = latlon_to_idx(climate_lat_lst, climate_lon_lst, row['Lat'], row['Lon'])
            climate_data_for_point = extract_nc_obs(climate_dir, climate_lat_idx, climate_lon_idx)

            try:
                soil_data_for_point = soil_data[(soil_data['Lat']==my_ceil(float(row['Lat']), precision=1)) & (soil_data['Lon']==my_floor(float(row['Lon']), precision=1))][soil_vars].values[0]
            except:
                soil_data_for_point = [np.nan, np.nan, np.nan, np.nan, np.nan]

            try:
                canopy_data_for_point = canopy_data[(canopy_data['Lat']==my_ceil(float(row['Lat']), precision=1)) & (canopy_data['Lon']==my_floor(float(row['Lon']), precision=1))][canopy_vars].values[0]
            except:
                canopy_data_for_point = [np.nan]

            try:
                globcover_data_for_point = globcover_data[(globcover_data['Lat']==my_ceil(float(row['Lat']), precision=1)) & (globcover_data['Lon']==my_floor(float(row['Lon']), precision=1))][globcover_vars].values[0]
            except:
                globcover_data_for_point = [np.nan]

            if climate_data_for_point['t2m'].isnull().all(): continue

            else:
                metrics = get_preceding_climate(climate_data_for_point, obs_year, year_lag)

                row['SamplingDateStr'] = obs_year
                new_row = np.concatenate([row.values, metrics.to_numpy().T.flatten(), soil_data_for_point, canopy_data_for_point, globcover_data_for_point])

                xy_by_experiment.loc[i] = new_row

        xy_by_experiment.to_csv('../data/experiments/exp' + str(year_lag) +version+'.csv')

    return

if __name__=='__main__':
    main()
