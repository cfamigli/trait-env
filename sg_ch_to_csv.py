
import gdal
import numpy as np
from pandas import DataFrame
import glob
import sys
from align_try_covs import reset_lon, latlon_to_idx

def tif_to_arr(tif_dir):

    # open raster file with gdal
    raster_files = glob.glob(tif_dir + '*.tif')
    raster_files.sort()

    varnames = [file.split('/')[-1].split('.tif')[0] for file in raster_files]

    arrs = []
    for i, (file, varname) in enumerate(zip(raster_files, varnames)):
        raster = gdal.Open(file, gdal.GA_ReadOnly)
        data = raster.GetRasterBand(1).ReadAsArray()*1.
        data[(data==0) |(data==255) | (data==-32768)] = np.nan
        ndval = np.nan

        # get geographic transform information
        gt = raster.GetGeoTransform()
        width, height = raster.RasterXSize, raster.RasterYSize
        rminx, rminy, rmaxx, rmaxy = round(gt[0],1), round(gt[3] + width*gt[4] + height*gt[5],1), round(gt[0] + width*gt[1] + height*gt[2],1), round(gt[3],1)

        if rminy != -90:

            data = np.vstack([data, np.ones((int((abs(-90)-abs(rminy))/gt[1]), width))*ndval])

        if rmaxy != 90:

            data = np.vstack([np.ones((int((abs(90)-abs(rmaxy))/gt[1]), width))*ndval, data])

        arrs.append(data)

    return arrs, varnames

def main():

    grp = sys.argv[1]

    if grp=='soil':

        soil_dir = '../../soilgrids/sg_orig/'
        arrs, varnames = tif_to_arr(soil_dir)

        soil_lat_lst, soil_lon_lst = reset_lon(nlat=1800, nlon=3600, breakpoint=None, type='SoilGrids')

        soil_df = DataFrame()

        soil_df['Lat'], soil_df['Lon'] = [la for la in soil_lat_lst for lo in soil_lon_lst], [lo for la in soil_lat_lst for lo in soil_lon_lst]

        for i, (arr, varname) in enumerate(zip(arrs, varnames)):

            soil_df[varname] = arr.reshape(-1)

        soil_df[~soil_df.isnull().any(axis=1)].to_csv(soil_dir + 'sg_processed.csv', index=False)

    ########################################################
    ########################################################
    ########################################################

    elif grp=='canopy':

        canopy_dir = '../../soilgrids/ch/'
        arrs, varnames = tif_to_arr(canopy_dir)

        canopy_lat_lst, canopy_lon_lst = reset_lon(nlat=1800, nlon=3600, breakpoint=None, type='Simard')

        canopy_df = DataFrame()

        canopy_df['Lat'], canopy_df['Lon'] = [la for la in canopy_lat_lst for lo in canopy_lon_lst], [lo for la in canopy_lat_lst for lo in canopy_lon_lst]

        for i, (arr, varname) in enumerate(zip(arrs, varnames)):

            canopy_df[varname] = arr.reshape(-1)

        canopy_df[~canopy_df.isnull().any(axis=1)].to_csv(canopy_dir + 'ch_processed.csv', index=False)

    ########################################################
    ########################################################
    ########################################################

    elif grp=='globcover':

        globcover_dir = '../../Globcover2009_V2/trait-env/'
        arrs, varnames = tif_to_arr(globcover_dir)

        globcover_lat_lst, globcover_lon_lst = reset_lon(nlat=1800, nlon=3600, breakpoint=None, type='Globcover')

        globcover_df = DataFrame()

        globcover_df['Lat'], globcover_df['Lon'] = [la for la in globcover_lat_lst for lo in globcover_lon_lst], [lo for la in globcover_lat_lst for lo in globcover_lon_lst]

        for i, (arr, varname) in enumerate(zip(arrs, varnames)):

            globcover_df[varname] = arr.reshape(-1)

        globcover_df[~globcover_df.isnull().any(axis=1)].to_csv(globcover_dir + 'globcover_processed.csv', index=False)

    return

if __name__=='__main__':
    main()
