import configparser
import datetime
import logging
import os
import time
from functools import wraps

import dask.array as da
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray as rio
import stackstac
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
from rasterio.features import rasterize
from retry import retry
from scipy import ndimage
from scipy.ndimage import binary_dilation
from urllib3 import Retry


def initialize_logger(name: str = "my-logger"):
    """
    Initializes a logger with the given name. If no name is provided, it default to 'my-logger'.

    Parameters:
    - name (str): Optional. The name of the logger [default: "my-logger"].

    Returns:
    - logger (logger): The initialized logger.
    """
    logging.basicConfig()
    logger = logging.getLogger(name)

    # Extract chain, assessment_kind and aoi from config_dict
    config_dict = get_config(config_file="config.ini")
    chain = config_dict["chain"]
    assessment_kind = config_dict["assessment_kind"]
    aoi = config_dict["aoi"]

    # use current time for log file name, along with chain, assessment_kind and aoi
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name = f"{aoi}_{chain}_{assessment_kind}_{now_str}.log"

    log_dir = os.path.join(".", "..", "log")
    log_file = os.path.join(log_dir, log_file_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def timed(func):
    """This decorator prints the execution time for the decorated function."""
    logger = logging.getLogger("my-logger")

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 3)))
        return result

    return wrapper


def stac_to_dataset(
    aoi: dict,
    time_range: tuple,
    collections: list,
    query: dict,
    assets: list,
    epsg: int,
    resolution: float,
    nodata: float = np.nan,
) -> xr.Dataset:
    """
    Perform a SpatioTemporal Asset Catalog (STAC) query and return the results as an Xarray Dataset.

    This function queries a STAC catalog for geospatial data that falls within a specified area of interest and time range,
    applies additional filters using a JSON query, and then assembles the results into an Xarray Dataset.

    Parameters:
    ----------
    aoi : dictionary
        A dictionary representing the area of interest in WGS84 coordinates.

    time_range : tuple
        A tuple (start_time, end_time) representing the temporal range for the query.

    collections : list of str
        A list of STAC collection IDs to filter the search.

    query : dict
        A JSON-like dictionary that specifies additional filters for the query.

    assets : list of str
        A list of asset names to include in the output dataset.

    epsg : int
        The EPSG code specifying the coordinate reference system (CRS) for the output dataset.

    resolution : float
        The spatial resolution in the units of the specified CRS (epsg).

    nodata : float, optional
        The value to be used for nodata or missing data. Default is np.nan.

    Returns:
    -------
    xarray.Dataset
        An Xarray Dataset containing geospatial data from the STAC catalog.
    """

    retry = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[404, 502, 503, 504],
        allowed_methods=None,
    )
    stac_api_io = StacApiIO(max_retries=retry)

    stac = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io,
    )

    search = stac.search(
        intersects=aoi,
        datetime=time_range,
        collections=collections,
        query=query,
    )

    items = search.item_collection()

    data = stackstac.stack(
        items,
        assets=assets,
        epsg=epsg,
        resolution=resolution,
    ).where(
        lambda x: x != nodata, other=np.nan
    )  # sentinel-2 uses 0 as nodata
    ds = data.to_dataset("band")

    return ds


def dilate_each_matrix(mask_np, iterations=1, structure=None):
    """
    Applies binary dilation to each matrix in a 3D array, iterating over axis 0.
    In this case, False values (considered cloudy) will be dilated.

    Parameters:
        mask_np (numpy.ndarray): 3D array
        iterations (int): Number of dilation iterations for each matrix.
        structure (numpy.ndarray): Structuring element for dilation. If None, full connectivity is used.

    Returns:
        numpy.ndarray: 3D array with dilation applied to each matrix, with False dilated.
    """
    # Allocate array for dilated result
    dilated_mask_np = np.empty_like(mask_np)

    for i in range(mask_np.shape[0]):
        # Invert the mask before dilation: True becomes False and vice versa
        inverted_mask = np.logical_not(mask_np[i])

        # Apply binary dilation to the inverted mask
        dilated_inverted_mask = binary_dilation(
            inverted_mask, iterations=iterations, structure=structure
        )

        # Invert the mask again after dilation
        dilated_mask_np[i] = np.logical_not(dilated_inverted_mask)

    return dilated_mask_np


def s2l2a_cloudmask(
    ds: xr.Dataset, da_qa: xr.DataArray, dilation_iter: int = 13
) -> xr.Dataset:
    """
    Applies cloud masking to a given xarray Dataset containing
    Sentinel-2 L2A data based on the quality band. It also applies
    dilation to the cloud mask to expand cloud coverage and include
    cloud edges and potentially cloud shadows.

    Parameters:
        ds (xarray.Dataset): Input Dataset to be cloud-masked.
        da_qa (xarray.DataArray): Quality band DataArray containing information about cloud presence.
        dilation_iter (int): Number of dilation iterations to perform, default is 13.

    Returns:
        xarray.Dataset: Cloud-masked Dataset with dilated cloud masks.
    """

    # Define values in the QA band that indicate cloud and water-related information
    # on cloud shadows, clouds, cirrus
    mask_values = [9, 10]  # 3

    mask = True  # Initialize mask as True (all pixels initially unmasked)
    for value in mask_values:
        # Update mask: Multiply mask with False (0) where pixels have cloud-related values
        mask *= da_qa != value
    # Convert the cloud mask to a binary numpy array for dilation
    mask_np = mask.values

    # Perform binary dilation on the cloud mask
    dilated_mask_np = dilate_each_matrix(mask_np, iterations=dilation_iter)

    # Convert the dilated numpy array back to a DataArray with the same dimensions as the original mask
    dilated_mask = xr.DataArray(dilated_mask_np, dims=da_qa.dims, coords=da_qa.coords)

    # Mask the original Dataset using the dilated cloud mask
    ds_masked = ds.where(dilated_mask)

    return ds_masked


def s2l2a_edgemask(ds: xr.Dataset) -> xr.Dataset:
    """
    Applies masking to a given xarray Dataset containing
    Sentinel-2 L2A data based on the valid values contained in
    the B8A and B09 bands.

    Parameters:
        ds (xarray.Dataset): Input Dataset to be masked.

    Returns:
        xarray.Dataset: Masked Dataset.
    """

    # select the B8A and B9 bands and apply the mask
    b8a_band = ds["B8A"]
    b9_band = ds["B09"]

    # create a mask where either of the selected bands has valid data
    combined_mask = b8a_band.notnull() & b9_band.notnull()

    # Apply the mask to the original dataset
    ds_masked = ds.where(combined_mask)

    return ds_masked


def s2l2a_snowmask(ds: xr.Dataset, band_green: str, band_swir: str) -> xr.Dataset:
    """
    Applies snow masking to a given xarray Dataset containing
    Sentinel-2 L2A data based on the NDSI values.

    Parameters:
        ds (xarray.Dataset): Input Dataset to be cloud-masked.
        band_green (string): name of the green band
        band_swir (string): name of the SWIR band

    Returns:
        xarray.Dataset: Snow-masked Dataset.
    """

    ndsi_threshold = 0.1
    ndsi = (ds[band_green] - ds[band_swir]) / (ds[band_green] + ds[band_swir])
    not_snow = ndsi < ndsi_threshold

    # Mask the original Dataset using the snow mask
    ds_masked = ds.where(not_snow)

    return ds_masked


def s2l2a_masked(ds: xr.Dataset) -> xr.Dataset:
    """
    Applies masking to the given Sentinel-2 Level-2A dataset.

    Parameters:
        ds (xr.Dataset): The Sentinel-2 Level-2A dataset to be masked.

    Returns:
        xr.Dataset: The masked Sentinel-2 Level-2A dataset.

    Example Usage:
        import xarray as xr

        # Load Sentinel-2 Level-2A dataset using xarray
        ds = xr.open_dataset('path/to/dataset.nc')

        # Apply masking
        ds_masked = s2l2a_masked(ds)
    """
    da_qa = ds["SCL"]
    # Apply existing masks
    ds_masked = s2l2a_cloudmask(ds, da_qa)
    ds_masked = s2l2a_edgemask(ds_masked)
    band_green = "B03"
    band_swir = "B11"
    ds_masked = s2l2a_snowmask(ds_masked, band_green, band_swir)

    return ds_masked


def harmonize_to_old(ds: xr.Dataset) -> xr.Dataset:
    """
    Harmonizes new Sentinel-2 data to the old baseline. This function take as input a Xarray Dataset containing
    both images with old (03.00) and new (04.00) processing baselines.

    Parameters
        ds (xarray.Dataset): A Dataset with four dimensions: time, band, y, x

    Returns
        xarray.Dataset: A Dataset with all values harmonized to the old processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)

    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
    dataset_bands = list(ds.data_vars)

    old = ds.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(dataset_bands))
    not_to_process = list(set(dataset_bands) - set(to_process))
    new = ds.sel(time=slice(cutoff, None))[not_to_process]  # .drop_sel(band=to_process)

    new_harmonized = ds.sel(time=slice(cutoff, None))[to_process].clip(offset)
    new_harmonized -= offset

    new = xr.merge([new, new_harmonized])[
        dataset_bands
    ]  # .sel(band=ds.band.data.tolist())
    return xr.concat([old, new], dim="time")


def coords_to_attrs(ds: xr.Dataset, excluded: list = []) -> xr.Dataset:
    """
    Given a Xarray Dataset, converts its non-dimension coordinates to attributes and drops these coordinates.
    Any additional coordinates which should not be converted and dropped can be specified in the 'excluded' list.

    Args:
        ds (xarray.Dataset): The Dataset to process.
        excluded (list): The list of additional excluded coordinates.

    Returns:
        xarray.Dataset: The processed Dataset.
    """
    for coord in ds.coords:
        if coord not in ["x", "y", "time"] + excluded:
            ds.attrs[str(coord)] = str(ds.coords[coord].values)
            ds = ds.drop_vars(coord)
    return ds


def isel_func(ds: xr.Dataset, indexer: xr.DataArray) -> xr.Dataset:
    """
    Given a Xarray Dataset, applies the isel function by selecting time indexes contained in a Xarray DataArray and
    drops the time coordinate. This function is meant to be applied to a Dask chunk in the xarray.Dataset.map_blocks
    function.

    Args:
        ds (xarray.Dataset): The Dataset to apply the isel function to.
        indexer (xarray.DataArray): The DataArray containing the time indexes to select from ds.

    Returns:
        xarray.Dataset: The processed Dataset.
    """
    ds_sel = ds.isel(time=indexer).drop_vars("time")
    return ds_sel


@retry(RuntimeError, tries=10, delay=2)
def medoid_mosaic(ds: xr.Dataset) -> xr.Dataset:
    """
    The method  apply a Medoid process for pixels with multiple observations acquired on different dates (Kennedy et al., 2018),
    whereby the spectral value for each band for each pixel was compared to the median value for the same pixel calculated from among values available
    or all images selected for that year.
    Then, the pixel whose spectral values are the most like the medians using Euclidean spectral distance was selected,
    and the relative band values were used to populate the image composite. This implementation expects as input Sentinel-2 L2A dataset.

    Parameters:
        ds (xarray.Dataset): Input Sentinel-2 L2A Dataset.

    Returns:
        xarray.Dataset: Dataset containing the medoid composite.
    """
    # chunks the dataset along the x and y dimensions (time should remain unchunked)
    ds = ds.chunk({"time": -1, "y": 1024, "x": 1024})

    # drops coordinates with no associated dimension
    ds = coords_to_attrs(ds)

    # time median
    ds_median = ds.median(dim="time", skipna=True)

    # Euclidean distance between each observation and their median, summing among all bands.
    # da_dist is a DataArray with the same dimensions as ds and only one variable, the distance.
    # fillna is necessary because if there's a pixel with nan for all times, argmin can't choose one
    # of these and throws an error.
    da_dist = da.sqrt(((ds - ds_median) ** 2).fillna(np.inf).to_array().sum("variable"))

    # for each pixel, finds the time index with the smallest value (for the distance variable).
    # medoid_index is a DataArray with the same spatial dimensions as ds and only one variable, the time index
    medoid_index = da_dist.argmin(dim="time", skipna=True)
    medoid_index.chunk({"x": ds.chunksizes["x"], "y": ds.chunksizes["y"]})

    # to use map_blocks with a function that modifies the Dataset shape, you need to provide a template
    # dataset that has exactly the same shape as the output of map_blocks
    ds_template = ds.isel(time=0).drop_vars("time")

    # applies the isel function in a vectorized way
    ds_medoid = ds.map_blocks(isel_func, args=[medoid_index], template=ds_template)

    # counts the number of observations to measure the quality of the medoid and stores it in a new dataset variable

    n_images = ds["B02"].count(dim="time")
    ds_medoid["nImages"] = n_images

    return ds_medoid


def datarray2dataset(da: xr.DataArray) -> xr.Dataset:
    """
    Convert from DataArray to Dataset. If the 'long_name' attribute is present and matches the number of bands,
    use it to rename the bands. Otherwise, convert without renaming and notify the user.

    Parameters:
        da (xarray.DataArray): Input DataArray.

    Returns:
        xarray.Dataset: Dataset with variables, renamed if possible.
    """

    # Convert to DataSet
    dataset = da.to_dataset(dim="band")

    # Check if 'long_name' attribute exists and matches the number of bands
    if "long_name" in da.attrs and len(da.attrs["long_name"]) == da.sizes["band"]:
        variable_names = da.attrs["long_name"]
        name_dict = {i: name for i, name in enumerate(variable_names, start=1)}
        dataset = dataset.rename(name_dict)
    else:
        print(
            "'long_name' attribute is not present or does not match the number of bands. Bands will not be renamed."
        )

    return dataset


def apply_fnf_mask(
    data_array: xr.DataArray, mask_tiff_path: str, epsg: int
) -> xr.DataArray:
    """
    Apply a mask to an xarray.DataArray based on a external forest map (in GeoTIFF format).
    The forest mask must have:
    - 0 or nan in case of not forest type pixel
    - a number >=1 for a forest type pixel.

    Parameters:
        data_array (xarray.DataArray): The xarray.DataArray to be masked.
        mask_tiff_path (str): The file path to the  GeoTIFF defining the mask.

    Returns:
        xarray.DataArray: The xarray.DataArray with the mask applied.
    """
    # Read the mask GeoTIFF using rioxarray
    mask = xr.open_dataset(mask_tiff_path)
    data_array.rio.set_crs("EPSG:" + str(epsg))

    # Reproject and align the mask_data_array to the same grid as data_array using 'nearest' resampling
    mask_repr_match = mask.rio.reproject_match(data_array)

    mask_repr_match = mask_repr_match.assign_coords(
        {
            "x": data_array.x,
            "y": data_array.y,
        }
    )

    # Apply the mask
    # if the provided mask is non-binary (i.e. it's a forest category map), makes it binary by
    # putting zeroes where there are no forests and ones everywhere else
    cond0 = mask_repr_match == 0
    cond1 = mask_repr_match == 1
    isbinary = np.logical_or(cond0, cond1).all()
    if isbinary == False:
        mask_repr_match = mask_repr_match.fillna(0)
        mask_repr_match = mask_repr_match.where(mask_repr_match == 0, 1)
    masked_data = data_array.where(mask_repr_match == 1, 255)

    return masked_data["band_data"].transpose("band", "y", "x")


def clip_raster_by_vector(
    data_array: xr.DataArray, vector: xr.DataArray
) -> xr.DataArray:
    """
    Clip a DataArray based on the polygon defined by the vector and return a new DataArray.
    This method put no data value outside the vector border.
    Parameters:
        data_array (xarray.DataArray): The xarray.DataArray to be clipped.
        vector (xarray.DataArray): The xarray.DataArray defining the polygon for clipping.

    Returns:
        xarray.DataArray: The clipped DataArray.
    """
    epsg = data_array.rio.crs.to_string()
    # Clip the input DataArray to the polygon defined by the vector
    clipped_data_array = data_array.rio.clip(vector.to_crs(epsg))

    return clipped_data_array


def rasterize_geodataframe(gdf, raster_template_path):
    """
    Rasterize a GeoDataFrame using a template raster.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to rasterize.
    - raster_template_path (str): The path to the template raster.

    Returns:
    - raster_xarray (xarray.DataArray): The rasterized data as a DataArray.
    - raster_meta (dict): The metadata of the resulting raster.

    """
    # Read the template raster
    with rasterio.open(raster_template_path) as src:
        raster_meta = src.meta.copy()

        # Rasterize the GeoDataFrame
        rasterized = rasterize(
            [(geometry, 1) for geometry in gdf.geometry],
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )

        raster_meta["dtype"] = "uint8"

        raster_xarray = xr.DataArray(
            rasterized,
            dims=("y", "x"),
            coords={
                "y": np.arange(raster_meta["height"]),
                "x": np.arange(raster_meta["width"]),
            },
        )

    return raster_xarray, raster_meta


def normalized_difference(da_1: xr.DataArray, da_2: xr.DataArray) -> xr.DataArray:
    """
    Given two Xarray DataArray representing two different bands of the same dataset,
    computes the normalized difference between the two bands.
    Parameters:
        da_1 (xarray.DataArray): First DataArray.
        da_2 (xarray.DataArray): Second DataArray.

    Returns:
        xarray.DataArray: DataArray containing the normalized difference.
    """
    return (da_1 - da_2) / (da_1 + da_2)


def calculate_ndmi_nbr_msi(ds: xr.Dataset) -> xr.DataArray:
    """
    Given a Xarray Dataset containing Sentinel-2 L2A data, computes the
    NDMI, NDR and MSI indexes.

    Parameters:
        ds (xarray.Dataset): Input Dataset containing Sentinel-2 L2A data.

    Returns:
        ndmi (xarray.DataArray): DataArray containing the NDMI index.
        nbr (xarray.DataArray): DataArray containing the NBR index.
        msi (xarray.DataArray): DataArray containing the MSI index.
    """
    # convert xarray datarray to dataset

    ndmi = normalized_difference(ds["B08"], ds["B11"])
    nbr = normalized_difference(ds["B08"], ds["B12"])
    msi = ds["B11"] / ds["B08"]
    return ndmi, nbr, msi


def calculate_magnitude(
    ndmi_pre: xr.DataArray,
    nbr_pre: xr.DataArray,
    msi_pre: xr.DataArray,
    ndmi_post: xr.DataArray,
    nbr_post: xr.DataArray,
    msi_post: xr.DataArray,
) -> xr.DataArray:
    """
    Given six Xarray DataArray containing the NDMI, NBR and MSI indexes over two
    concatenated periods of time, the method uses a particular implementation of the Three Indices Three Dimensions (3I3D)
    algorithm (Francini et a., 2022 - https://doi.org/10.3390/s22052015).
    In this method, the forest health decrease magnitude is calculated by averaging just those three PMIs and
    it has values between 0 and 1.
    Lastly, for storage purposes, so that images can be stored using byte datatype,
    the float values between 0 and 1 are multiplied by 255 (2^8)

    Parameters:
        ndmi_pre (xarray.DataArray): DataArray containing the NDMI index for the initial period of time.
        nbr_pre (xarray.DataArray): DataArray containing the NBR index for the initial period of time.
        msi_pre (xarray.DataArray): DataArray containing the MSI index for the initial period of time.
        ndmi_post (xarray.DataArray): DataArray containing the NDMI index for the final period of time.
        nbr_post (xarray.DataArray): DataArray containing the NBR index for the final period of time.
        msi_post (xarray.DataArray): DataArray containing the MSI index for the final period of time.

    Returns:
        xarray.Dataset: Dataset containing the magnitude.
    """

    x1 = ndmi_post - ndmi_pre
    y1 = nbr_post - nbr_pre
    z1 = msi_post - msi_pre

    # va = xr.merge([x1, y1, z1]).to_dataset('band').rename({1: 'ndmi_va', 2: 'nbr_va', 3: 'msi_va'})

    modulo_a = np.sqrt(x1**2 + y1**2 + z1**2)
    phi_a = np.arctan(y1 / x1) * 180 / 3.1416
    theta_a = np.arccos(z1 / modulo_a) * 180 / 3.1416

    phi_a = phi_a.where(x1 >= 0, phi_a + 180)
    p_theta_a = np.abs(theta_a - 45)
    p_theta_a = np.abs((135 - p_theta_a) / 135)
    p_phi_a = np.abs(phi_a - 225)
    p_phi_a = np.abs((315 - p_phi_a) / 315)

    magnitude = (p_theta_a + p_phi_a + modulo_a) / 3
    magnitude = magnitude * 255
    # magnitude = magnitude.where(magnitude <= 255, 255)

    return magnitude


def calculate_magnitude_trio(
    ndmi_pre: xr.DataArray,
    nbr_pre: xr.DataArray,
    msi_pre: xr.DataArray,
    ndmi_x: xr.DataArray,
    nbr_x: xr.DataArray,
    msi_x: xr.DataArray,
    ndmi_post: xr.DataArray,
    nbr_post: xr.DataArray,
    msi_post: xr.DataArray,
) -> xr.DataArray:
    """
    Given nine Xarray DataArray containing the NDMI, NBR and MSI indexes over three
    concatenated periods of time, uses the 3I3D algorithm to compute
    the change magnitude.

    Parameters:
        ndmi_pre (xarray.DataArray): DataArray containing the NDMI index for the initial period of time.
        nbr_pre (xarray.DataArray): DataArray containing the NBR index for the initial period of time.
        msi_pre (xarray.DataArray): DataArray containing the MSI index for the initial period of time.
        ndmi_post (xarray.DataArray): DataArray containing the NDMI index for the final period of time.
        nbr_post (xarray.DataArray): DataArray containing the NBR index for the final period of time.
        msi_post (xarray.DataArray): DataArray containing the MSI index for the final period of time.

    Returns:
        xarray.Dataset: Dataset containing the change magnitude.
    """

    # Sentinel-2 L2A bands common name
    # ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    # ['blue', 'green', 'red', 'redE1', 'redE2', 'redE3', 'nir', 'redE4', 'swir1', 'swir2']

    x1 = ndmi_x - ndmi_pre
    y1 = nbr_x - nbr_pre
    z1 = msi_x - msi_pre

    x2 = ndmi_post - ndmi_x
    y2 = nbr_post - nbr_x
    z2 = msi_post - msi_x

    modulo_a = np.sqrt(x1**2 + y1**2 + z1**2)
    phi_a = np.arctan(y1 / x1) * 180 / 3.1416
    theta_a = np.arccos(z1 / modulo_a) * 180 / 3.1416

    phi_a = phi_a.where(x1 >= 0, phi_a + 180)
    p_theta_a = np.abs(theta_a - 45)
    p_theta_a = np.abs((135 - p_theta_a) / 135)
    p_phi_a = np.abs(phi_a - 225)
    p_phi_a = np.abs((315 - p_phi_a) / 315)

    # magnitude = (p_theta_a + p_phi_a + modulo_a) / 3
    # magnitude = magnitude * 255
    # magnitude = magnitude.where(magnitude <= 255, 255).astype('uint8')

    modulo_b = np.sqrt(x2**2 + y2**2 + z2**2)
    phi_b = np.arctan(y2 / x2) * 180 / 3.1416
    theta_b = np.arccos(z2 / modulo_b) * 180 / 3.1416
    phi_b = phi_b.where(x2 >= 0, phi_b + 180)
    p_theta_b = np.abs(theta_b - 135)
    p_theta_b = np.abs((135 - p_theta_b) / 135)
    p_phi_b = np.abs(phi_b - 45)
    p_phi_b = np.abs((225 - p_phi_b) / 225)

    de = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2

    magnitude = (p_theta_a + p_theta_b + p_phi_a + p_phi_b + de) / 5
    magnitude = magnitude * 255
    # magnitude = magnitude.where(magnitude <= 255, 255).astype('uint8')

    return magnitude


def apply_mmu_img(da: xr.DataArray, mmu: int) -> xr.DataArray:
    """
    Fill holes and discards objects smaller than the Minimum Mapping Unit (MMU) in a binary change map
    represented by a 2D DataArray.

    Parameters:
    - da (xarray.DataArray): Input change map represented as a 2D DataArray.
    - mmu (int): Minimum Mapping Unit, specifying the minimum size (in pixels) for retained objects.

    Returns:
    - xarray.DataArray: Modified change map with objects smaller than the MMU removed.

    Raises:
    - AssertionError: If the input change map contains values other than 0 and 1.

    The function identifies connected components (objects) in the change map and removes objects
    smaller than the specified MMU. It ensures that the resulting change map maintains the original
    NoData values and Coordinate Reference System (CRS).

    Note: This function uses the scikit-image.ndimage module for labeling connected components.

    Example:
    ```python
    import xarray as xr
    from apply_mmu import apply_mmu_img

    # Assuming 'change_map' is a 2D DataArray with a NoData value
    mmu_value = 100  # Set the desired MMU value
    modified_change_map = apply_mmu_img(change_map, mmu_value)
    ```
    """
    nodata_value = da.rio.nodata
    crs = da.rio.crs

    # the change mask must only contain 0 and 1 values, where 0 is considered background and 1 identifies pixels of interest for the mmu
    da_new = da.where(da != 3, 0)
    da_new = da_new.where(da != 2, 1)
    da_new = da_new.where(da != 1, 0)
    img_array = da_new.where(da_new != nodata_value, 0).to_numpy()[0]
    assert (
        (img_array == 0) | (img_array == 1)
    ).all(), "Change map must only have 0 and 1 values."

    # plus-shaped kernel with radius 1 for morphological operations
    plus_shaped_kernel = ndimage.generate_binary_structure(2, 1)

    # swaps zeros with ones in order to find small patches of zeros (small holes)
    img_array_neg = (~(img_array == True)) * 1

    # finds all connected components (objects) by superimposing a plus-shaped kernel with radius 1 on each non-zero element of the image
    # labeled_array has the same shape as the input change map, but pixels of each object have a unique integer value
    # num_features is the number of objects found
    labeled_array, num_features = ndimage.label(
        img_array_neg, structure=plus_shaped_kernel
    )

    # sums values in the change map inside the regions defined by labels (i.e. inside objects)
    # objects_areas contains the number of pixels (i.e. the area) inside each object
    objects_areas = ndimage.sum(
        img_array_neg, labels=labeled_array, index=np.arange(num_features + 1)
    )

    # fills only objects smaller than (or equal to) the mmu
    mask = objects_areas <= mmu
    labels_to_fill = np.where(mask)
    # uses a mask matrix to only fill these objects
    # the mask matrix is True everywhere except in objects to fill
    mask_matrix = np.full(labeled_array.shape, True)
    for label in labels_to_fill[0]:
        # puts False in objects to fill (i.e. objects with the selected label)
        mask_matrix *= ~(labeled_array == label)

    # puts a 1 in the small holes (the negation of the mask matrix identifies pixels to fill)
    img_array_filled = np.where(~mask_matrix, 1, img_array)
    img_array = img_array_filled

    # same as above, but to find objects larger than (or equal to) the mmu
    labeled_array, num_features = ndimage.label(img_array, structure=plus_shaped_kernel)

    objects_areas = ndimage.sum(
        img_array, labels=labeled_array, index=np.arange(num_features + 1)
    )

    # keeps only objects larger than (or equal to) the mmu
    mask = objects_areas >= mmu
    labels_to_keep = np.where(mask)
    # uses a mask matrix to only keep these objects
    # the mask matrix is True everywhere except in objects to keep
    mask_matrix = np.full(labeled_array.shape, True)
    for label in labels_to_keep[0]:
        # puts False in objects to keep (i.e. objects with the selected label)
        mask_matrix *= ~(labeled_array == label)

    # updates the change mask
    da_new = da.copy()
    # the negation of the mask matrix identifies pixels to keep
    img_array_masked = np.where(~mask_matrix, img_array, 0)
    da_new[0] = img_array_masked

    # restores no-values and crs from the original change map
    da_new = xr.where(da == nodata_value, da, da_new)
    da_new = xr.where(da == 3, da, da_new)
    da_new = xr.where(da_new == 1, 2, da_new)
    da_new = xr.where(da_new == 0, 1, da_new)
    da_new = da_new.rio.write_crs(crs)
    da_new = da_new.rio.set_nodata(nodata_value)
    return da_new


@retry(RuntimeError, tries=10, delay=2)
def compute_change(
    magnitude,
    th,
    fn_path,
    save_path,
    save_path_cog,
    stac_epsg,
    epsg,
    save_flag,
    clip_geometry,
    overwrite,
    mmu=1,
):
    """

    This method takes the results of calculate_magnitude_trio function (implementation of 3I3D algorithm for the SVC-S4-7b)
    and it computes forest disturbance map based on a given magnitude threshold defined in the config file.
    This step results in a forest disturbance map classified into disturbed (value = 2) and undisturbed classes (value=1),
    No-Data (value=255). These values have been defined by e-Geos.
    It optionally applys the MMU.
    It is worth noting that the forest disturbance approach is applied to forest covers, thus, a forest mask is needed.
    It takes the following parameters:

    Parameters:
    - magnitude: A DataArray representing the magnitude of change.
    - th: The threshold value to determine change.
    - fn_path: The file path of the fnf mask.
    - save_path: The file path to save the output raster.
    - save_path_cog: The file path to save the output raster as a COG (Cloud-Optimized GeoTIFF).
    - stac_epsg: The EPSG code of the input magnitude data.
    - epsg: The EPSG code of the desired output projection.
    - save_flag: A boolean flag indicating whether to save the output raster.
    - clip_geometry: A geometry object to clip the output raster.
    - overwrite: A boolean flag indicating whether to overwrite the output file if it already exists.
    - mmu: The minimum mapping unit. Default value is 1.

    The method performs the following steps:

    1. Checks for missing values in the magnitude DataArray.
    2. Divides the magnitude DataArray into two classes based on the threshold value: values >= threshold are considered class 2, otherwise class 1.
    3. Sets class 3 (missing values) in the change DataArray based on the missing values in the magnitude DataArray.
    4. Sets the CRS (Coordinate Reference System) of the change DataArray using the provided stac_epsg value, and reprojects it to the desired EPSG using the epsg value.
    5. Applies the fnf mask (forest mask) to the change DataArray using the fn_path and epsg values.
    6. If mmu value is greater than 1, applies the minimum mapping unit to the change DataArray using the apply_mmu_img function.
    7. If save_flag is True or the output file does not exist (and overwrite is True), proceeds to save the change DataArray as a GeoTIFF file.
    8. Clips the change DataArray based on the provided clip_geometry.
    9. Writes the change DataArray without a nodata value using the rio.write_nodata method.
    10. Saves the change DataArray as a GeoTIFF file using the save_path and save_path_cog values.
    11. Returns the change DataArray.

    Note: This method uses the @retry decorator to automatically retry the method in case of a RuntimeError. The retry decorator is not provided in this documentation.

    """
    magnitude_isna = magnitude.isnull()

    change = xr.where((magnitude >= th), 2, 1)
    change = change.where(~magnitude_isna, 3)
    change = change.rio.write_crs("EPSG:{}".format(stac_epsg)).rio.reproject(
        "EPSG:{}".format(epsg), nodata=255
    )

    change = apply_fnf_mask(change, fn_path, epsg)

    # filling nan from fnf mask
    # change = change.fillna(255)
    # change.rio.set_nodata(255)

    # Optionally applys the MMU
    if mmu > 1:
        change = apply_mmu_img(da=change, mmu=mmu)

    if save_flag and (overwrite or (not os.path.exists(save_path))):
        try:
            if change.shape[0] == 0 or change.shape[1] == 0:
                raise ValueError(
                    "Dimension of 3I3D output is null, cannot create geotiff file."
                )

            # clips the result on the provided area of interest
            if clip_geometry is not None:
                change = clip_raster_by_vector(change, clip_geometry)

            change = change.rio.write_nodata(None)
            change.rio.to_raster(save_path)
            save_cog(save_path, save_path_cog)

        except rasterio.errors.RasterioIOError as e:
            print(f"Error during tiff creation: {e}")
        except ValueError as e:
            print(f"Validation error: {e}")

    return change


@retry(RuntimeError, tries=10, delay=2)
def compute_classification(
    magnitude,
    fn_path,
    save_path,
    save_path_cog,
    stac_epsg,
    epsg,
    save_flag,
    clip_geometry,
    overwrite,
):
    """

    Compute Classification

    This method computes the classification of a given magnitude dataset.

    Parameters:
    - `magnitude` (xarray.DataArray): The magnitude dataset.
    - `fn_path` (str): The file path to the FNF mask dataset.
    - `save_path` (str): The file path to save the classification raster.
    - `save_path_cog` (str): The file path to save the classification raster as a Cloud-Optimized GeoTIFF (COG).
    - `stac_epsg` (int): The EPSG code of the original magnitude dataset.
    - `epsg` (int): The EPSG code to reproject the classification raster.
    - `save_flag` (bool): Flag to indicate whether to save the classification raster or not.
    - `clip_geometry` (ogr.Geometry): Optional geometry object to clip the classification raster.
    - `overwrite` (bool): Flag to indicate whether to overwrite existing classification raster if it exists.

    Returns:
    - `classes` (xarray.DataArray): The computed classification raster.

    Raises:
    - `ValueError`: If the dimension of the magnitude output is null.
    - `rasterio.errors.RasterioIOError`: If there is an error during TIFF creation.

    """
    if (not os.path.exists(save_path)) or overwrite:
        class0 = magnitude <= 170
        class1 = (magnitude > 170) * (magnitude <= 190)
        class2 = (magnitude > 190) * (magnitude <= 210)
        class3 = magnitude > 210
        classna = magnitude.isnull()

        classes = xr.where(class0, 0, magnitude)
        classes = xr.where(class1, 1, classes)
        classes = xr.where(class2, 2, classes)
        classes = xr.where(class3, 3, classes)
        classes = xr.where(classna, 4, classes)

        classes = classes.rio.write_crs("EPSG:{}".format(stac_epsg)).rio.reproject(
            "EPSG:{}".format(epsg), nodata=255
        )
        classes = apply_fnf_mask(classes, fn_path, epsg)

        if save_flag and (overwrite or (not os.path.exists(save_path))):
            try:
                if classes.shape[0] == 0 or classes.shape[1] == 0:
                    raise ValueError(
                        "Dimension of 3I3D output is null, cannot create geotiff file."
                    )

                # clips the result on the provided area of interest
                if clip_geometry is not None:
                    classes = clip_raster_by_vector(classes, clip_geometry)

                classes = classes.compute()
                # filling nan from fnf mask
                # classes = classes.fillna(255)
                # classes.rio.set_nodata(255)
                classes = classes.rio.write_nodata(None)
                classes.rio.to_raster(save_path)
                save_cog(save_path, save_path_cog)

            except rasterio.errors.RasterioIOError as e:
                print(f"Error during tiff creation: {e}")
            except ValueError as e:
                print(f"Validation error: {e}")
    else:
        classes = rio.open_rasterio(save_path).chunk("auto").compute()
    # classes = classes.compute()

    return classes


def ufunc_classify(value):
    """
    Classifies a given value into one of five categories based on certain conditions.

    Parameters:
    - value: A numeric value that needs to be classified.

    Returns:
    - An integer representing the category of the given value. The possible categories are: 1, 2, 3, 4, or 5.

    If the given value does not fall into any category, the method returns np.nan.

    Example Usage:
    >>> ufunc_classify(1.5)
    2

    >>> ufunc_classify(6)
    nan
    """
    cond1 = value > 0 and value <= 1
    cond2 = value > 1 and value <= 2
    cond3 = value > 2 and value <= 3
    cond4 = value > 3 and value <= 4
    cond5 = value > 4 and value <= 5

    if cond1:
        return 1
    elif cond2:
        return 2
    elif cond3:
        return 3
    elif cond4:
        return 4
    elif cond5:
        return 5
    else:
        return np.nan


def ufunc_reclassify(value, table):
    """
    Reclassifies a value based on the lookup table.

    Args:
        value (int): The value to be reclassified.
        table (dict): A lookup table that maps input values to output values.

    Returns:
        int: The reclassified value.
    """
    if (value in table.keys()) and value != 255:
        reclassified = table[value]
    else:
        reclassified = 255
    return reclassified


@retry(RuntimeError, tries=10, delay=2)
def compute_vulnerability(
    classes,
    ft_path,
    ft_cod_path,
    save_path,
    save_path_cog,
    epsg,
    save_flag,
    clip_geometry,
    overwrite,
):
    """

    This method computes vulnerability based on the given inputs and saves the result as a GeoTIFF file.

    Parameters:
    - classes: A rasterio.DatasetReader object representing the classified classes.
    - ft_path: A string representing the file path of the feature raster.
    - ft_cod_path: A string representing the file path of the feature code table.
    - save_path: A string representing the file path to save the vulnerability result.
    - save_path_cog: A string representing the file path to save the Cloud-Optimized GeoTIFF (COG).
    - epsg: An integer representing the EPSG code for the desired coordinate reference system (CRS).
    - save_flag: A boolean indicating whether to save the result or not.
    - clip_geometry: A GeoJSON-like object representing the geometry with which to clip the result.
    - overwrite: A boolean indicating whether to overwrite an existing file with the same save path.

    Returns:
    - vulnerability: A rasterio.DatasetReader object representing the vulnerability result.

    Note: This method uses the 'retry' decorator to retry the execution of the method if a 'RuntimeError' occurs. The maximum number of tries is 10, and there is a 2-second delay between
    * each try.

    """
    classes.rio.set_nodata(255)
    classes = classes.to_dataset("band")

    classified_fha_data = xr.where(classes == 255, 255, classes + 1)
    classified_fha_data = classified_fha_data.rio.write_crs(classes.rio.crs)
    classified_fha_data[1] = classified_fha_data[1].astype("int64")

    ft_raster = rio.open_rasterio(ft_path)
    ft_raster = ft_raster.to_dataset("band")

    classified_fha_data_repr = classified_fha_data.rio.reproject_match(
        ft_raster, resampling=rasterio.enums.Resampling.bilinear
    )
    classified_fha_resampled = classified_fha_data_repr.assign_coords(
        {
            "x": ft_raster.x,
            "y": ft_raster.y,
        }
    )

    classified_fha_resampled = classified_fha_resampled.rio.clip_box(
        *classified_fha_data.rio.bounds(), crs=classified_fha_data.rio.crs
    )
    ft_raster = ft_raster.rio.clip_box(
        *classified_fha_data.rio.bounds(), crs=classified_fha_data.rio.crs
    )

    cond1 = (classified_fha_resampled > 0) & (classified_fha_resampled <= 1)
    cond2 = (classified_fha_resampled > 1) & (classified_fha_resampled <= 2)
    cond3 = (classified_fha_resampled > 2) & (classified_fha_resampled <= 3)
    cond4 = (classified_fha_resampled > 3) & (classified_fha_resampled <= 4)
    cond5 = (classified_fha_resampled > 4) & (classified_fha_resampled <= 5)
    cond6 = (classified_fha_resampled <= 0) | (classified_fha_resampled > 5)

    cmask_raster_FHA = xr.where(cond1, 1, classified_fha_resampled)
    cmask_raster_FHA = xr.where(cond2, 2, cmask_raster_FHA)
    cmask_raster_FHA = xr.where(cond3, 3, cmask_raster_FHA)
    cmask_raster_FHA = xr.where(cond4, 4, cmask_raster_FHA)
    cmask_raster_FHA = xr.where(cond5, 5, cmask_raster_FHA)
    cmask_raster_FHA = xr.where(cond6, 255, cmask_raster_FHA)

    masked_ft = xr.where(
        classified_fha_resampled != 255, ft_raster.astype("int64"), 255
    )
    overlapped_raster = (masked_ft + 10) ** cmask_raster_FHA

    df_cat_ft = pd.read_excel(ft_cod_path)
    keys = df_cat_ft["Num_ID"].values
    values = df_cat_ft["ID"].values
    correspondence_table = {key: value for key, value in zip(keys, values)}

    vulnerability = xr.apply_ufunc(
        ufunc_reclassify,
        overlapped_raster[1],
        kwargs={"table": correspondence_table},
        vectorize=True,
        input_core_dims=[[]],
        dask="parallelized",
        output_dtypes=["float64"],
    )

    vulnerability = vulnerability.rio.write_crs(
        classified_fha_resampled.rio.crs
    ).rio.reproject("EPSG:{}".format(epsg), nodata=255)
    vulnerability = vulnerability.to_dataset().rename({1: "vulnerability"})

    if save_flag and (overwrite or (not os.path.exists(save_path))):
        try:
            if len(vulnerability.x) == 0 or len(vulnerability.y) == 0:
                raise ValueError(
                    "Dimension of 3I3D output is null, cannot create geotiff file."
                )

            # clips the result on the provided area of interest
            if clip_geometry is not None:
                vulnerability = clip_raster_by_vector(vulnerability, clip_geometry)

            # filling nan from fnf mask
            # vulnerability = vulnerability.fillna(255)
            # vulnerability.to_array().rio.set_nodata(255).rio.to_raster(save_path)
            vulnerability = vulnerability.to_array().rio.write_nodata(None)
            vulnerability.rio.to_raster(save_path)
            save_cog(save_path, save_path_cog)

        except rasterio.errors.RasterioIOError as e:
            print(f"Error during tiff creation: {e}")
        except ValueError as e:
            print(f"Validation error: {e}")

    return vulnerability


@retry(RuntimeError, tries=10, delay=2)
def save_input_zarr(ds_list, path_list, overwrite, clip_geometry, epsg):
    """
    Save input data to Zarr format.

    Parameters:
    - ds_list (list): List of xarray.Dataset objects containing the input data.
    - path_list (list): List of paths where the Zarr data should be saved.
    - overwrite (bool): Boolean flag indicating whether to overwrite existing files.
    - clip_geometry (shapely.geometry.base.BaseGeometry or None): Shapely geometry used for clipping.
    - epsg (int): EPSG code specifying the coordinate reference system.

    Returns:
    - None

    """
    for ds, path in zip(ds_list, path_list):
        if (not os.path.exists(path)) or overwrite:
            ds = coords_to_attrs(ds, excluded=["id"])
            ds.attrs["spec"] = str(ds.attrs["spec"])
            if clip_geometry is not None:
                ds = (
                    ds.rio.write_crs("EPSG:{}".format(epsg))
                    .astype("float32")
                    .rio.clip_box(
                        minx=clip_geometry.bounds.minx,
                        miny=clip_geometry.bounds.miny,
                        maxx=clip_geometry.bounds.maxx,
                        maxy=clip_geometry.bounds.maxy,
                        crs=clip_geometry.crs,
                    )
                )
            ds.chunk({"time": 1, "x": 1024, "y": 1024}).to_zarr(
                path, mode="w" if overwrite else None
            )


def save_cog(input_tif, output_tif):
    """
    Saves a GeoTIFF file as a Cloud Optimized GeoTIFF (COG).

    Parameters:
        input_tif (str): The path of the input GeoTIFF file.
        output_tif (str): The path of the output COG file.

    Example usage:
        >>> save_cog('input.tif', 'output.cog')

    Note:
        This method uses the 'rio cogeo create' command from the 'rasterio' library to create a Cloud Optimized GeoTIFF.
    """
    cmd = "rio cogeo create -p lzw {} {}".format(input_tif, output_tif)
    os.system(cmd)


class health_assessment:
    """
    :param start_date_pre: Start date for pre assessment
    :param end_date_pre: End date for pre assessment
    :param start_date_mid: Start date for mid assessment
    :param end_date_mid: End date for mid assessment
    :param start_date_post: Start date for post assessment
    :param end_date_post: End date for post assessment
    :param aoi: Area of interest
    :param clip_geometry: Clipping geometry
    :param mgrs_tile: MGRS tile
    :param epsg: EPSG code
    :param cloudcover: Maximum cloud cover threshold
    :param save_to_disk: Saving options
    :param results_path: Path to store the assessment results
    :param aoi_name: Name of the Area of Interest
    :param assessment_kind: Kind of assessment (weekly or yearly)
    :param chain: Classification chain
    :param fn_path: File path for forest non-forest classifier
    :param ft_path: File path for Feature Transform model
    :param ft_cod_path: File path for Feature Transform dictionary
    :param th: Threshold value for classification
    :param mmu: Minimum Mapping Unit (MMU)
    :param overwrite: Overwrite existing results if True
    :param log_level: Logging level (debug, info, warning, error, critical)
    """

    def __init__(
        self,
        start_date_pre: pd.Timestamp,
        end_date_pre: pd.Timestamp,
        start_date_mid: pd.Timestamp,
        end_date_mid: pd.Timestamp,
        start_date_post: pd.Timestamp,
        end_date_post: pd.Timestamp,
        aoi: dict,
        clip_geometry,
        mgrs_tile: str,
        epsg: int,
        cloudcover: float,
        save_to_disk: dict,
        results_path: str,
        aoi_name: str,
        assessment_kind: str,
        chain: str,
        fn_path: str,
        ft_path: str,
        ft_cod_path: str,
        th: float = 224,
        mmu: int = 1,
        overwrite: bool = False,
        log_level: str = "warning",
    ):
        self.start_date_pre = start_date_pre
        self.end_date_pre = end_date_pre
        self.start_date_mid = start_date_mid
        self.end_date_mid = end_date_mid
        self.start_date_post = start_date_post
        self.end_date_post = end_date_post
        self.aoi = aoi
        self.clip_geometry = clip_geometry
        self.mgrs_tile = mgrs_tile
        self.stac_epsg = 32632
        self.epsg = epsg
        self.cloudcover = cloudcover
        self.save_to_disk = save_to_disk
        self.results_path = results_path
        self.aoi_name = aoi_name
        self.assessment_kind = assessment_kind
        self.chain = chain
        self.fn_path = fn_path
        self.ft_path = ft_path
        self.ft_cod_path = ft_cod_path
        self.overwrite = overwrite
        self.log_level = log_level

        self.logger = initialize_logger()

        if log_level == "debug":
            self.logger.setLevel(logging.DEBUG)
        elif log_level == "info":
            self.logger.setLevel(logging.INFO)
        elif log_level == "warning":
            self.logger.setLevel(logging.WARNING)
        elif log_level == "error":
            self.logger.setLevel(logging.ERROR)
        elif log_level == "critical":
            self.logger.setLevel(logging.CRITICAL)

        # collection to query
        self.collections = ["sentinel-2-l2a"]

        # query to select a cloud cover threshold and a specific tile
        self.query = {
            "eo:cloud_cover": {"lt": self.cloudcover},
            "s2:mgrs_tile": {"in": self.mgrs_tile},
        }

        # bands of interest
        self.assets = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
            "SCL",
        ]
        # spatial resolution (meters)
        self.resolution = 10
        # nodata value for Sentinel-2 L2A
        self.nodata = 0

        # save flags
        self.save_input_dataset = self.save_to_disk["save_input_dataset"]
        self.save_medoid_pre = self.save_to_disk["save_medoid_pre"]
        self.save_medoid_post = self.save_to_disk["save_medoid_post"]
        self.save_indexes_pre = self.save_to_disk["save_indexes_pre"]
        self.save_indexes_post = self.save_to_disk["save_indexes_post"]
        self.save_classification = self.save_to_disk["save_classification"]

        # data directories
        self.input_dir = os.path.join(self.results_path, "input")
        self.temporary_dir = os.path.join(self.results_path, "temp")
        self.output_dir = os.path.join(self.results_path, "output")
        self.fhm_dir = os.path.join(self.output_dir, "FHM")
        self.fcm_dir = os.path.join(self.output_dir, "FCM")
        self.fv_dir = os.path.join(self.output_dir, "FV")
        self.fhm_dir_cog = os.path.join(self.output_dir, "FHM", "COG")
        self.fcm_dir_cog = os.path.join(self.output_dir, "FCM", "COG")
        self.fv_dir_cog = os.path.join(self.output_dir, "FV", "COG")
        self.magnitude_dir = os.path.join(self.temporary_dir, "magnitude")
        self.sentinel_dir = os.path.join(self.temporary_dir, "sentinel_input")

        # time range for the queries
        self.time_range_pre = (
            self.start_date_pre.strftime("%Y-%m-%d")
            + "/"
            + self.end_date_pre.strftime("%Y-%m-%d")
        )
        self.time_range_post = (
            self.start_date_post.strftime("%Y-%m-%d")
            + "/"
            + self.end_date_post.strftime("%Y-%m-%d")
        )

        self.logger.debug("Pre time range: {}".format(self.time_range_pre))
        self.logger.debug("Post time range: {}".format(self.time_range_post))

        # defines products names
        self.svc_number = self.chain.replace("SVC", "")[:2]
        self.svc_letter = self.chain.replace("SVC", "")[-1]
        if self.svc_number == "10" and self.assessment_kind == "weekly":
            self.product_number = "01"
        elif (self.svc_number == "10" and self.assessment_kind == "yearly") or (
            self.svc_number == "07" and self.svc_letter == "B"
        ):
            self.product_number = "02"
        elif self.svc_number == "07" and self.svc_letter == "C":
            self.product_number = "03"
        self.product_id = (
            "OU-S4" + "-{}".format(self.svc_number) + "-{}".format(self.product_number)
        )

        self.start_date_pre_filename = str(self.start_date_pre.date()).replace("-", "")
        self.end_date_pre_filename = str(self.end_date_pre.date()).replace("-", "")
        self.start_date_post_filename = str(self.start_date_post.date()).replace(
            "-", ""
        )
        self.end_date_post_filename = str(self.end_date_post.date()).replace("-", "")

        self.sentinel_pre_filename = os.path.join(
            self.sentinel_dir,
            "ds_{}_{}.zarr".format(
                self.aoi_name, self.time_range_pre.replace("/", "_")
            ),
        )
        self.sentinel_post_filename = os.path.join(
            self.sentinel_dir,
            "ds_{}_{}.zarr".format(
                self.aoi_name, self.time_range_post.replace("/", "_")
            ),
        )

        self.medoid_pre_filename = os.path.join(
            self.input_dir,
            "{}_{}_{}_pre_medoid.tif".format(
                aoi_name, self.start_date_pre_filename, self.end_date_pre_filename
            ),
        )
        self.medoid_post_filename = os.path.join(
            self.input_dir,
            "{}_{}_{}_post_medoid.tif".format(
                aoi_name, self.start_date_post_filename, self.end_date_post_filename
            ),
        )

        self.ndmi_pre_filename = os.path.join(
            self.input_dir,
            "{}_{}_pre_ndmi.tif".format(aoi_name, self.start_date_pre_filename),
        )
        self.nbr_pre_filename = os.path.join(
            self.input_dir,
            "{}_{}_pre_nbr.tif".format(aoi_name, self.start_date_pre_filename),
        )
        self.msi_pre_filename = os.path.join(
            self.input_dir,
            "{}_{}_pre_msi.tif".format(aoi_name, self.start_date_pre_filename),
        )
        self.ndmi_post_filename = os.path.join(
            self.input_dir,
            "{}_{}_post_ndmi.tif".format(aoi_name, self.start_date_post_filename),
        )
        self.nbr_post_filename = os.path.join(
            self.input_dir,
            "{}_{}_post_nbr.tif".format(aoi_name, self.start_date_post_filename),
        )
        self.msi_post_filename = os.path.join(
            self.input_dir,
            "{}_{}_post_msi.tif".format(aoi_name, self.start_date_post_filename),
        )

        if self.assessment_kind == "weekly":
            self.start_date_pre_filename = str(
                (self.start_date_pre + pd.DateOffset(days=+14)).date()
            ).replace("-", "")
            self.end_date_post_filename = str(
                (self.start_date_pre + pd.DateOffset(days=+21)).date()
            ).replace("-", "")

        self.magnitude_filename = os.path.join(
            self.magnitude_dir,
            self.product_id
            + "_Magnitude"
            + "_1{}".format(assessment_kind[0])
            + "_{}".format(aoi_name)
            + "_{}T000000".format(self.start_date_pre_filename)
            + "_{}T000000".format(self.end_date_post_filename)
            + ".tif",
        )

        self.classes_filename = os.path.join(
            self.fhm_dir,
            self.product_id
            + "_ForestHealthMap"
            + "_1{}".format(assessment_kind[0])
            + "_{}".format(aoi_name)
            + "_{}T000000".format(self.start_date_pre_filename)
            + "_{}T000000".format(self.end_date_post_filename)
            + ".tif",
        )

        self.classes_filename_cog = os.path.join(
            self.fhm_dir_cog,
            self.product_id
            + "_ForestHealthMap"
            + "_1{}".format(assessment_kind[0])
            + "_{}".format(aoi_name)
            + "_{}T000000".format(self.start_date_pre_filename)
            + "_{}T000000".format(self.end_date_post_filename)
            + ".tif",
        )

        self.vulnerability_filename = os.path.join(
            self.fv_dir,
            self.product_id
            + "_ForestVulnerabilityMap"
            + "_1{}".format(assessment_kind[0])
            + "_{}".format(aoi_name)
            + "_{}T000000".format(self.start_date_pre_filename)
            + "_{}T000000".format(self.end_date_post_filename)
            + ".tif",
        )

        self.vulnerability_filename_cog = os.path.join(
            self.fv_dir_cog,
            self.product_id
            + "_ForestVulnerabilityMap"
            + "_1{}".format(assessment_kind[0])
            + "_{}".format(aoi_name)
            + "_{}T000000".format(self.start_date_pre_filename)
            + "_{}T000000".format(self.end_date_post_filename)
            + ".tif",
        )

        self.time_ranges = [self.time_range_pre, self.time_range_post]
        self.sentinel_path_list = [
            self.sentinel_pre_filename,
            self.sentinel_post_filename,
        ]
        self.start_dates = [self.start_date_pre, self.start_date_post]
        self.end_dates = [self.end_date_pre, self.end_date_post]
        self.medoid_paths = [self.medoid_pre_filename, self.medoid_post_filename]
        self.medoid_flags = [self.save_medoid_pre, self.save_medoid_post]
        self.indexes_pre_filename = [
            self.ndmi_pre_filename,
            self.nbr_pre_filename,
            self.msi_pre_filename,
        ]
        self.indexes_post_filename = [
            self.ndmi_post_filename,
            self.nbr_post_filename,
            self.msi_post_filename,
        ]
        self.index_paths = [self.indexes_pre_filename, self.indexes_post_filename]
        self.index_flags = [self.save_indexes_pre, self.save_indexes_post]

        # in the case of SVC07B also third image and additional operations are needed
        if self.chain == "SVC07B":
            self.th = th
            self.mmu = mmu
            self.time_range_mid = (
                self.start_date_mid.strftime("%Y-%m-%d")
                + "/"
                + self.end_date_mid.strftime("%Y-%m-%d")
            )
            self.logger.debug("Mid time range: {}".format(self.time_range_mid))
            self.save_medoid_mid = self.save_to_disk["save_medoid_mid"]
            self.save_indexes_mid = self.save_to_disk["save_indexes_mid"]
            self.start_date_mid_filename = str(self.start_date_mid.date()).replace(
                "-", ""
            )
            self.end_date_mid_filename = str(self.end_date_mid.date()).replace("-", "")
            self.sentinel_mid_filename = os.path.join(
                self.sentinel_dir,
                "ds_{}_{}.zarr".format(
                    self.aoi_name, self.time_range_mid.replace("/", "_")
                ),
            )
            self.medoid_mid_filename = os.path.join(
                self.input_dir,
                "{}_{}_{}_mid_medoid.tif".format(
                    self.aoi_name,
                    self.start_date_mid_filename,
                    self.end_date_mid_filename,
                ),
            )
            self.ndmi_mid_filename = os.path.join(
                self.input_dir,
                "{}_{}_mid_ndmi.tif".format(aoi_name, self.start_date_mid_filename),
            )
            self.nbr_mid_filename = os.path.join(
                self.input_dir,
                "{}_{}_mid_nbr.tif".format(aoi_name, self.start_date_mid_filename),
            )
            self.msi_mid_filename = os.path.join(
                self.input_dir,
                "{}_{}_mid_msi.tif".format(aoi_name, self.start_date_mid_filename),
            )

            self.time_ranges = self.time_ranges + [self.time_range_mid]
            self.sentinel_path_list = self.sentinel_path_list + [
                self.sentinel_mid_filename
            ]
            self.start_dates = self.start_dates + [self.start_date_mid]
            self.end_dates = self.end_dates + [self.end_date_mid]
            self.medoid_paths = self.medoid_paths + [self.medoid_mid_filename]
            self.medoid_flags = self.medoid_flags + [self.save_medoid_mid]
            self.indexes_mid_filename = [
                self.ndmi_mid_filename,
                self.nbr_mid_filename,
                self.msi_mid_filename,
            ]
            self.index_paths = self.index_paths + [self.indexes_mid_filename]
            self.index_flags = self.index_flags + [self.save_indexes_mid]

            self.change_filename = os.path.join(
                self.fcm_dir,
                self.product_id
                + "_ForestChangeMap"
                + "_1{}".format(assessment_kind[0])
                + "_{}".format(aoi_name)
                + "_{}T000000".format(self.start_date_mid_filename)
                + "_{}T000000".format(self.end_date_mid_filename)
                + ".tif",
            )

            self.change_filename_cog = os.path.join(
                self.fcm_dir_cog,
                self.product_id
                + "_ForestChangeMap"
                + "_1{}".format(assessment_kind[0])
                + "_{}".format(aoi_name)
                + "_{}T000000".format(self.start_date_mid_filename)
                + "_{}T000000".format(self.end_date_mid_filename)
                + ".tif",
            )

        # flag that allows to skip time ranges that have already been completed
        self.skip_timerange = False
        if (self.chain in ["SVC07C", "SVC10C"]) and (self.assessment_kind == "yearly"):
            if os.path.exists(self.vulnerability_filename):
                self.skip_timerange = True
        elif self.chain == "SVC07B":
            if os.path.exists(self.change_filename):
                self.skip_timerange = True
        else:
            if os.path.exists(self.classes_filename):
                self.skip_timerange = True

    def create_directories(self):
        """
        Method: create_directories

        This method creates directories for different purposes based on the provided parameters. It ensures that the directories are created if they don't already exist.

        Parameters:
        - self: The object instance itself.

        Returns:
        - None

        """
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.temporary_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.chain in ["SVC07B"]:
            os.makedirs(self.fcm_dir, exist_ok=True)
            os.makedirs(self.fcm_dir_cog, exist_ok=True)
        if self.chain in ["SVC07C", "SVC10C"]:
            os.makedirs(self.fhm_dir, exist_ok=True)
            os.makedirs(self.fhm_dir_cog, exist_ok=True)
            if self.assessment_kind == "yearly":
                os.makedirs(self.fv_dir, exist_ok=True)
                os.makedirs(self.fv_dir_cog, exist_ok=True)
        os.makedirs(self.magnitude_dir, exist_ok=True)
        os.makedirs(self.sentinel_dir, exist_ok=True)

    @retry(RuntimeError, tries=10, delay=2)
    @timed
    def prepare_input_datsets(self):
        """
        Prepares Sentinel 2 Level 2A datasets.

        Retry decorator is applied to handle possible runtime errors and retry the method execution up to 10 times with a delay of 2 seconds between each attempt.
        Timed decorator is applied to measure the execution time of the method.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            list: A list of xarray datasets representing the prepared Sentinel 2 Level 2A datasets.
        """
        self.logger.info("Preparing Sentinel 2 L2A datasets.")
        ds_list = []
        # queries sentinel data from Microsoft Planetary
        for time_range, path in zip(self.time_ranges, self.sentinel_path_list):
            ds = stac_to_dataset(
                aoi=self.aoi,
                time_range=time_range,
                collections=self.collections,
                query=self.query,
                assets=self.assets,
                epsg=self.stac_epsg,
                resolution=self.resolution,
                nodata=self.nodata,
            )
            """""
            epsg = ds.rio.crs.to_string()
            self.clip_geometry = self.clip_geometry.to_crs(epsg)
            ds = ds.rio.write_crs(epsg).astype('float32').rio.clip_box(minx=self.clip_geometry.bounds.minx,
                                            miny=self.clip_geometry.bounds.miny,
                                            maxx=self.clip_geometry.bounds.maxx,
                                            maxy=self.clip_geometry.bounds.maxy,
                                            crs=self.clip_geometry.crs)
            for i, time in enumerate(ds.time.values[:6]):
                id = ds['id'].values[i]
                tif_path = os.path.join(self.sentinel_dir, '{}_pre_mask.tif'.format(id))
                ds.isel(time=i).to_array('band').rio.to_raster(tif_path)
            """
            # saving the input dataset and reading it from disk is more stable for dask delayed computations
            if self.save_input_dataset:
                save_input_zarr(
                    [ds], [path], self.overwrite, self.clip_geometry, self.stac_epsg
                )

            ds = xr.open_zarr(path)
            self.logger.debug(
                "Dataset dimensions: time, x, y = {}, {}, {}".format(
                    len(ds.time), len(ds.x), len(ds.y)
                )
            )
            ds_list.append(ds)
        return ds_list

    def baseline_adjustment(self, ds_list):
        """
        Adjusts the baseline of the dataset list based on a specified cutoff date.

        :param ds_list: A list of datasets.
        :type ds_list: list
        :return: A list of datasets with the baseline adjusted.
        :rtype: list
        """
        self.logger.info("Adjusting baseline.")
        # adjustment for sentinel 2 baseline change
        cutoff = datetime.datetime(2022, 1, 25)

        ds_list_harmonized = []
        for ds, start_date, end_date in zip(ds_list, self.start_dates, self.end_dates):
            start_before_cutoff = start_date <= cutoff
            start_after_cutoff = start_date > cutoff
            end_before_cutoff = end_date <= cutoff
            end_after_cutoff = end_date > cutoff

            if start_after_cutoff or (start_before_cutoff and end_after_cutoff):
                ds_harmonized = harmonize_to_old(ds)
            else:
                ds_harmonized = ds
            ds_list_harmonized.append(ds_harmonized)

        return ds_list_harmonized

    def mask_datasets(self, ds_list):
        """
        Masks the datasets in the given list.

        Parameters:
        - ds_list: A list of xarray Dataset objects. Each Dataset represents a dataset to be masked.

        Returns:
        - ds_list_masked: A list of masked xarray Dataset objects. Each Dataset in the list has been masked.
        """
        self.logger.info("Masking datasets.")
        # masks the datasets
        ds_list_masked = []
        for ds in ds_list:
            ds_masked = s2l2a_masked(ds).drop_vars("SCL")
            ds_list_masked.append(ds_masked)
            epsg = ds.rio.crs.to_string()
            self.clip_geometry = self.clip_geometry.to_crs(epsg)
            ds_masked = ds_masked.astype("float32").rio.clip_box(
                minx=self.clip_geometry.bounds.minx,
                miny=self.clip_geometry.bounds.miny,
                maxx=self.clip_geometry.bounds.maxx,
                maxy=self.clip_geometry.bounds.maxy,
                crs=self.clip_geometry.crs,
            )
            """
            for i, time in enumerate(ds_masked.time.values[:6]):
                id = ds_masked['id'].values[i]
                tif_path = os.path.join(self.sentinel_dir, '{}_post_mask.tif'.format(id))
                ds_masked.isel(time=i).to_array('band').rio.to_raster(tif_path)
            """

        return ds_list_masked

    @retry(RuntimeError, tries=10, delay=2)
    @timed
    def compute_medoids(self, ds_list):
        """
        This method computes the medoids for a given list of datasets.

        Parameters:
        - ds_list: list of xarray.Dataset objects. Each dataset represents a time series of raster data.

        Returns:
        - ds_list_medoid: list of xarray.Dataset objects. Each dataset represents the computed medoid for the corresponding input dataset.

        """
        self.logger.info("Computing medoids.")
        # computes the medoids
        ds_list_medoid = []
        for ds, path, flag in zip(ds_list, self.medoid_paths, self.medoid_flags):
            if (not os.path.exists(path)) or self.overwrite:
                self.logger.debug(
                    "Computing medoid of times: {}.".format(ds.time.values)
                )
                ds_medoid = medoid_mosaic(ds).compute()
                if flag:
                    ds_medoid.rio.write_crs(
                        "EPSG:{}".format(self.stac_epsg)
                    ).rio.to_raster(path)
            else:
                ds_medoid = rio.open_rasterio(path)
            ds_list_medoid.append(ds_medoid)
        return ds_list_medoid

    @retry(RuntimeError, tries=10, delay=2)
    @timed
    def compute_indexes(self, ds_list):
        """
        Computes and saves the indexes for each dataset in ds_list.

        The method calculates the Normalized Difference Moisture Index (NDMI),
        Normalized Burn Ratio (NBR), and Multispectral Similarity Index (MSI)
        for each dataset in ds_list. The indexes are then saved to the specified
        paths.

        Args:
            ds_list (list): A list of datasets.

        Returns:
            list: A list containing the computed indexes.

        Raises:
            RuntimeError: If the computation of indexes fails.
            RetryError: If the computation of indexes fails after multiple retries.

        """
        self.logger.info("Computing indexes.")
        # calculates and saves the indexes
        ds_list_index = []
        for i, ds, flag in zip(range(len(ds_list)), ds_list, self.index_flags):
            ndmi, nbr, msi = calculate_ndmi_nbr_msi(ds)
            ds_list_index.append(ndmi)
            ds_list_index.append(nbr)
            ds_list_index.append(msi)
            for index, path in zip([ndmi, nbr, msi], self.index_paths[i]):
                if flag:
                    index.rio.write_crs("EPSG:{}".format(self.stac_epsg)).rio.to_raster(
                        path
                    )
        return ds_list_index

    @retry(RuntimeError, tries=10, delay=2)
    @timed
    def compute_assessment(self):
        """
        This method is the entry point for computing the assessment of"""
        # skips time ranges that have already been completed
        if self.skip_timerange:
            self.logger.debug("Skipping time range.")
            return

        self.create_directories()
        ds_list = self.prepare_input_datsets()
        ds_list_masked = self.mask_datasets(ds_list)
        ds_list_harmonized = self.baseline_adjustment(ds_list_masked)
        ds_list_medoid = self.compute_medoids(ds_list_harmonized)

        # conversion to xarray dataset if needed
        ds_list = []
        for ds in ds_list_medoid:
            if not isinstance(ds, xr.Dataset):
                ds = datarray2dataset(ds)
                ds_list.append(ds)
            else:
                ds_list.append(ds)

        ds_list_index = self.compute_indexes(ds_list)

        # computes the change magnitude
        self.logger.info("Computing change magnitude.")
        if self.chain == "SVC07B":
            (
                ndmi_pre,
                nbr_pre,
                msi_pre,
                ndmi_post,
                nbr_post,
                msi_post,
                ndmi_mid,
                nbr_mid,
                msi_mid,
            ) = ds_list_index
            magnitude = calculate_magnitude_trio(
                ndmi_pre,
                nbr_pre,
                msi_pre,
                ndmi_mid,
                nbr_mid,
                msi_mid,
                ndmi_post,
                nbr_post,
                msi_post,
            )
            # classifies the change magnitude
            self.logger.info("Classifying the change magnitude.")
            change = compute_change(
                magnitude,
                self.th,
                self.fn_path,
                self.change_filename,
                self.change_filename_cog,
                self.stac_epsg,
                self.epsg,
                self.save_classification,
                self.clip_geometry,
                self.overwrite,
                self.mmu,
            )
        else:
            ndmi_pre, nbr_pre, msi_pre, ndmi_post, nbr_post, msi_post = ds_list_index
            magnitude = calculate_magnitude(
                ndmi_pre, nbr_pre, msi_pre, ndmi_post, nbr_post, msi_post
            )

            # classifies the change magnitude
            self.logger.info("Classifying the change magnitude.")
            classes = compute_classification(
                magnitude,
                self.fn_path,
                self.classes_filename,
                self.classes_filename_cog,
                self.stac_epsg,
                self.epsg,
                self.save_classification,
                self.clip_geometry,
                self.overwrite,
            )

            # computes the vulnerability map
            self.logger.info("Computing the vulnerability map.")
            if (self.chain in ["SVC07C", "SVC10C"]) and (
                self.assessment_kind == "yearly"
            ):
                compute_vulnerability(
                    classes,
                    self.ft_path,
                    self.ft_cod_path,
                    self.vulnerability_filename,
                    self.vulnerability_filename_cog,
                    self.epsg,
                    self.save_classification,
                    self.clip_geometry,
                    self.overwrite,
                )


def get_config(config_file: str = "config.ini"):
    config = configparser.ConfigParser()
    path_current_directory = os.path.dirname(__file__)
    path_config_file = os.path.join(path_current_directory, config_file)
    config.read(path_config_file)

    config_dict = dict(config["COMMON"])

    chain = config_dict["chain"]
    assessment_kind = config_dict["assessment_kind"]
    aoi = config_dict["aoi"]

    assert chain in ["SVC10C", "SVC07B", "SVC07C"], chain
    assert aoi in ["AOI1", "AOI2", "AOI3"], aoi
    if chain in ["SVC07B", "SVC07C"]:
        assert assessment_kind == "yearly", assessment_kind
    elif chain == "SVC10C":
        assert assessment_kind in ["weekly", "yearly"], assessment_kind

    for k, v in dict(config[chain + "_" + assessment_kind]).items():
        config_dict[k] = v

    return config_dict
