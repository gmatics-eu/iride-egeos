import pandas as pd
import planetary_computer
import pystac_client
import pytest
import stackstac
from affine import Affine
from distributed import Client
import numpy as np
from stackstac.raster_spec import RasterSpec
from pyproj import Transformer
import xarray as xr

client = Client()

class TestForestHealth:
    @pytest.fixture
    def query_to_ds(self):
        lonmin = 12.340169208743843
        latmin = 41.70483827786684
        lonmax = 12.389092701175484
        latmax = 41.718165463023546
        aoi = {
            "type": "Polygon",
            "coordinates": [
                [
                    [lonmin, latmin],
                    [lonmax, latmin],
                    [lonmax, latmax],
                    [lonmin, latmax],
                    [lonmin, latmin]
                ]
            ],
        }
        mgrs_tile = '33TTG'
        cloudcover = 70
        collections = ['sentinel-2-l2a']
        query = {"eo:cloud_cover": {"lt": cloudcover},
                 "s2:mgrs_tile": {"eq": mgrs_tile}}

        # bands of interest
        assets = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'SCL']
        epsg = 32633
        transformer = Transformer.from_crs(crs_from="EPSG:4326", crs_to=epsg, always_xy=True)
        xmin, ymin = transformer.transform(lonmin, latmin)
        xmax, ymax = transformer.transform(lonmax, latmax)
        bbox = (xmin, xmax, ymin, ymax)
        resolution = 10
        nodata = 0
        date = pd.to_datetime('2020-03-01') + pd.DateOffset(days=98)
        start_date_pre = date + pd.DateOffset(days=-14)
        end_date_pre = date + pd.DateOffset(days=14)
        time_range = start_date_pre.strftime('%Y-%m-%d') + '/' + end_date_pre.strftime('%Y-%m-%d')

        stac = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = stac.search(
            intersects=aoi,
            datetime=time_range,
            collections=collections,
            query=query,
        )
        items = search.item_collection()
        data = (
            stackstac.stack(
                items,
                assets=assets,
                epsg=epsg,
                resolution=resolution,
            )
            .where(lambda x: x != nodata, other=np.nan)  # sentinel-2 uses 0 as nodata
        )
        ds = data.to_dataset('band')

        ds = ds.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)).load()

        return ds

    def test_data_vars(self, query_to_ds):
        assert list(query_to_ds.data_vars) == ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12',
                                               'SCL'], \
            "The query returned a Dataset with wrong data vars"

    def test_coords(self, query_to_ds):
        ds_coords = sorted(query_to_ds.coords)
        assert ds_coords == ['center_wavelength', 'common_name', 'constellation', 'eo:cloud_cover', 'epsg',
                             'full_width_half_max', 'gsd', 'id', 'instruments', 'platform', 'proj:bbox', 'proj:epsg',
                             's2:cloud_shadow_percentage', 's2:dark_features_percentage', 's2:datastrip_id',
                             's2:datatake_id', 's2:datatake_type', 's2:degraded_msi_data_percentage',
                             's2:generation_time', 's2:granule_id', 's2:high_proba_clouds_percentage',
                             's2:mean_solar_azimuth', 's2:mean_solar_zenith', 's2:medium_proba_clouds_percentage',
                             's2:mgrs_tile', 's2:nodata_pixel_percentage', 's2:not_vegetated_percentage',
                             's2:processing_baseline', 's2:product_type', 's2:product_uri',
                             's2:reflectance_conversion_factor', 's2:saturated_defective_pixel_percentage',
                             's2:snow_ice_percentage', 's2:thin_cirrus_percentage', 's2:unclassified_percentage',
                             's2:vegetation_percentage', 's2:water_percentage', 'sat:orbit_state',
                             'sat:relative_orbit', 'time', 'title', 'x', 'y'], \
            "The query returned a Dataset with wrong dims"

    def test_attrs(self, query_to_ds):
        assert query_to_ds.attrs == {'crs': 'epsg:32633',
                                     'resolution': 10,
                                     'spec': RasterSpec(epsg=32633,
                                                        bounds=(199980, 4590240, 309780, 4700040),
                                                        resolutions_xy=(10, 10)),
                                     'transform': Affine(10.0, 0.0, 199980.0, 0.0, -10.0, 4700040.0)}, \
            "The query returned a Dataset with wrong attrs"

    def test_dims(self, query_to_ds):
        assert dict(query_to_ds.dims) == {'band': 11, 'time': 6, 'x': 412, 'y': 135}, \
            "The query returned a Dataset with wrong dims"

    def test_isel_func(self, query_to_ds):
        from scripts.utils import isel_func
        ds = isel_func(query_to_ds, 0)
        assert dict(ds.dims) == {'band': 11, 'x': 412, 'y': 135}, \
            "The isel_func is not working as expected - it returned a Dataset with wrong dims"

    def test_coords_to_attrs(self, query_to_ds):
        from scripts.utils import coords_to_attrs

        cleaned_ds = coords_to_attrs(query_to_ds)
        assert list(cleaned_ds.coords) == ['time', 'x', 'y'], \
            "The coords_to_attrs function returned a Dataset with incorrect coords"

        expected = {'spec': RasterSpec(epsg=32633,
                                       bounds=(199980, 4590240, 309780, 4700040),
                                       resolutions_xy=(10, 10)),
                    'crs': 'epsg:32633',
                    'transform': Affine(10.0, 0.0, 199980.0, 0.0, -10.0, 4700040.0),
                    'resolution': 10,
                    'id': "['S2A_MSIL2A_20200525T100031_R122_T33TTG_20200910T215435'\n "
                          "'S2B_MSIL2A_20200530T100029_R122_T33TTG_20200911T133112'\n "
                          "'S2A_MSIL2A_20200604T100031_R122_T33TTG_20200826T023226'\n "
                          "'S2B_MSIL2A_20200609T100029_R122_T33TTG_20200826T185001'\n "
                          "'S2A_MSIL2A_20200614T100031_R122_T33TTG_20200827T081726'\n "
                          "'S2B_MSIL2A_20200619T100029_R122_T33TTG_20200823T083526']",
                    'constellation': 'Sentinel 2',
                    's2:generation_time': "['2020-09-10T21:54:35.655Z' "
                                          "'2020-09-11T13:31:12.107Z'\n "
                                          "'2020-08-26T02:32:26.944Z' "
                                          "'2020-08-26T18:50:01.574Z'\n "
                                          "'2020-08-27T08:17:26.590Z' '2020-08-23T08:35:26.776Z']",
                    's2:water_percentage': '[46.363291 46.811178 21.742529 '
                                           '46.179581 48.292616 17.635615]',
                    's2:datastrip_id':
                        "['S2A_OPER_MSI_L2A_DS_ESRI_20200910T215436_S20200525T100503_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_DS_ESRI_20200911T133114_S20200530T100502_N02.12'\n "
                        "'S2A_OPER_MSI_L2A_DS_ESRI_20200826T023229_S20200604T100905_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_DS_ESRI_20200826T185004_S20200609T100615_N02.12'\n "
                        "'S2A_OPER_MSI_L2A_DS_ESRI_20200827T081728_S20200614T100243_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_DS_ESRI_20200823T083528_S20200619T100854_N02.12']",
                    's2:unclassified_percentage': '[0.828401 3.021058 1.971672 '
                                                  '1.517702 2.284757 0.643153]',
                    's2:processing_baseline': '02.12',
                    's2:vegetation_percentage': '[28.744218 10.92589   6.165734 '
                                                '21.723017 16.37969  14.056589]',
                    's2:cloud_shadow_percentage': '[0.784367 1.111994 0.027813 '
                                                  '1.315905 1.053367 0.343944]',
                    's2:snow_ice_percentage': '[4.38000e-04 4.78724e-01 2.24370e-02 '
                                              '7.21530e-02 4.04210e-02 1.19000e-04]',
                    's2:product_uri':
                        "['S2A_MSIL2A_20200525T100031_N0212_R122_T33TTG_20200910T215435.SAFE'\n "
                        "'S2B_MSIL2A_20200530T100029_N0212_R122_T33TTG_20200911T133112.SAFE'\n "
                        "'S2A_MSIL2A_20200604T100031_N0212_R122_T33TTG_20200826T023226.SAFE'\n "
                        "'S2B_MSIL2A_20200609T100029_N0212_R122_T33TTG_20200826T185001.SAFE'\n "
                        "'S2A_MSIL2A_20200614T100031_N0212_R122_T33TTG_20200827T081726.SAFE'\n "
                        "'S2B_MSIL2A_20200619T100029_N0212_R122_T33TTG_20200823T083526.SAFE']",
                    's2:nodata_pixel_percentage': '[0.       0.       1.368748 '
                                                  '0.       0.       0.      ]',
                    'proj:epsg': '32633',
                    'sat:orbit_state': 'descending',
                    's2:mean_solar_zenith': '[24.32636335 23.69917644 23.22087375 22.92478747 '
                                            '22.77495748 22.79653068]',
                    'eo:cloud_cover': '[10.409597 32.043052 63.112731 12.419995 19.201538 58.264982]',
                    's2:degraded_msi_data_percentage': '0.0',
                    's2:mgrs_tile': '33TTG',
                    's2:granule_id':
                        "['S2A_OPER_MSI_L2A_TL_ESRI_20200910T215436_A025716_T33TTG_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_TL_ESRI_20200911T133114_A016879_T33TTG_N02.12'\n "
                        "'S2A_OPER_MSI_L2A_TL_ESRI_20200826T023229_A025859_T33TTG_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_TL_ESRI_20200826T185004_A017022_T33TTG_N02.12'\n "
                        "'S2A_OPER_MSI_L2A_TL_ESRI_20200827T081728_A026002_T33TTG_N02.12'\n "
                        "'S2B_OPER_MSI_L2A_TL_ESRI_20200823T083528_A017165_T33TTG_N02.12']",
                    's2:dark_features_percentage': '[0.313994 0.392281 0.082439 '
                                                   '0.701169 0.792005 0.193228]',
                    's2:product_type': 'S2MSI2A',
                    'platform': "['Sentinel-2A' 'Sentinel-2B' 'Sentinel-2A' "
                                "'Sentinel-2B' 'Sentinel-2A'\n 'Sentinel-2B']",
                    's2:reflectance_conversion_factor': '[0.97604796 0.97424025 0.97262376 '
                                                        '0.97120902 0.97000511 0.96901978]',
                    's2:datatake_id': "['GS2A_20200525T100031_025716_N02.12' "
                                      "'GS2B_20200530T100029_016879_N02.12'\n "
                                      "'GS2A_20200604T100031_025859_N02.12' "
                                      "'GS2B_20200609T100029_017022_N02.12'\n "
                                      "'GS2A_20200614T100031_026002_N02.12' "
                                      "'GS2B_20200619T100029_017165_N02.12']",
                    'instruments': 'msi',
                    's2:mean_solar_azimuth': '[144.42898589 143.14739235 142.00125089 '
                                             '140.88437752 139.98712359\n'
                                             ' 139.22322435]',
                    's2:datatake_type': 'INS-NOBS',
                    's2:high_proba_clouds_percentage': '[ 2.787781 24.197268 32.940415  '
                                                       '5.505609 14.44447   6.735336]',
                    's2:not_vegetated_percentage': '[12.555695  5.215822  6.874645 '
                                                   '16.070478 11.955604  8.86237 ]',
                    's2:medium_proba_clouds_percentage': '[ 0.672831  4.636843 15.724258  '
                                                         '2.242717  3.496306 31.577986]',
                    's2:saturated_defective_pixel_percentage': '0.0',
                    'sat:relative_orbit': '122',
                    's2:thin_cirrus_percentage': '[ 6.948985  3.208941 14.448059  '
                                                 '4.67167   1.260762 19.951659]',
                    'gsd': '[10. 10. 10. 20. 20. 20. 10. 20. 20. 20. 20.]',
                    'title': "['Band 2 - Blue - 10m' "
                             "'Band 3 - Green - 10m' "
                             "'Band 4 - Red - 10m'\n"
                             " 'Band 5 - Vegetation red edge 1 - 20m'\n"
                             " 'Band 6 - Vegetation red edge 2 - 20m'\n"
                             " 'Band 7 - Vegetation red edge 3 - 20m' "
                             "'Band 8 - NIR - 10m'\n"
                             " 'Band 8A - Vegetation red edge 4 - 20m' "
                             "'Band 11 - SWIR (1.6) - 20m'\n "
                             "'Band 12 - SWIR (2.2) - 20m' "
                             "'Scene classfication map (SCL)']",
                    'proj:bbox': '{4590240.0, 199980.0, 309780.0, 4700040.0}',
                    'common_name': "['blue' 'green' 'red' 'rededge' "
                                   "'rededge' 'rededge' 'nir' 'rededge'\n"
                                   " 'swir16' 'swir22' None]",
                    'center_wavelength': '[0.49 0.56 0.665 0.704 0.74 0.783 0.842 0.865 1.61 2.19 None]',
                    'full_width_half_max': '[0.098 0.045 0.038 0.019 0.018 '
                                           '0.028 0.145 0.033 0.143 0.242 None]',
                    'epsg': '32633'}
        expected_deserialized = {k: str(v) for k, v in expected.items()}
        actual_deserialized = {k: str(v) for k, v in cleaned_ds.attrs.items()}
        assert actual_deserialized == expected_deserialized, \
            "The coords_to_attrs function returned a Dataset with wrong attrs"

    def test_s2l2a_masked(self, query_to_ds):
        from scripts.utils import s2l2a_masked
        from scripts.utils import coords_to_attrs

        expected_ds = xr.open_zarr('./data/test_s2l2a_masked.zarr')
        actual_ds = s2l2a_masked(query_to_ds)
        actual_ds = coords_to_attrs(actual_ds)
        actual_ds.attrs = {k: str(v) for k, v in actual_ds.attrs.items()}
        assert expected_ds.equals(actual_ds), \
            "function s2l2a_masked returns different dataset"

    def test_medoid_mosaic(self, query_to_ds):
        from scripts.utils import s2l2a_masked
        from scripts.utils import coords_to_attrs
        from scripts.utils import medoid_mosaic

        expected_ds = xr.open_zarr('./data/test_medoid.zarr')

        actual_ds = s2l2a_masked(query_to_ds).drop_vars('SCL')
        actual_ds = coords_to_attrs(actual_ds)
        actual_ds.attrs = {k: str(v) for k, v in actual_ds.attrs.items()}
        actual_ds = medoid_mosaic(actual_ds)
        actual_ds.transpose('x', 'y')
        assert expected_ds.equals(actual_ds), \
            "function medoid_mosaic returns different dataset"
