import os
import time
import warnings

import geopandas as gpd
import pandas as pd
from dask.distributed import Client
from pyproj import Transformer

from chains.module import get_config, health_assessment

warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    tstart = time.time()

    config_dict = get_config(config_file="config.ini")

    root = config_dict["root"]
    chain = config_dict["chain"]
    shp_relative_path = config_dict["shp_relative_path"]
    assessment_kind = config_dict["assessment_kind"]
    aoi = config_dict["aoi"]
    epsg_output = config_dict["epsg_output"]
    cloud_cover_threshold = float(config_dict["cloud_cover_threshold"])
    clip_aoi = config_dict["clip_aoi"] == "True"
    save_input_dataset = config_dict["save_input_dataset"] == "True"
    save_classification = config_dict["save_classification"] == "True"
    dask_n_workers = int(config_dict["dask_n_workers"])
    dask_memory_limit = config_dict["dask_memory_limit"]
    dask_dashboard_address = ":" + config_dict["dask_dashboard_address"]
    overwrite = config_dict["overwrite"] == "True"
    start_date = config_dict["start_date"]
    log_level = config_dict["log_level"]
    if "min_3i3d_magnitude" in config_dict.keys():
        th = float(config_dict["min_3i3d_magnitude"])
    else:
        th = None
    if "minimum_change_size" in config_dict.keys():
        mmu = int(config_dict["minimum_change_size"])
        assert mmu > 0, "Minimum change size must be >0."
    else:
        mmu = None
    if "end_date" in config_dict.keys():
        end_date = config_dict["end_date"]
    offset = [int(off) for off in config_dict["offset"].split(",")]

    shp_path = os.path.join(root, shp_relative_path)
    chain_path = os.path.join(root, "data_processing", chain)
    results_path = os.path.join(chain_path, assessment_kind)
    save_to_disk = {
        "save_input_dataset": save_input_dataset,
        "save_classification": save_classification,
    }

    # Dictionary containing the sub areas of interest and their corresponding selected S2 tiles
    aoi_dict = {
        "AOI1": {
            "shapefile": "AOI_SVC_S4-07C-10_AOI1.shp",  #'aoi_cloud_test.shp',
            "mgrs_tile": ["33TTG"],  # ,'T32TQM'],
            "fnf_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "lazio_ft.tif"
            ),
            "ft_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "lazio_ft.tif"
            ),
            "ft_cod_relative_path": os.path.join(
                "PM-2.05_Other_data_processing",
                "carteforestali",
                "Legend_S4_07_10_qd_RB.xlsx",
            ),
        },
        "AOI2": {
            "shapefile": "AOI_SVC_S4-07C-10_AOI2.shp",
            "mgrs_tile": "32TQS",
            "fnf_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "trentino_ft.tif"
            ),
            "ft_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "trentino_ft.tif"
            ),
            "ft_cod_relative_path": os.path.join(
                "PM-2.05_Other_data_processing",
                "carteforestali",
                "Legend_S4_07_10_qd_RB.xlsx",
            ),
        },
        "AOI3": {
            "shapefile": "AOI_SVC_S4-07B_AOI3.shp",
            "mgrs_tile": "33TUG",
            "fnf_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "abruzzo_ft.tif"
            ),
            "ft_relative_path": os.path.join(
                "PM-2.05_Other_data_processing", "carteforestali", "abruzzo_ft.tif"
            ),
            "ft_cod_relative_path": os.path.join(
                "PM-2.05_Other_data_processing",
                "carteforestali",
                "Legend_S4_07_10_qd_RB.xlsx",
            ),
        },
    }

    # selects only one of the tiles covering the above aoi
    mgrs_tile = aoi_dict[aoi]["mgrs_tile"]
    aoi_geometry = (
        gpd.read_file(os.path.join(shp_path, aoi_dict[aoi]["shapefile"]))
        .dissolve()
        .geometry
    )
    aoi_geometry_4326 = (
        gpd.read_file(os.path.join(shp_path, aoi_dict[aoi]["shapefile"]))
        .to_crs(4326)
        .dissolve()
        .geometry
    )
    fnf_relative_path = aoi_dict[aoi]["fnf_relative_path"]
    ft_relative_path = aoi_dict[aoi]["ft_relative_path"]
    ft_cod_relative_path = aoi_dict[aoi]["ft_cod_relative_path"]
    fn_path = os.path.join(root, fnf_relative_path)
    ft_path = os.path.join(root, ft_relative_path)
    ft_cod_path = os.path.join(root, ft_cod_relative_path)

    # dictionary containing the area of interests and the corresponding tiles used to perform the STAC query
    lonmin, latmin, lonmax, latmax = aoi_geometry_4326.total_bounds
    aoi_stac = {
        "type": "Polygon",
        "coordinates": [
            [
                [lonmin, latmin],
                [lonmax, latmin],
                [lonmax, latmax],
                [lonmin, latmax],
                [lonmin, latmin],
            ]
        ],
    }

    epsg = int(epsg_output.split(":")[1])  # reproject output to this epsg
    transformer = Transformer.from_crs(
        crs_from="EPSG:4326", crs_to=epsg, always_xy=True
    )

    if clip_aoi:
        clip_geometry = aoi_geometry
    else:
        clip_geometry = None

    time_ranges = []
    if assessment_kind == "yearly":
        if chain == "SVC10C":
            start_date_pre = pd.to_datetime(start_date) + pd.DateOffset(
                years=-offset[-1]
            )
            end_date_pre = pd.to_datetime(end_date) + pd.DateOffset(years=-offset[-1])
            for years_offset in offset:
                start_date_post = pd.to_datetime(start_date_pre) + pd.DateOffset(
                    years=years_offset
                )
                end_date_post = pd.to_datetime(end_date_pre) + pd.DateOffset(
                    years=years_offset
                )
                time_ranges.append(
                    (start_date_pre, end_date_pre, start_date_post, end_date_post)
                )
        elif chain == "SVC07C":
            start_date_post = pd.to_datetime(start_date)
            end_date_post = pd.to_datetime(end_date)
            for years_offset in offset:
                start_date_pre = pd.to_datetime(start_date_post) + pd.DateOffset(
                    years=-years_offset
                )
                end_date_pre = pd.to_datetime(end_date_post) + pd.DateOffset(
                    years=-years_offset
                )
                time_ranges.append(
                    (start_date_pre, end_date_pre, start_date_post, end_date_post)
                )
        elif chain == "SVC07B":
            for years_offset in offset:
                start_date_pre = pd.to_datetime(start_date) + pd.DateOffset(
                    years=-years_offset
                )
                end_date_pre = pd.to_datetime(end_date) + pd.DateOffset(
                    years=-years_offset
                )
                start_date_post = pd.to_datetime(start_date) + pd.DateOffset(
                    years=years_offset
                )
                end_date_post = pd.to_datetime(end_date) + pd.DateOffset(
                    years=years_offset
                )
                start_date_mid = pd.to_datetime(start_date)
                end_date_mid = pd.to_datetime(end_date)
                time_ranges.append(
                    (
                        start_date_pre,
                        end_date_pre,
                        start_date_mid,
                        end_date_mid,
                        start_date_post,
                        end_date_post,
                    )
                )
    elif assessment_kind == "weekly":
        for days_offset in offset:
            # selects a time range for the weekly assessment
            date = pd.to_datetime(start_date) + pd.DateOffset(days=days_offset)
            start_date_pre = date + pd.DateOffset(days=-14)
            end_date_pre = date + pd.DateOffset(
                days=14, hours=23, minutes=59, seconds=59
            )
            start_date_post = date + pd.DateOffset(days=15)
            end_date_post = date + pd.DateOffset(
                days=29, hours=23, minutes=59, seconds=59
            )
            time_ranges.append(
                (start_date_pre, end_date_pre, start_date_post, end_date_post)
            )

    client = Client(
        dashboard_address=dask_dashboard_address,
        n_workers=dask_n_workers,
        memory_limit=dask_memory_limit,
    )
    print("Dask dashboard url: {}".format(client.dashboard_link))
    print("Starting chain {} on AOI {}.".format(chain, aoi))

    for i, time_range in enumerate(time_ranges):
        print("Time range: {}.".format(time_range))
        if chain == "SVC07B":
            (
                start_date_pre,
                end_date_pre,
                start_date_mid,
                end_date_mid,
                start_date_post,
                end_date_post,
            ) = time_range
        else:
            start_date_mid = None
            end_date_mid = None
            start_date_pre, end_date_pre, start_date_post, end_date_post = time_range

        isfirst = i == 0
        islast = i == len(time_ranges) - 1
        isyearly = assessment_kind == "yearly"
        """
        if (isfirst or islast) and isyearly:
            if chain == 'SVC07B':
                save_to_disk['save_medoid_pre'] = True
                save_to_disk['save_medoid_post'] = True
                save_to_disk['save_medoid_mid'] = True
                save_to_disk['save_indexes_pre'] = True
                save_to_disk['save_indexes_post'] = True
                save_to_disk['save_indexes_mid'] = True
            elif chain == 'SVC07C':
                save_to_disk['save_medoid_pre'] = True
                save_to_disk['save_medoid_post'] = True
                save_to_disk['save_indexes_pre'] = False
                save_to_disk['save_indexes_post'] = False
            elif chain == 'SVC10C' and isfirst:
                save_to_disk['save_medoid_pre'] = True
                save_to_disk['save_medoid_post'] = False
                save_to_disk['save_indexes_pre'] = True
                save_to_disk['save_indexes_post'] = False
            elif chain == 'SVC10C' and islast:
                save_to_disk['save_medoid_pre'] = False
                save_to_disk['save_medoid_post'] = True
                save_to_disk['save_indexes_pre'] = False
                save_to_disk['save_indexes_post'] = True
        else:
            save_to_disk['save_medoid_pre'] = False
            save_to_disk['save_medoid_post'] = False
            save_to_disk['save_indexes_pre'] = False
            save_to_disk['save_indexes_post'] = False
        """
        save_to_disk["save_medoid_pre"] = True
        save_to_disk["save_medoid_post"] = True
        save_to_disk["save_medoid_mid"] = True
        save_to_disk["save_indexes_pre"] = True
        save_to_disk["save_indexes_post"] = True
        save_to_disk["save_indexes_mid"] = True

        ha = health_assessment(
            start_date_pre=start_date_pre,
            end_date_pre=end_date_pre,
            start_date_mid=start_date_mid,
            end_date_mid=end_date_mid,
            start_date_post=start_date_post,
            end_date_post=end_date_post,
            aoi=aoi_stac,
            clip_geometry=clip_geometry,
            mgrs_tile=mgrs_tile,
            epsg=epsg,
            cloudcover=cloud_cover_threshold,
            save_to_disk=save_to_disk,
            results_path=results_path,
            aoi_name=aoi,
            assessment_kind=assessment_kind,
            chain=chain,
            fn_path=fn_path,
            ft_path=ft_path,
            ft_cod_path=ft_cod_path,
            th=th,
            mmu=mmu,
            overwrite=overwrite,
            log_level=log_level,
        )

        ha.compute_assessment()

    tend = time.time()
    delta_t = tend - tstart
    print("Total computation time: {}s".format(delta_t))
