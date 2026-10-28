"""
satrain_now.obs.goes
====================

Functionality to extract GOES observations.
"""
from functools import cache
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import click
from numcodecs.zarr3 import Blosc
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from tqdm import tqdm

from pansat.time import TimeRange
from pansat.utils import (
    resample_data,
    get_resample_info
)
from pansat.products.satellite.goes import (
    GOES16L1BRadiances,
    GOES17L1BRadiances,
    GOES18L1BRadiances,
    GOES19L1BRadiances,
)

from satrain_now.domains import get_domain


LOGGER = logging.getLogger(__name__)

@cache
def get_goes_lonlats(
        x_0: float,
        x_1: float,
        n_x: int,
        y_0: float,
        y_1: float,
        n_y: int,
        perspective_point_height: float,
        longitude_of_projection_origin: float,
        semi_major_axis: float,
        semi_minor_axis: float,
        sweep_angle_axis: str,
):
    """
    Calculate longitude and latitude coordinates for a GOES ABI dataset.

    Args:
        x_0: Start value of the x coordinates.
        x_1: End value of the x coordinates.
        n_x: The number of values along the x dimension.
        y_0: Start value of the y coordinates.
        y_1: End value of the y coordinates.
        n_y: The number of values along the y dimension.
        perspective_point_height: Perspective point height from projection attributes.
        longitude_of_projection_origin: Longitude of projection origin from projection attributes.
        semi_major_axis: The length of the semi-major axis from the projection attributes
        semi_minor_axis: The length of the semi-minor axis from the projection attributes
        sweep_angle_axis: The sweep-angle axis from the projection attributes.

    Returns:
        lon, lat: 2D arrays of longitude and latitude.
    """
    # Define GOES geostationary projection.
    crs = CRS.from_proj4(
        f"+proj=geos "
        f"+h={perspective_point_height} "
        f"+lon_0={longitude_of_projection_origin} "
        f"+sweep={sweep_angle_axis} "
        f"+a={semi_major_axis} "
        f"+b={semi_minor_axis}"
    )

    transformer = Transformer.from_crs(
        crs, CRS.from_epsg(4326), always_xy=True
    )

    x = np.linspace(x_0, x_1, n_x) * perspective_point_height
    y = np.linspace(y_0, y_1, n_y) * perspective_point_height

    xx, yy = np.meshgrid(x, y)

    lon, lat = transformer.transform(xx, yy)

    # Mask pixels outside the Earth disk.
    lon[~np.isfinite(lon)] = np.nan
    lat[~np.isfinite(lat)] = np.nan

    return lon, lat


_RESAMPLE_INFO = {}


def get_resample_info_cached(dataset: xr.Dataset, domain):
    """
    Cached calculation of resampling info.

    Args:
        dataset: An xarray.Dataset containing the GOES observations with latitude
            and longitude coordinates.
        domain: A pyresample area definition defining the target grid.

    Return:
        Nearest-neighbor resampling information that can be used with the
        pansat.utils.resample_data function.
    """
    global _RESAMPLE_INFO
    lons = dataset.longitude.data
    lon_min, lon_max, n_lons = np.nanmin(lons), np.nanmin(lons), lons.size
    lats = dataset.latitude.data
    lat_min, lat_max, n_lats = np.nanmin(lats), np.nanmax(lats), lats.size
    key = (lon_min, lon_max, n_lons, lat_min, lat_max, n_lats)

    if not key in _RESAMPLE_INFO:
        _RESAMPLE_INFO[key] = get_resample_info(
            dataset,
            domain,
            radius_of_influence=5e3
        )
    _RESAMPLE_INFO.setdefault(key, get_resample_info(dataset, domain))


class GOESABIObservations:

    def __init__(
            self,
            pansat_class,
            ):
        self.pansat_class = pansat_class


    def extract_observations(
            self,
            nominal_time,
            domain,
            time_window: np.timedelta64 = np.timedelta64(5, "m")
    ):
        time_range = TimeRange(
            nominal_time - np.timedelta64(180, "s"),
            nominal_time + np.timedelta64(180, "s")
        )

        all_obs = []
        for chan in range(1, 17):
            goes_l1b = self.pansat_class(channel=chan, region="F")
            try:
                file_record = goes_l1b.find_files(time_range)[0]
                with goes_l1b.open_remote(file_record) as data:

                    # Get projection properties.
                    proj = data["goes_imager_projection"].attrs
                    x_0 = data.x.data[0]
                    x_1 = data.x.data[-1]
                    n_x = data.x.size
                    y_0 = data.y.data[0]
                    y_1 = data.y.data[-1]
                    n_y = data.y.size
                    lons, lats = get_goes_lonlats(
                        x_0, x_1, n_x,
                        y_0, y_1, n_y,
                        perspective_point_height=proj["perspective_point_height"],
                        longitude_of_projection_origin=proj["longitude_of_projection_origin"],
                        semi_major_axis=proj["semi_major_axis"],
                        semi_minor_axis=proj["semi_minor_axis"],
                        sweep_angle_axis=proj["sweep_angle_axis"]
                    )

                    # Check if we can discard data.
                    lons_d, lats_d = domain.get_lonlats()
                    lon_min, lon_max = np.nanmin(lons_d), np.nanmax(lons_d)
                    lat_min, lat_max = np.nanmin(lons_d), np.nanmax(lats_d)
                    mask = (
                        (lon_min <= lons) * (lons <= lon_max) *
                        (lat_min <= lats) * (lats <= lat_max)
                    )
                    inds = np.where(mask)
                    row_start = inds[0].min()
                    row_end = inds[0].max()
                    row_slice = slice(row_start, row_end)
                    col_start = inds[1].min()
                    col_end = inds[1].max()
                    col_slice = slice(col_start, col_end)

                    data = data[["Rad"]][{"x": col_slice, "y": row_slice}]
                    data["longitude"] = (("y", "x"), lons[row_slice, col_slice])
                    data["latitude"] = (("y", "x"), lats[row_slice, col_slice])

                    resample_info = get_resample_info_cached(data, domain)
                    data_r = resample_data(
                        data[["Rad", "longitude", "latitude"]].compute(),
                        domain,
                        info=resample_info
                    )
                    data_r = data_r.rename(Rad="obs")
                    all_obs.append(data_r)
            except IndexError:
                LOGGER.warning(
                    "No GOES ABI file found for %s.",
                    nominal_time
                )
        obs = xr.concat(all_obs, dim="channel")
        return obs


SATELLITES = {
    "goes_16": GOESABIObservations(GOES16L1BRadiances),
    "goes_18": GOESABIObservations(GOES18L1BRadiances),
    "goes_19": GOESABIObservations(GOES19L1BRadiances),
}


def extract_goes_abi_observations(
        goes_satellite: GOESABIObservations,
        domain: pyresample.goemetry.AreaDef,
        start_time: np.datetime64,
        end_time: np.datetime64,
        time_step: np.timedelta64,
        output_file: Path,
        n_workers: int = 4
):
    """
    Extract GOES ABI observations for a given time period and store as .zarr file.

    Args:
        goes_satellite: A GOESABIObservations object defining the GOES satellite
            from which to extract the observations.
        domain: The domain to which to resample the observations.
        start_time: Start time of the interval over which to extract observations.
        end_time: End time of the interval over which to extract the observations.
        time_step: The time interval at which to extract observations.
        output_file: A Path object pointing to the file in which to store the extracted
            observations.
        n_workers: The number of workers to use for the extraction.
    """
    output_file = Path(output_file)
    from satrain_now.domains import get_domain
    times = np.arange(start_time, end_time + 0.5 * time_step, time_step)

    compressor = Blosc(cname="zstd", clevel=4)
    encoding = {
        "obs": {
            "chunks": (1, 16, 256, 256),
            "dtype": "float32",
            "compressors": compressor
        },
    }

    if 1 < n_workers:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            tasks = [
                (time, pool.submit(goes_satellite.extract_observations, time, domain))
                for time in times
            ]
            for time, task in tqdm(tasks, "Extracting GOES observations"):
                try:
                    res = task.result()
                    res = res.expand_dims("time").assign_coords(time=np.array([time]))
                    if output_file.exists():
                        res.to_zarr(
                            output_file,
                            mode="a",
                            append_dim="time",
                        )
                    else:
                        res.to_zarr(
                            output_file,
                            mode="a",
                            encoding=encoding,
                        )

                except Exception:
                    LOGGER.exception(
                        "Encoutered an error when extracting observations for %s.",
                        time
                    )
    else:
        for time in times:
            try:
                res = goes_satellite.extract_observations(time, domain)
                res = res.expand_dims("time").assign_coords(time=np.array([time]))
                if output_file.exists():
                    res.to_zarr(
                        output_file,
                        mode="a",
                        append_dim="time"
                    )
                else:
                    res.to_zarr(
                        output_file,
                        mode="a",
                        encoding=encoding,
                    )
            except Exception:
                LOGGER.exception(
                    "Encoutered an error when extracting observations for %s.",
                    time
                )


@click.command()
@click.argument("satellite", type=str)
@click.argument("domain", type=str)
@click.argument("start_time", type=str)
@click.argument("end_time", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--n_workers",
    type=int,
    default=1,
    help="The number of processes to use for the data extraction."
)
def cli(
    satellite: str,
    domain: str,
    start_time: str,
    end_time: str,
    output_file: str,
    n_workers: int = 1
):
    """Extract GOES satellite observation data.

    SATELLITE: Satellite to extract data from (goes16, goes18, goes19)
    DOMAIN: The name of the domain for which to extract observations.
    START_TIME: The start time of the interval for which to extract observations.
    END_TIME: The end time of the interval for which to extract observations.
    """
    goes_satellite = SATELLITES.get(satellite, None)
    if goes_satellite is None:
        LOGGER.error(
            "Satellite %s is not available. Available satellites are %s.",
            satellite,
            list(SATELLITES.keys())
        )
        return 1

    domain = get_domain(domain)
    start_time = np.datetime64(start_time, "h").astype("datetime64[s]")
    end_time = np.datetime64(end_time, "h").astype("datetime64[s]")
    output_file = Path(output_file)
    extract_goes_abi_observations(
        goes_satellite,
        domain,
        start_time,
        end_time,
        time_step=np.timedelta64(600, "s"),
        output_file=output_file,
        n_workers=n_workers
    )
