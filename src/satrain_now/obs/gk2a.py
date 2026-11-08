"""
satrain_now.obs.gk2a
====================

Functionality to extract GK2A observations.
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
from pyresample.geometry import AreaDefinition
from pansat.products.satellite.gk2a import (
    GK2AProduct,
    AMI_CHANNELS
)

from satrain_now.domains import get_domain


LOGGER = logging.getLogger(__name__)


@cache
def get_ami_lonlats(
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
):
    """
    Calculate longitude and latitude coordinates for GK2A AMI observations.

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
    # Define GK2A geostationary projection.
    crs = CRS.from_proj4(
        f"+proj=geos "
        f"+h={perspective_point_height} "
        f"+lon_0={longitude_of_projection_origin} "
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

    return lon.copy(), lat.copy()


_RESAMPLE_INFO = {}


def calibrate_obs(
        dataset: xr.Dataset
):
    """
    Convert data from raw pixel values to reflectivities or brightness
    temperatures.

    Args:
        dataset: The loaded AMI data.

    Return:
        A data array containing the calibrated observations.
    """
    attrs = dataset.attrs
    data = dataset["image_pixel_values"]
    qual = data & 0b1100000000000000

    data &= 0b0011111111111111
    data = data.where(qual == 0)

    gain = attrs["DN_to_Radiance_Gain"]
    offset = attrs["DN_to_Radiance_Offset"]
    #data = np.clip(data, 0.0, None)
    data = gain * data + offset

    if "Radiance_to_Albedo_c" in attrs:
        rad_to_alb = attrs["Radiance_to_Albedo_c"]
        rad_to_alb *= 100
        data = data * rad_to_alb
    else:
        c0 = attrs["Teff_to_Tbb_c0"]
        c1 = attrs["Teff_to_Tbb_c1"]
        c2 = attrs["Teff_to_Tbb_c2"]

        cval = attrs["light_speed"]
        kval = attrs["Boltzmann_constant_k"]
        hval = attrs["Plank_constant_h"]

        wave_num = 1e6 / float(attrs["channel_center_wavelength"])
        e1 = (2 * hval * cval * cval) * np.power(wave_num, 3)
        e2 = (data.data * 1e-5)
        t_eff = ((hval * cval / kval) * wave_num) / np.log((e1 / e2) + 1)

        # Now convert to actual brightness temperature
        bt_data = c0 + c1 * t_eff + c2 * t_eff * t_eff
        data.data = bt_data
    return data


def get_resample_info_cached(dataset: xr.Dataset, domain):
    """
    Cached calculation of resampling info.

    Args:
        dataset: An xarray.Dataset containing the GK2A observations with latitude
            and longitude coordinates.
        domain: A pyresample area definition defining the target grid.

    Return:
        Nearest-neighbor resampling information that can be used with the
        pansat.utils.resample_data function.
    """
    global _RESAMPLE_INFO
    lons = dataset.longitude.data
    lon_min, lon_max, n_lons = np.nanmin(lons), np.nanmax(lons), lons.size
    lats = dataset.latitude.data
    lat_min, lat_max, n_lats = np.nanmin(lats), np.nanmax(lats), lats.size
    key = (lon_min, lon_max, n_lons, lat_min, lat_max, n_lats)

    if not key in _RESAMPLE_INFO:
        _RESAMPLE_INFO[key] = get_resample_info(
            dataset,
            domain,
            radius_of_influence=5e3
        )
    return _RESAMPLE_INFO[key]


class GK2AAMIObservations:
    """
    An interface for extracting AMI L1B calibrated reflectances and brightness
    temperature.
    """

    def __init__(
            self,
            pansat_class,
            ):
        """
        Args:
            pansat_class: The pansat product class for the G2KA AMI L1b
                product.
        """
        self.pansat_class = pansat_class


    def extract_observations(
            self,
            nominal_time,
            domain,
            time_window: np.timedelta64 = np.timedelta64(5, "m")
    ):
        """
        Extract goes observations closes to the given nominal time and map
        observations to the given domain.

        Args:
            nominal_time: The nominal time for which to extract the observations.

        """
        time_range = TimeRange(
           nominal_time - np.timedelta64(180, "s"),
           nominal_time + np.timedelta64(180, "s")
        )

        all_obs = []
        for chan_ind, chan in enumerate(AMI_CHANNELS):
            ami_l1b = self.pansat_class(region="fd", channel=chan)
            try:
                print(time_range, ami_l1b)
                file_records = ami_l1b.find_files(time_range)
                date = nominal_time.astype("datetime64[s]")
                time_diffs = [
                    abs((date - rec.central_time.start).total_seconds())
                    for rec in file_records
                ]
                file_record = file_records[np.argmin(time_diffs)]

                with ami_l1b.open_remote(file_record) as data:

                    x_0 = data.attrs["image_upperleft_x"]
                    x_1 = data.attrs["image_lowerright_x"]
                    y_0 = data.attrs["image_upperleft_y"]
                    y_1 = data.attrs["image_lowerright_y"]

                    # Get projection properties.
                    n_x = data.dim_image_x.size
                    n_y = data.dim_image_y.size

                    major_axis = float(data.attrs["earth_equatorial_radius"])
                    minor_axis = float(data.attrs["earth_polar_radius"])
                    height = float(data.attrs["nominal_satellite_height"]) - major_axis
                    lon_0 = data.attrs["sub_longitude"] * 180 / np.pi

                    lons, lats = get_ami_lonlats(
                        x_0, x_1, n_x,
                        y_0, y_1, n_y,
                        perspective_point_height=height,
                        longitude_of_projection_origin=lon_0,
                        semi_major_axis=major_axis,
                        semi_minor_axis=minor_axis,
                    )

                    # Check if we can discard data.
                    lons_d, lats_d = domain.get_lonlats()
                    lon_min, lon_max = np.nanmin(lons_d), np.nanmax(lons_d)
                    lat_min, lat_max = np.nanmin(lats_d), np.nanmax(lats_d)
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

                    data = data[["image_pixel_values"]][{
                        "dim_image_x": col_slice,
                        "dim_image_y": row_slice
                    }].compute()
                    obs = calibrate_obs(data)
                    data = xr.Dataset({
                        "longitude": (("y", "x"), lons[row_slice, col_slice]),
                        "latitude": (("y", "x"), lats[row_slice, col_slice]),
                        "obs": (("y", "x"), obs.data)
                    })
                    resample_info = get_resample_info_cached(data, domain)
                    data_r = resample_data(
                        data[["obs", "longitude", "latitude"]],
                        domain,
                        info=resample_info
                    )
                    all_obs.append(data_r)
            except IndexError:
                LOGGER.warning(
                    "No GK2A AMI file found for %s.",
                    nominal_time
                )
        obs = xr.concat(all_obs, dim="channel")
        return obs


gk2a = GK2AAMIObservations(GK2AProduct)


def extract_gk2a_ami_observations(
        domain: AreaDefinition,
        start_time: np.datetime64,
        end_time: np.datetime64,
        time_step: np.timedelta64,
        output_file: Path,
        n_workers: int = 4
):
    """
    Extract GK2A AMI observations for a given time period and store as .zarr file.

    Args:
        domain: The domain to which to resample the observations.
        start_time: Start time of the interval over which to extract observations.
        end_time: End time of the interval over which to extract the observations.
        time_step: The time interval at which to extract observations.
        output_file: A Path object pointing to the file in which to store the extracted
            observations.
        n_workers: The number of workers to use for the extraction.
    """
    output_file = Path(output_file)
    times = np.arange(start_time, end_time + 0.5 * time_step, time_step)

    compressor = Blosc(cname="zstd", clevel=4)
    encoding = {
        "obs": {
            "chunks": (1, 16, 256, 256),
            "dtype": "float32",
            "compressors": compressor
        },
        "time": {
            "dtype": "float32"
        }
    }

    if 1 < n_workers:
        pool = ProcessPoolExecutor(max_workers=n_workers)
        try:
            tasks = [
                (time, pool.submit(gk2a.extract_observations, time, domain))
                for time in times
            ]
            for time, task in tqdm(tasks, "Extracting GK2A AMI observations"):
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
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted, terminating worker processes.")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False)
    else:
        for time in tqdm(times, "Extracting GK2A AMI observations"):
            try:
                res = gk2a.extract_observations(time, domain)
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
    domain: str,
    start_time: str,
    end_time: str,
    output_file: str,
    n_workers: int = 1
):
    """Extract GOES satellite observation data.

    DOMAIN: The name of the domain for which to extract observations.
    START_TIME: The start time of the interval for which to extract observations.
    END_TIME: The end time of the interval for which to extract observations.
    """
    domain = get_domain(domain)
    start_time = np.datetime64(start_time, "h").astype("datetime64[s]")
    end_time = np.datetime64(end_time, "h").astype("datetime64[s]")
    output_file = Path(output_file)
    extract_gk2a_ami_observations(
        domain,
        start_time,
        end_time,
        time_step=np.timedelta64(600, "s"),
        output_file=output_file,
        n_workers=n_workers
    )
