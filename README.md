# satrain_now

Benchmark dataset for satellite precipitation nowcasting and analysis.


## Installation

``` shell
conda env create --file environment.yml
conda activate satrain_now
```

## Usage

### Extract GOES Imagery

``` shell
satrain_now extract_goes_obs goes_16 conus_small 2023-06-01 2023-06-02 goes.zarr
```
