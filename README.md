# WRF-to-PyART

Convert regular WRF output simulations (specifically, reflectivity) to Pyart grid object for further radar analysis.

The main purpose of this conversion is to apply the [TINT](https://github.com/openradar/TINT) tracking algorithm to WRF simulated reflectivity. So far, this conversion script only converts reflectivity field in the WRF output.

Last update - 20210530 - Hungjui Yu

## Dependencies

* [Py-ART](https://arm-doe.github.io/pyart/index.html)
* [wrf-python](https://wrf-python.readthedocs.io/en/latest/index.html)
* [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)
* Numpy
* xarray
* math
* matplotlib

## How to run?
