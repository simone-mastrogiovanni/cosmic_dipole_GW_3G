# Detection and estimation of the cosmic dipole with the Einstein Telescope and Cosmic Explorer

This repository hosts some of the tools used to study the detectability of the cosmic dipole using GW number counts. If you use the following code, please cite

```
Inser here bibtex for citation
```

## Installing

In order to use the codes in this repository, you need to

* Clone this repository with `git clone https://github.com/simone-mastrogiovanni/cosmic_dipole_GW_3G.git`
* Install the python environment [igwn-py38](https://computing.docs.ligo.org/conda/environments/igwn-py38/). We strongly encourage the use of Conda.
* import all the functions in the module `gwdip.py`. Some of the functions make use of [icarogw](https://git.ligo.org/cbc-cosmo/icarogw/-/tree/main/icarogw) which is a python package for BBHs population analyses. This code is provided in the dir `source_code`, and it is automatically loaded by `gwdip.py`. So, you don't need extra installation.

## Examples

We provide two python notebooks to show some of the code functionalities.

* `horizons_for_detection.ipynb`: Is the python notebook used to generate the figure of merit for BBH and BNS detectiosn with ET+2CE.
* `BBH_example_dipole`: The notebook provides an example on how to simulate GW detections with cosmic dipole, calculate the statistical estimator for detection, plot its sky maps and run Monte Carlo Markov Chains to estimate the dipole direction and amplitude.

