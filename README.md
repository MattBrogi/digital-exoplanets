# digital-exoplanets
Hands-on session on High Resolution Spectroscopy at Digital Exoplanets 2019, Prague

This tutorial will guide through common analysis techniques to extract the signal of exoplanet atmospheres from high-resolution spectra. It is focussed on Infrared spectroscopy and cross-correlation functions, hence it is substantially different from the single-line detections (especially Na) obtained with optical spectroscopy. The main differences regard the more severe contamination from the Earth's atmosphere (telluric spectrum) and the overall decreased stability of NIR spectrographs compared to e.g. HARPS in the optical.

## Outline

There will be three main tasks that we will perform:

1. Model a sequence of spectra including the planet signal, a stellar spectrum, and telluric absorption lines, with time-variable and wavelength-dependent noise to reproduce realistic observing conditions

2. Reverse-engineer the data by applying the standard data analysis techniques and reduce the spectral sequence to the pre-cross correlation stage (noise + planet signal)

3. Apply the cross correlation with a range of models and investigate how the cross correlation signal varies as function of model.

## Requirements

- Most of the code will run with standard 'numpy' and 'scipy' libraries (plus 'matplotlib' for visualisation). The MCMC part will run with the 'emcee' package.

- The code is written for Python 3.x, but basic Python users should be able to tweak it to work in Python 2.x.
