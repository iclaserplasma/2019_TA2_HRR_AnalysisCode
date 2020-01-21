# Probe to Density
#### christopher.underwood@york.ac.uk

The following code is written to convert interferograms into a density profile of the plasma channel.

The code is run from 
>densityExtraction.py 

where the input is the image file. 

The code can work without a reference image, as long as there is a region of fringes equal to the size of the plasma channel in the image.
Otherwise a reference image of the plasma channel region will be required.

This region is selected in:
> cropToPlasmaChannel( ... , centreCoorsOfFringeRegion = [xc, yc])

Where xc and yc are the center coors.

## Inputs

1. Plasma channel coors
2. Fourier space coors, although this has also been automated. Do not yet know how robust the automation is.
3. The spatial calibration of the probe, mPerPix =


There are other inputs, padding and cropping parameters, but hopefully these will be constant.


## Strange code things.
>phase_correction.py

The class definition is different depending on whether running by itself, or as part of densityExtraction.

