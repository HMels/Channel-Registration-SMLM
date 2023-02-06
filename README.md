Channel-Registration-SMLM

This program implements 2D Catmul-Rom splines together with the Affine transform to properly align (Register) two channels in super resolution microscopy.

## Requirements
- Python 3.x 
- Spyder 5.0.0 (but earlier versions will work as well) 
- Photonpy 1.0.39
- Picasso 0.3.1 (not necessarily needed)
- TensorFlow 2.4.1 (and earlier versions) 
- Numpy 1.19.2 (tensorflow only works up till this version)
- Gast 0.3.3

## Installation
1. Install the necessary packages using ```pip install <package>```
2. Run the program using Python on Spyder

## Usage
This program is used for the Quantative Nanoscopy group of Delft Center Systems and Control of Delft University of Technology.

## Author
Mels Habold, for his Master Thesis called Catmul-Rom spline Channel Registration in Single Molecule Localization Microscopy

## Summary

Single Molecule Localization Microscopy (SMLM) is a technique that uses techniques on fluorescence microscopy in order to bypass the Abbe diffraction limit and reach smaller levels of precision. It does this by fitting the Point Spread Functions (PSFs) of the lens to the pixel data to estimate the true emitter positions. In SMLM, different colour fluorophores can be used to flag different structures in the sample, which means two colour-channels are created. On this scale optical systems start to have effects called aberrations that mutate the PSF, and are dependent on the position and colour of the emitter. Both channels need to be aligned, and the aberrations need to be corrected for.

This thesis proposes a method called Catmull-Rom splines Channel Registration (CRsCR) to create a registration map that aligns one channel to the other. This method is based on a similar method described by Niekamp[1] which we call piece-wise affine Chanel Registration (PACR). First, CRsCR links localizations between channels to form localization pairs. It then optimizes the distances between these localization pairs via an affine Linear Least Squares transform to correct for global aberrations. Lastly, local aberrations are corrected for via Catmull-Rom splines interpolation.

CRsCR reaches similar precision as PACR, as well as a similar distribution of errors over the field of view. The most important limitation of this method lies within the linking of localizations. This is only possible after channels have been sufficiently aligned such that localization pairs lie in the vicinity of each other. Therefore, an improvement to CRsCR that allows it to be applied to data that does not allow clean linking of localization pairs has been discussed.

## Acknowledgements

This project was created for the Quantitative Nanoscopy group of Delft Center Systems and Control of Delft University of Technology.
