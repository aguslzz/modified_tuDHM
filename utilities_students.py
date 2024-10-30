# -*- coding: utf-8 -*-
"""
Title-->            Utilities script
Authors-->          Raul Castaneda,
Date-->             10/08/2023
Universities-->     EAFIT University (Applied Optics Group)
                    UMASS (Optical Imaging Research Laboratory)
"""

# import lybraries
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize


# Function to read an image file from the disk
def imageRead(namefile):
    # inputs:
    # namefile - direction image to read
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)

    return loadImage


# Function to display an Image
def imageShow(image, title):
    # inputs:
    # image - The image to show
    # title - Title of the displayed image
    plt.imshow(image, cmap='gray'), plt.title(title)
    plt.show()

    return


# Function to compute the amplitude of a given complex field
def amplitude(complexField, log):
    # inputs:
    # complexField - The input complex field to compute the amplitude
    # log - boolean variable to determine if a log representation is applied
    out = np.abs(complexField)

    if log == True:
        out = 20 * np.log(out)

    return out


# Function to compute the phase of a given complex field
def phase(complexField):
    # inputs:
    # complexField - The input complex field to compute the phase
    out = np.angle(complexField)

    return out


def intensity(complexField, log):
    # inputs:
    # complexField - The input complex field to compute the intensity
    # log - boolean variable to determine if a log representation is applied
    out = np.abs(complexField)
    out = out * out

    if log == True:
        out = 20 * np.log(out)
        out[out == np.inf] = 0
        out[out == -np.inf] = 0

    return out


# Function to compute the Fourier Transform
def ft(field):
    # inputs:
    # field - The input to compute the Fourier Transform
    ft = np.fft.fft2(field)
    ft = np.fft.fftshift(ft)

    return ft


# Function to compute the Fourier Transform
def ift(field):
    # inputs:
    # field - The input to compute the Fourier Transform
    ift = np.fft.ifft2(field)
    ift = np.fft.fftshift(ift)
    ift = np.fft.fftshift(ift)

    return ift


# Spatial filtering process for FCF implementation
def spatialFilteringCF(inp, width, height):
    # inputs:
    # inp - field
    # width -
    # height -
    field_spec = np.fft.fft2(inp)
    field_spec = np.fft.fftshift(field_spec)

    # Finding the max peaks for +1 order in I or II quadrant
    mask = np.zeros((width, height))
    mask[0:height, 0:round(int(width/2))] = 1
    field_spec_tem = field_spec * mask
    maximum = np.amax(field_spec_tem)
    fy_max, fx_max = np.where(field_spec_tem == maximum)

    # Determination of the ROI size. To do this, we use the theoretical size of the +1 or -1 diffraction orders according...
    # ... to the distance of their centers to the DC coordinates (middle point in the DHM hologram spectrum).
    d = np.sqrt(np.power(fy_max - width / 2, 2) + np.power(fx_max - height / 2, 2))
    radius = d / 3
    mask = circularMask(width, height, radius, fy_max, fx_max, False)

    # Filtering the hologram
    tmp = field_spec * mask

    # Coming back to spatial domain (retrieving filtered hologram)
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)

    return out, fx_max, fy_max


# Function to create a circular mask
def circularMask(width, height, radius, centX, centY, visualize):
    # inputs:
    # width - size image Y
    # height - size image X
    # radius - circumference radius
    # centX - coordinate Y center
    # centY - coordinate X center
    X, Y = np.ogrid[:width, :height]
    mask = np.zeros((width, height))
    circle = np.sqrt((X - centX) ** 2 + (Y - centY) ** 2) <= radius
    mask[circle] = 1

    if visualize:
        imageShow(mask, 'mask')

    return mask


# function to retrieve the complex object information from an off-axis hologram
def reconstruction(field, wavelength, dxy, distance):
    # inputs:
    # field: hologram to reconstruct
    # wavelength: wavelength used to register the hologram
    # dxy: pixel pitch of camera used to register the hologram
    # distance: propagation distance
    field = np.array(field)
    width, height = field.shape

    # Creating a mesh_grid to operate in world coordinates
    Y, X = np.meshgrid(np.arange(-height / 2, height / 2), np.arange(-width / 2, width / 2), indexing='ij')
    fx_0 = width / 2
    fy_0 = height / 2
    k = (2 * math.pi) / wavelength

    # The spatial filtering process is executed
    print("Spatial filtering process started.....")
    holo_filter, fx_max, fy_max = spatialFilteringCF(field, width, height)
    print("Spatial filtering process finished.")

    # Please insert your code here
    '''
    .
    .
    . 
    . 
    . 
    . 
    . 
    . 
    .     
    '''

    # Best phase compensation
    print("Phase compensation started....")
    theta_x = math.asin((fx_0 - best_fx_max) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - best_fy_max) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    comp_phase = holo_filter * ref_wave


    # propagation of the complex object filed by implemented the AngularSpectrum
    comp_phase = angularSpectrum(comp_phase, width, height, wavelength, distance, dxy)

    return comp_phase


# Function to propagate an optical field using the Angular Spectrum approach
def angularSpectrum(field, width, height, wavelength, distance, dxy):
    # Inputs:
    # field - complex field
    # width - size image Y
    # height - size image X
    # wavelength - wavelength
    # z - propagation distance
    # dxy - sampling pitches
    X, Y = np.meshgrid(np.arange(-height / 2, height/ 2), np.arange(-width / 2, width / 2), indexing='ij')

    dfx = 1 / (dxy * height)
    dfy = 1 / (dxy * width)

    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)

    kernel = np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))
    phase = np.exp(1j * distance * 2 * math.pi * np.sqrt(kernel))

    tmp = field_spec * phase
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)

    return out
