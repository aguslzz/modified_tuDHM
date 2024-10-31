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
from scipy.optimize import minimize, minimize_scalar
import time


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

def vortexConvolution(inp, l_vortex, plot = False):

    width, height = (np.array(inp)).shape

    # Generate a grid of x and y coordinates
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]

    # Compute the angle from the origin in radians, and wrap it to the range [0, 2π]
    angle = np.arctan2(y, x)   # Angle in radians
    vortex = np.exp(1j * l_vortex * angle)
    
    # Plot the vortex
    
    if plot:
        plt.imshow(np.angle(vortex), cmap='gray')
        plt.colorbar(label="Angle (radians)")
        plt.title("Angle from Origin (0 to 2π)")
        plt.show()
    
    conv = (inp * vortex)
    ift_conv = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(conv)))

    #if plot: 
        #plt.imshow(intensity(ift_conv, True), cmap='grey', extent=(-size//2, size//2, -size//2, size//2))
        #plt.title("Hologram and vortex convolution in spectra")

    return ift_conv


# Spatial filtering process for FCF implementation
def spatialFilteringCF_vortex_v2(inp, width, height, l_vortex):
    # inputs:
    # inp - field
    # width -
    # height -
    field_spec_vortex = vortexConvolution(inp, l_vortex)

    # Finding the max peaks for +1 order in I or II quadrant
    mask = np.zeros((width, height))
    mask[0:height, 0:round(int(width/2))] = 1
    field_spec_tem = field_spec_vortex * mask
    maximum = np.amax(field_spec_tem)
    fy_max, fx_max = np.where(field_spec_tem == maximum)

    # Determination of the ROI size. To do this, we use the theoretical size of the +1 or -1 diffraction orders according...
    # ... to the distance of their centers to the DC coordinates (middle point in the DHM hologram spectrum).
    d = np.sqrt(np.power(fy_max - width / 2, 2) + np.power(fx_max - height / 2, 2))
    radius = d / 3
    mask = circularMask(width, height, radius, fy_max, fx_max, False)

    # Filtering the hologram
    tmp = field_spec_vortex * mask

    # Coming back to spatial domain (retrieving filtered hologram)
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)

    minimum = np.amin(field_spec_tem)
    fy_min, fx_min = np.where(field_spec_tem == minimum)

    return out, fx_max, fy_max, fx_min, fy_min


def search_initial_conditions(inp, lvortex):
    height, width = (np.array(inp)).shape
    field_spec_vortex = vortexConvolution(inp, lvortex)
    
    # Mask the -1 diffraction term
    mask = np.zeros((height, width))
    mask[0:height, 0:round(int(width/2))] = 1
    field_spec_tem = field_spec_vortex * mask
    
    # Notice we assume that the minimun value pixel will be a better search starting 
    # point than the maximum value pixel without the vortex convolution. 
    minimum = np.amin(field_spec_tem)
    fy_min, fx_min = np.where(field_spec_tem == minimum)
    
    return fx_min, fy_min

    
# Spatial filtering process for FCF implementation
def spatialFilteringCF_vortex(inp, width, height, l_vortex):
    # inputs:
    # inp - field
    # width -
    # height -
    field_spec_vortex = vortexConvolution(inp, l_vortex)

    # Finding the max peaks for +1 order in I or II quadrant
    mask = np.zeros((width, height))
    mask[0:height, 0:round(int(width/2))] = 1
    field_spec_tem = field_spec_vortex * mask
    maximum = np.amax(field_spec_tem)
    fy_max, fx_max = np.where(field_spec_tem == maximum)

    # Determination of the ROI size. To do this, we use the theoretical size of the +1 or -1 diffraction orders according...
    # ... to the distance of their centers to the DC coordinates (middle point in the DHM hologram spectrum).
    d = np.sqrt(np.power(fy_max - width / 2, 2) + np.power(fx_max - height / 2, 2))
    radius = d / 3
    mask = circularMask(width, height, radius, fy_max, fx_max, False)

    # Filtering the hologram
    tmp = field_spec_vortex * mask
    
    # Coming back to spatial domain (retrieving filtered hologram)
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)

    return out, fx_max, fy_max


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


# cost function for the CFS implementation
def costFunction(seeds, width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k):
    # inputs:
    # seeds - seed for compute the minimization
    # width - size image Y
    # height - size image X
    # holo_filter - complex object filter hologram
    # dxy - pixel size
    # X - coordinate X meshgrid
    # Y - coordinate Y meshgrid
    # fx_0 - coordinate x DC diffraction
    # fy_0 - coordinate y DC diffraction
    # k - vector number
    J = 0
    theta_x = math.asin((fx_0 - seeds[0]) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - seeds[1]) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    phase = np.angle(holo_filter * ref_wave)
    phase = phase + math.pi

    phase = (phase > 0.2)
    sumIB = phase.sum()
    J = (width * height) - sumIB
    #J = np.std(phase)

    return J

def binarized_hologram_std_cf(seeds, width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k):
    # inputs:
    # seeds - seed for compute the minimization
    # width - size image Y
    # height - size image X
    # holo_filter - complex object filter hologram
    # dxy - pixel size
    # X - coordinate X meshgrid
    # Y - coordinate Y meshgrid
    # fx_0 - coordinate x DC diffraction
    # fy_0 - coordinate y DC diffraction
    # k - vector number
    J = 0
    theta_x = math.asin((fx_0 - seeds[0]) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - seeds[1]) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    phase = np.angle(holo_filter * ref_wave)
    phase = phase + math.pi
    # phase = (phase > 0.2)
    # sumIB = phase.sum()
    # J = (width * height) - sumIB
    J = np.std(phase)

    return J


def binarized_hologram_maxsum_cf(seeds, width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k):
    # inputs:
    # seeds - seed for compute the minimization
    # width - size image Y
    # height - size image X
    # holo_filter - complex object filter hologram
    # dxy - pixel size
    # X - coordinate X meshgrid
    # Y - coordinate Y meshgrid
    # fx_0 - coordinate x DC diffraction
    # fy_0 - coordinate y DC diffraction
    # k - vector number
    J = 0
    theta_x = math.asin((fx_0 - seeds[0]) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - seeds[1]) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    phase = np.angle(holo_filter * ref_wave)
    phase = phase + math.pi
    phase = (phase > 0.2)
    sumIB = phase.sum()
    J = (width * height) - sumIB
    
    return J



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

    # loading seeds
    seeds = (fx_max[0], fy_max[0])
    step = 1
    bounds = [(fx_max - step, fx_max + step), (fy_max - step, fy_max + step)]

    # minimization
    print("Minimization process started.....")
    start_time = time.time()
    res = minimize(costFunction, seeds, args=(width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k),
                   method='Nelder-Mead', bounds=bounds, tol=1e-9)
    end_time = time.time()
    print(f"Minimization process finished. Cost function value = {res.fun}")
    execution_time = end_time - start_time
    print(f"Minization process time = {execution_time}")
    
    best_fx_max = res.x[0]
    best_fy_max = res.x[1]
    print('fx: ', best_fx_max)
    print('fy: ', best_fy_max)


    # Best phase compensation
    print("Phase compensation started....")
    theta_x = math.asin((fx_0 - best_fx_max) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - best_fy_max) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    comp_phase = holo_filter * ref_wave


    # propagation of the complex object filed by implemented the AngularSpectrum
    comp_phase = angularSpectrum(comp_phase, width, height, wavelength, distance, dxy)

    return comp_phase, res

# function to retrieve the complex object information from an off-axis hologram
def reconstruction_vortex(field, wavelength, dxy, distance, l_vortex):
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
    holo_filter, fx_max, fy_max = spatialFilteringCF_vortex(field, width, height, l_vortex)
    print("Spatial filtering process finished.")

    # loading seeds
    seeds = (fx_max[0], fy_max[0])
    step = 1
    bounds = [(fx_max - step, fx_max + step), (fy_max - step, fy_max + step)]

    # minimization
    print("Minimization process started.....")
    start_time = time.time()
    res = minimize(costFunction, seeds, args=(width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k),
                   method='Nelder-Mead', bounds=bounds, tol=1e-9)
    end_time = time.time()
    print(f"Minimization process finished. Cost function value = {res.fun}")
    execution_time = end_time - start_time
    print(f"Minization process time = {execution_time}")
    best_fx_max = res.x[0]
    best_fy_max = res.x[1]
    print('fx: ', best_fx_max)
    print('fy: ', best_fy_max)


    # Best phase compensation
    print("Phase compensation started....")
    theta_x = math.asin((fx_0 - best_fx_max) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - best_fy_max) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    comp_phase = holo_filter * ref_wave


    # propagation of the complex object filed by implemented the AngularSpectrum
    comp_phase = angularSpectrum(comp_phase, width, height, wavelength, distance, dxy)

    return comp_phase, res


# function to retrieve the complex object information from an off-axis hologram
def reconstruction_vortex_v2(field, wavelength, dxy, distance, l_vortex):
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
    b, fx_max, fy_max, fx_min, fy_min = spatialFilteringCF_vortex_v2(field, width, height, l_vortex)
    print("Spatial filtering process finished.")

    holo_filter, aa, aaa = spatialFilteringCF(field, width, height)

    # loading seeds
    seeds = (fx_min[0], fy_min[0])
    step = 1
    bounds = [(fx_min - step, fx_min + step), (fy_min - step, fy_min + step)]

    # minimization
    print("Minimization process started.....")
    start_time = time.time()
    res = minimize(costFunction, seeds, args=(width, height, holo_filter, wavelength, dxy, X, Y, fx_0, fy_0, k),
                   method='Nelder-Mead', bounds=bounds, tol=1e-9)
    end_time = time.time()
    print(f"Minimization process finished. Cost function value = {res.fun}")
    execution_time = end_time - start_time
    print(f"Minization process time = {execution_time}")
    
    best_fx_max = res.x[0]
    best_fy_max = res.x[1]
    print('fx: ', best_fx_max)
    print('fy: ', best_fy_max)


    # Best phase compensation
    print("Phase compensation started....")
    theta_x = math.asin((fx_0 - best_fx_max) * wavelength / (width * dxy))
    theta_y = math.asin((fy_0 - best_fy_max) * wavelength / (height * dxy))
    ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dxy) + (math.sin(theta_y) * Y * dxy)))
    comp_phase = holo_filter * ref_wave


    # propagation of the complex object filed by implemented the AngularSpectrum
    comp_phase = angularSpectrum(comp_phase, width, height, wavelength, distance, dxy)

    return comp_phase, res




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


