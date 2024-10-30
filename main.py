import utilities as ut


# Lines to read the hologram
hologram = ut.imageRead('C:/Users/racastaneq/PycharmProjects/DHM_Reconstruction methods/Samples/holo-RBC-20p205-2-3.png')
ut.imageShow(hologram, 'Sample')

# Parameters of reconstruction (everything must be same units)
wavelength = 0.532
dxy = 2.4
distance = 0

# Compute the FT of the hologram
ft_holo = ut.ft(hologram)
ut.imageShow(ut.intensity(ft_holo, True), 'FT hologram')

# numerical reconstruction
complexObject = ut.reconstruction(hologram, wavelength, dxy, 500)
amplitude = ut.amplitude(complexObject, False)
phase = ut.phase(complexObject)
ut.imageShow(amplitude, 'Amplitude')
ut.imageShow(phase, 'Phase')

