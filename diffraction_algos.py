# ================================================================
# Copyright 2021, Vrije Universiteit Brussel. All rights reserved.
# Author: David Blinder
# Description: point-cloud CGH algorithms + conversion to tensors.
# ================================================================

import math
import numpy as np
import tensorflow as tf

# Function to check if integer is power of 2
def isPowerOfTwo (x):
    return (x and (not(x & (x - 1))) )

# Hologram settings object
class Hologram_Settings:   
    def __init__(self, res, pp, wlen, B, F):
        self.res = res      # hologram resolution (in pixels)
        self.pp = pp        # pixel pitch (in m)
        self.wlen = wlen    # wavelength (in m)
        self.B = B          # block size
        self.F = F          # scaling factor (multiple of block size: effective FFT block size = B*F)
        assert isPowerOfTwo(F)

    @classmethod
    def default_settings(cls):
        return cls((2048, 2048), 4e-6, 633e-9, 32, 2)

    def __compute_4d_dims(self, ffactor):
        blockdim = (self.res[0]/self.B, self.res[1]/self.B) # block dimensions
        assert blockdim[0].is_integer() and blockdim[1].is_integer()
        SS = self.B * ffactor # segment size
        return (int(blockdim[0]), int(blockdim[1]), SS, SS) # coefficient tensor dimension

    def pas_dimensions(self):
        return self.__compute_4d_dims(self.F)

    def stft_dimensions(self):
        return self.__compute_4d_dims(1)


# computes exp(1j*phase), for real-valued 'phase'
def expi(phase):
    return tf.complex(tf.math.cos(phase), tf.math.sin(phase))

def reference_fresnel_CGH(hs, pcloud, ampl = None):
    """
    Generates a hologram from a point-cloud using ray-tracing, quadratic Fresnel approximation.
    INPUTS:
        hs (Hologram_Settings)
        pcloud (Nx3):   point list of N points in (x, y, z) coordinates
        ampl (Nx1):     array of amplitudes [optional, default: all amplitudes = 1]
    """

    # Output hologram
    H = tf.Variable(tf.zeros(hs.res, tf.complex64))

    # Coordinate matrices
    coordmat = lambda i: hs.pp*(tf.range(hs.res[i], dtype=tf.float32))
    X = tf.transpose([coordmat(0)])
    Y = coordmat(1)

    # Constants
    kval = math.pi/hs.wlen         # wave number
    mradfac = hs.wlen/(2*hs.pp*hs.pp)    # maximum radius factor

    # iterate over all points
    for i in range(tf.shape(pcloud)[0]):
        # obtain bounding coordinates to avoid PSF aliasing
        radiusbound = pcloud[i,2] * mradfac
        centerpixel = pcloud[i,0:2]/hs.pp
        minc = np.maximum(centerpixel - radiusbound, 0).astype(int)
        maxc = np.minimum(centerpixel + radiusbound + 1, hs.res).astype(int)
        xrng = slice(minc[0], maxc[0])
        yrng = slice(minc[1], maxc[1])

        # compute PSF within allowed bounds
        sqrrad = tf.square(X[xrng] - pcloud[i,0]) + tf.square(Y[yrng] - pcloud[i,1])
        PSF = expi(kval*(2*pcloud[i,2] + sqrrad/pcloud[i,2]))
        if ampl is not None: PSF *= ampl[i]
        H[xrng, yrng].assign(H[xrng, yrng] + PSF)

    return H

def accurate_fresnel_stereogram(hs, pcloud, ampl = None):
    """
    Returns the coefficients of the accurate phase-added stereogram quadratic Fresnel approximation in a 4D tensor.
    INPUTS:
        hs (Hologram_Settings)
        pcloud (Nx3):   point list of N points in (x, y, z) coordinates
        ampl (Nx1):     array of amplitudes [optional, default: all amplitudes = 1]
    """
    SS = hs.B * hs.F # segment size
    cdim = hs.pas_dimensions() # coefficient tensor dimension
    wk = 2/hs.wlen # double reciprocal of the wavelength

    # PAS coefficient matrix
    C = np.empty(cdim, np.complex64)

    # block center coordinates
    block_centers = lambda i: (np.arange(0, hs.res[i], hs.B, dtype=np.float32) + hs.B/2)*hs.pp
    ucenters = block_centers(0)
    vcenters = block_centers(1)

    # iterate over every block
    for u in range(cdim[0]):
        for v in range(cdim[1]):
            blockdata = np.zeros((SS, SS), np.complex64)
            center = np.array([ucenters[u], vcenters[v]]);

            # iterate over every point
            for p in range(tf.shape(pcloud)[0]):
                pos = center - pcloud[p, 0:2]   
                f = pos / (pcloud[p,2]*hs.wlen)
                fi = np.rint(f*hs.pp*SS).astype(int)
                fr = fi + SS//2

                # are the target Fourier coefficient coordinates within block bounds?
                if np.all(fr>=0) and np.all(fr<SS):
                    coeff = expi(math.pi * (wk * pcloud[p,2] + np.sum(f*pos) - fi.sum()/hs.F))
                    if ampl is not None: coeff *= ampl[p]
                    blockdata[fr[0],fr[1]] += coeff

            # assign computed block to tensor
            C[u,v,:,:] = blockdata

    # transfer to GPU
    return tf.convert_to_tensor(C)

# Forward STFT on a hologram 'H' using block size 'B', returns TF tensor
def forward_stft(H, B):
    cdim = np.array([H.shape[0]//B, H.shape[1]//B, B, B])
    C = np.empty(cdim, np.complex64)

    # FFT transform and reorder samples into TF tensor
    for u in range(cdim[0]):
        for v in range(cdim[1]):
            C[u,v,:,:] = H[B*u:B*(u+1), B*v:B*(v+1)]

    return tf.signal.fftshift(tf.signal.fft2d(tf.convert_to_tensor(C)), (2,3))

# Inverse STFT on a TF tensor 'C', with optional crop block size 'B'
def inverse_stft(C, B = None):
    cdim = tf.shape(C)

    # inverse Fourier transform, slice
    C = tf.signal.ifft2d(tf.signal.ifftshift(C, (2,3)))

    # crop frequency blocks, if applicable
    if B: C = C[:, :, 0:B, 0:B]
    else: B = cdim[2]

    # output hologram
    H = np.empty(cdim[0:2]*B, np.complex64)

    # reorder samples into hologram
    for u in range(cdim[0]):
        for v in range(cdim[1]):
            H[B*u:B*(u+1), B*v:B*(v+1)] = C[u,v,:,:]

    return H