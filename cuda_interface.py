# ================================================================
# Copyright 2022, Vrije Universiteit Brussel. All rights reserved.
# Author: David Blinder
# Description: CUDA interface for fast CGH
# ================================================================

import math
import numpy as np
import diffraction_algos as dalgo
import Fast_Diffraction_CUDA as cudacode

def _CUDA_CGH(mode : bool, hs : dalgo.Hologram_Settings, pcloud, ampl):
    """
    Generates a hologram from a point-cloud using CUDA-based implementations
    INPUTS:
        mode: True (refrence) or False (fast PAS)
        hs (Hologram_Settings)
        pcloud (Nx3):   point list of N points in (x, y, z) coordinates
        ampl (Nx1):     array of amplitudes [optional, default: all amplitudes = 1]
    """
    assert pcloud.dtype == np.float32
    assert ampl.dtype == np.complex64
    (holo, time) = cudacode.compute_hologram(mode, hs.wlen, hs.pp, hs.B, hs.res[0] // hs.B, hs.F, np.transpose(pcloud), ampl)
    return (np.transpose(holo), time)

def CUDA_reference_fresnel_CGH(hs : dalgo.Hologram_Settings, pcloud, ampl = np.empty(0, dtype=np.complex64)):
    return _CUDA_CGH(True, hs, pcloud, ampl)

def CUDA_accurate_fresnel_stereogram(hs : dalgo.Hologram_Settings, pcloud, ampl = np.empty(0, dtype=np.complex64)):
    return _CUDA_CGH(False, hs, pcloud, ampl)