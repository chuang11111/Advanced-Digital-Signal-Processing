# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:03:23 2025

@author: Jessica
"""

import numpy as np
import matplotlib.pyplot as plt

k = 10  
N = 2 * k + 1  

# STEP 1: Define ideal frequency response Hd[m] 
Hd = []
for m in range(N):
    F = m / N  
    if F >= 0.5:
        F -= 1  
    Hd.append(1j * 2 * np.pi * F)  
Hd = np.array(Hd)

# STEP 2: Inverse DFT to get impulse response 
r1_n = np.fft.ifft(Hd)  

# STEP 3: Rearranging and shifting to range from 0 to 2k
indices = np.arange(0, 2 * k + 1)
r_n = r1_n[np.mod(indices - k, N)] 

# STEP 4: DTFT of r[n] to get frequency response 
F = np.linspace(0, 1, 10000)
R_F_imag = np.zeros_like(F)
n = np.arange(-k, k+1)


for i in range(len(F)):
    R_F_imag[i] = np.sum(r_n * np.exp(-1j * 2 * np.pi * F[i] * n)).imag


def ideal_Hf(F):
    F_wrapped = np.where(F > 0.5, F - 1, F)  
    return 2 * np.pi * F_wrapped  

plt.figure(figsize=(10, 5))
plt.stem(indices, r_n.real)
plt.title("Impulse Response r[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(F, R_F_imag, label="Imag[ R(F) ]", linewidth=2)
plt.plot(F, ideal_Hf(F), '--', label="Ideal", linewidth=2)
plt.title("Frequency Response (Imaginary)")
plt.xlabel("Normalized Frequency (F)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
