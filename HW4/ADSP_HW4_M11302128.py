# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:01:07 2025

@author: Jessica
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def C420(A):
    # Step 1: 分離 R, G, B
    R = A[:, :, 0].astype(np.float32)
    G = A[:, :, 1].astype(np.float32)
    B = A[:, :, 2].astype(np.float32)

    # Step 2: 轉 YCbCr
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

    # Step 3: Cb 和 Cr（4:2:0，每 2×2 區塊取 1 個像素）
    Cb_down = Cb[::2, ::2]
    Cr_down = Cr[::2, ::2]

    # Step 4: 使用插值法還原
    Cb_up = cv2.resize(Cb_down, (Cb.shape[1], Cb.shape[0]), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_down, (Cr.shape[1], Cr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 5: 還原回 RGB
    R_rec = Y + 1.402 * (Cr_up - 128)
    G_rec = Y - 0.344136 * (Cb_up - 128) - 0.714136 * (Cr_up - 128)
    B_rec = Y + 1.772 * (Cb_up - 128)

    # Clip 值到合法範圍 [0, 255]
    R_rec = np.clip(R_rec, 0, 255)
    G_rec = np.clip(G_rec, 0, 255)
    B_rec = np.clip(B_rec, 0, 255)

    B = np.stack((R_rec, G_rec, B_rec), axis=2).astype(np.uint8)
    return B


A = cv2.imread('Dog_edit.jpg')  
A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

B = C420(A)

psnr_value = psnr(A, B)

print(f"PSNR: {psnr_value:.2f} dB")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(B)
plt.title(f"Reconstructed\nPSNR: {psnr_value:.2f} dB")
plt.axis("off")

plt.tight_layout()
plt.show()
