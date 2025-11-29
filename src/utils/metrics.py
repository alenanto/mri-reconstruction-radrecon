import math
import numpy as np

from skimage.metrics import structural_similarity

def psnr_batch(y_batch, y_pred_batch):
    #calculate psnr for every batch and return mean
    mean_psnr = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_psnr += psnr(y, y_pred, y.max())
    return mean_psnr / y_batch.shape[0]

def psnr(y, y_pred, MAX_PIXEL_VALUE=1.0):
    rmse_ = rmse(y, y_pred)
    if rmse_ == 0:
        return float('inf')
    return 20 * math.log10(MAX_PIXEL_VALUE/rmse_+1e-10)

def ssim_batch(y_batch, y_pred_batch):
    mean_ssim = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_ssim += ssim(y, y_pred)
    return mean_ssim / y_batch.shape[0]

def ssim(y, y_pred):
    return structural_similarity(y, y_pred, data_range=y.max() - y.min())

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    return math.sqrt(mse(y, y_pred))