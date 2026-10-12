"""RGB fidelity on unclamped outputs; ROI is explicitly a fixed rectangle."""
import math
import torch


def crop_roi(x, roi):
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError('ROI is normalized [x0,y0,x1,y1]')
    h, w = x.shape[-2:]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    if x1 <= x0 or y1 <= y0: raise ValueError('ROI is empty at this resolution')
    return x[..., y0:y1, x0:x1]


class ErrorTotals:
    def __init__(self, roi):
        self.roi = roi; self.count = 0; self.sums = {key: 0.0 for key in ['mse', 'l1', 'roi_mse', 'roi_l1']}

    def add(self, prediction, target):
        if prediction.shape != target.shape: raise ValueError('RGB metric shape mismatch')
        error = prediction.float() - target.float()
        if not torch.isfinite(error).all(): raise FloatingPointError('nonfinite RGB on valid frame')
        values = {'mse': error.square().flatten(1).mean(1), 'l1': error.abs().flatten(1).mean(1),
            'roi_mse': crop_roi(error, self.roi).square().flatten(1).mean(1),
            'roi_l1': crop_roi(error, self.roi).abs().flatten(1).mean(1)}
        for key, v in values.items(): self.sums[key] += v.double().sum().item()
        self.count += len(error)

    def report(self):
        result = {key: value / self.count if self.count else None for key, value in self.sums.items()}
        result['frames'] = self.count
        result['psnr_from_mean_mse'] = (-10 * math.log10(max(result['mse'], 1e-15))) if self.count else None
        return result
