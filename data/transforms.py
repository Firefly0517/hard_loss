import numpy as np
import operator
import random
import torch
from scipy.ndimage import zoom

class Crop(object):
    """ Crop the given n-dimensional array either at a random location or
    centered.
    """
    def __init__(self, shape, type="center", keep_dim=False):
        assert type in ["center", "random"]
        self.shape = shape
        self.cropping_type = type
        self.keep_dim = keep_dim

    def slow_crop(self, X):
        img_shape = np.array(X.shape)

        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        
        # print('img_shape:', img_shape, 'size', size)
        
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            
            if self.cropping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
            
            elif self.cropping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            
            indexes.append(slice(delta_before, delta_before + size[ndim]))
        
        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = X.copy()
            arr_copy[~mask] = 0
            return arr_copy
        
        _X = X[tuple(indexes)]
        # print('cropped.shape', _X.shape)
        return _X
    
    def fast_crop(self, X):
        # X is a single image (CxWxHxZ)
        shape = X.shape

        delta = [shape[1]-self.shape[1], 
                 shape[2]-self.shape[2], 
                 shape[3]-self.shape[3]]

        if self.cropping_type == "center":
            offset = list(map(operator.floordiv, delta, [2]*len(delta)))
            X = X[:, offset[0]:offset[0]+self.shape[1],
                     offset[1]:offset[1]+self.shape[2],
                     offset[2]:offset[2]+self.shape[3]]

        elif self.cropping_type == "random":
            offset = [
                int(random.random()*128) % (delta[0]+1),
                int(random.random()*128) % (delta[1]+1),
                int(random.random()*128) % (delta[2]+1)
            ]
            X = X[:, offset[0]:offset[0]+self.shape[1],
                     offset[1]:offset[1]+self.shape[2],
                     offset[2]:offset[2]+self.shape[3]]
        else:
            raise ValueError("Invalid cropping_type", self.cropping_type)
        
        return X

    def __call__(self, X):
        return self.fast_crop(X)

class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """
    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None, probability=0.5):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.probability = probability

    def __call__(self, arr):
        if np.random.rand() >= self.probability:
            return arr
        
        img_shape = np.array(arr.shape)
        if type(self.patch_size) == int:
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(self.localization[ndim] - size[ndim]//2, 0)
            else:
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return arr
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return arr_cut

class Pad(object):
    """ Pad the given n-dimensional array
    """
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, X):
        _X = self._apply_padding(X)
        return _X

    def _apply_padding(self, arr):
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **self.kwargs)
        return fill_arr

def isotropic_resample_to_128(x):
    """
    x: numpy array, shape (1, D, H, W)
    return: numpy array, shape (1, 128, 128, 128)
    """
    assert x.ndim == 4 and x.shape[0] == 1

    _, D, H, W = x.shape
    target = 128

    # 等比例缩放因子
    scale = target / max(D, H, W)

    # zoom factor：channel 不缩放
    zoom_factors = (1.0, scale, scale, scale)

    # order=1：线性插值（推荐）
    x_resampled = zoom(
        x,
        zoom=zoom_factors,
        order=1,          # linear interpolation
        mode='constant',
        cval=0.0,
        prefilter=False
    )

    # --- 保证最终是 128³（防止 rounding 问题） ---
    _, d, h, w = x_resampled.shape

    def center_crop_or_pad(arr, target):
        out = np.zeros((1, target, target, target), dtype=arr.dtype)

        d0 = min(d, target)
        h0 = min(h, target)
        w0 = min(w, target)

        sd = (d - d0) // 2
        sh = (h - h0) // 2
        sw = (w - w0) // 2

        td = (target - d0) // 2
        th = (target - h0) // 2
        tw = (target - w0) // 2

        out[:, td:td+d0, th:th+h0, tw:tw+w0] = \
            arr[:, sd:sd+d0, sh:sh+h0, sw:sw+w0]

        return out

    x_final = center_crop_or_pad(x_resampled, target)
    return x_final

if __name__ == '__main__':
    import timeit
    x = np.random.rand(1, 128, 128, 128)

    cut = Cutout((1, 10, 10, 10), probability=1.)
    print(cut(x).shape)

    crop = Crop((1, 121, 128, 121), type="center")
    print(crop(x).shape)

    crop = Crop((1, 121, 128, 121), type="random")
    print(crop(x).shape)

    print("slow crop:", timeit.timeit(lambda: crop.slow_crop(x), number=10000))
    print("fast crop:", timeit.timeit(lambda: crop.fast_crop(x), number=10000))