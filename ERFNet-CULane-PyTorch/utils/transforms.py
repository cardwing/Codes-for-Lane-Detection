import random
import cv2
import numpy as np
import numbers

__all__ = ['GroupRandomCrop', 'GroupCenterCrop', 'GroupRandomPad', 'GroupCenterPad', 'GroupRandomScale', 'GroupRandomHorizontalFlip', 'GroupNormalize']


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images

class GroupRandomCropRatio(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        tw, th = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


class GroupRandomPad(object):
    def __init__(self, size, padding):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img_group):
        assert (len(self.padding) == len(img_group))
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, th - h))
        w1 = random.randint(0, max(0, tw - w))
        h2 = max(th - h - h1, 0)
        w2 = max(tw - w - w1, 0)

        for img, padding in zip(img_group, self.padding):
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=padding))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images


class GroupCenterPad(object):
    def __init__(self, size, padding):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img_group):
        assert (len(self.padding) == len(img_group))
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((th - h) / 2))
        w1 = max(0, int((tw - w) / 2))
        h2 = max(th - h - h1, 0)
        w2 = max(tw - w - w1, 0)

        for img, padding in zip(img_group, self.padding):
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=padding))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images


class GroupConcerPad(object):
    def __init__(self, size, padding):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img_group):
        assert (len(self.padding) == len(img_group))
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = 0
        w1 = 0
        h2 = max(th - h - h1, 0)
        w2 = max(tw - w - w1, 0)

        for img, padding in zip(img_group, self.padding):
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_CONSTANT, value=padding))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images

class GroupRandomScaleNew(object):
    def __init__(self, size=(976, 208), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        scale_w, scale_h = self.size[0] * 1.0 / 1640, self.size[1] * 1.0 / 350
        out_images = list()
        for img, interpolation in zip(img_group, self.interpolation):
            out_images.append(cv2.resize(img, None, fx=scale_w, fy=scale_h, interpolation=interpolation))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images


class GroupRandomScale(object):
    def __init__(self, size=(0.5, 1.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        scale = random.uniform(self.size[0], self.size[1])
        out_images = list()
        for img, interpolation in zip(img_group, self.interpolation):
            out_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images

class GroupRandomMultiScale(object):
    def __init__(self, size=(0.5, 1.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        scales = [0.5, 1.0, 1.5] # random.uniform(self.size[0], self.size[1])
        out_images = list()
        for scale in scales:
            for img, interpolation in zip(img_group, self.interpolation):
                out_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation))
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images

class GroupRandomScaleRatio(object):
    def __init__(self, size=(680, 762, 562, 592), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation
        self.origin_id = [0, 1360, 580, 768, 255, 300, 680, 710, 312, 1509, 800, 1377, 880, 910, 1188, 128, 960, 1784, 1414, 1150, 512, 1162, 950, 750, 1575, 708, 2111, 1848, 1071, 1204, 892, 639, 2040, 1524, 832, 1122, 1224, 2295]

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        w_scale = random.randint(self.size[0], self.size[1])
        h_scale = random.randint(self.size[2], self.size[3])
        h, w, _ = img_group[0].shape
        out_images = list()
        out_images.append(cv2.resize(img_group[0], None, fx=w_scale*1.0/w, fy=h_scale*1.0/h, interpolation=self.interpolation[0])) # fx=w_scale*1.0/w, fy=h_scale*1.0/h
        ### process label map ###
        origin_label = cv2.resize(img_group[1], None, fx=w_scale*1.0/w, fy=h_scale*1.0/h, interpolation=self.interpolation[1])
        origin_label = origin_label.astype(int)
        label = origin_label[:, :, 0] * 5 + origin_label[:, :, 1] * 3 + origin_label[:, :, 2]
        new_label = np.ones(label.shape) * 100
        new_label = new_label.astype(int)
        for cnt in range(37):
            new_label = (label == self.origin_id[cnt]) * (cnt - 100) + new_label
        new_label = (label == self.origin_id[37]) * (36 - 100) + new_label
        assert(100 not in np.unique(new_label))
        out_images.append(new_label)
        return out_images

class GroupRandomRotation(object):
    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = img_group[0].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            out_images = list()
            for img, interpolation, padding in zip(img_group, self.interpolation, self.padding):
                out_images.append(cv2.warpAffine(img, map_matrix, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=padding))
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


class GroupRandomBlur(object):
    def __init__(self, applied):
        self.applied = applied

    def __call__(self, img_group):
        assert (len(self.applied) == len(img_group))
        v = random.random()
        if v < 0.5:
            out_images = []
            for img, a in zip(img_group, self.applied):
                if a:
                    img = cv2.GaussianBlur(img, (5, 5), random.uniform(1e-6, 0.6))
                out_images.append(img)
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            out_images = [np.fliplr(img) for img in img_group]
            if self.is_flow:
                for i in range(0, len(out_images), 2):
                    out_images[i] = -out_images[i]  # invert flow pixel values when flipping
            return out_images
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_group):
        out_images = list()
        for img, m, s in zip(img_group, self.mean, self.std):
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            out_images.append(img)

        # cv2.imshow('img', (out_images[0] + np.array(self.mean[0])[np.newaxis, np.newaxis, ...]).astype(np.uint8))
        # cv2.imshow('label', (out_images[1] * 100).astype(np.uint8))
        # print(np.unique(out_images[1]))
        # cv2.waitKey()
        return out_images


# class ToTorchFormatTensor(object):
#     """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
#     to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

#     def __init__(self, data_type):
#         self.data_type = data_type

#     def __call__(self, pic):
#         if isinstance(pic, np.ndarray):
#             # handle numpy array
#             img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
#         else:
#             # handle PIL Image
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             img = img.view(pic.size[1], pic.size[0], len(pic.mode))
#             # put it from HWC to CHW format
#             # yikes, this transpose takes 80% of the loading time/CPU
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         return img.float()
