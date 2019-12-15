import random
import cv2
import numpy as np


def crop(img, offset):
    h1, w1, h2, w2 = offset
    return img[h1:h2, w1:w2, ...]

def center_crop(img, target_size):
    h, w = img.shape[0:2]
    th, tw = target_size
    h1 = max(0, int((h - th) / 2))
    w1 = max(0, int((w - tw) / 2))
    h2 = min(h1 + th, h)
    w2 = min(w1 + tw, w)
    return crop(img, [h1, w1, h2, w2])

def rotation(img, degrees, interpolation=cv2.INTER_LINEAR, value=0):
    if isinstance(degrees, list):
        if len(degrees) == 2:
            degree = random.uniform(degrees[0], degrees[1])
        else:
            degree = random.choice(degrees)
    else:
        degree = degrees

    h, w = img.shape[0:2]
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    img = cv2.warpAffine(
        img,
        map_matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value)
    return img


def flip(img):
    return np.fliplr(img).copy()

def random_flip(img):
    if random.random() < 0.5:
        return flip(img)
    else:
        return img

def random_intensity(img, scale):
    img = img * random.uniform(1-scale, 1+scale)
    img = np.clip(img, 0, 1)
    return img

def blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    img = cv2.GaussianBlur(img, kenrel_size, random.uniform(*sigma))
    return img


def random_blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    if random.random() < 0.5:
        return blur(img, kenrel_size, sigma)
    else:
        return img
def blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    img = cv2.GaussianBlur(img, kenrel_size, random.uniform(*sigma))
    return img


def random_blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    if random.random() < 0.5:
        return blur(img, kenrel_size, sigma)
    else:
        return img
