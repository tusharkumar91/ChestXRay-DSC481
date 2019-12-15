### Source - https://github.com/jacobgil/pytorch-explain-black-box ###

import torch
from torch.autograd import Variable
from torchvision import models
from models.resnet_xray_model import ResnetXRayClassificationModel
import cv2
import sys
import numpy as np
from models.build_model import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    
    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)
    
    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save(mask, img, blurred, index=None):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)
    
    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)
    if index is not None:
        cv2.imwrite("p4/{}_perturbated.png".format(index), np.uint8(255 * perturbated))
        cv2.imwrite("p4/{}_heatmap.png".format(index), np.uint8(255 * heatmap))
        cv2.imwrite("p4/{}_mask.png".format(index), np.uint8(255 * mask))
        cv2.imwrite("p4/{}_cam.png".format(index), np.uint8(255 * cam))
    cv2.imwrite("perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite("heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite("mask.png", np.uint8(255 * mask))
    cv2.imwrite("cam.png", np.uint8(255 * cam))

def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))
    
    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()
    
    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model():
    model = ResnetXRayClassificationModel()
    model.load_state_dict(torch.load('checkpoints/best_weights_resnet_oversample_70_15_15_intensity_0.1_rot10_crop_alldata_acc.pth', map_location='cpu'))
    model.eval()
    if use_cuda:
        model.cuda()
    for p in model.resnet.parameters():
        p.requires_grad = False
    for p in model.fc1.parameters():
         p.requires_grad = False
    
    return model


if __name__ == '__main__':
    # Hyper parameters.
    # TBD: Use argparse
    tv_beta = 0
    learning_rate = 0.1
    max_iterations = 100
    l1_coeff = 0
    tv_coeff = 0

    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 1000
    l1_coeff = 0.1
    tv_coeff = 2

    model = load_model()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    #model = load_model()
    original_img = cv2.imread('sample_slide_2.jpeg', 1)
    original_img = cv2.resize(original_img, (256, 256))
    original_img = original_img[17:241, 17:241, :]
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (51, 51), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 51)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)
    
    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    print(img.shape)
    with torch.no_grad():
        target = torch.nn.Softmax()(model(img))
    print(target)
    category = np.argmax(target.cpu().data.numpy())
    print("Category with highest probability", category)
    best_prob = target[0, category]
    for i in range(max_iterations):
        print('At {}/{}'.format(i+1, max_iterations))
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        # upsampled_mask = \
        #     upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
        #                           upsampled_mask.size(3))
        
        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)
        
        noise = np.zeros((224, 224), dtype=np.float32)
        cv2.randn(noise, 0, 0.02)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise
        
        outputs = torch.nn.Softmax()(model(perturbated_input))
        #print(outputs)
        #print(torch.mean(torch.abs(1 - mask)))
        
        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, category]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
        #if outputs[0, category] < best_prob:
        print(outputs[0, category])
        upsampled_mask = upsample(mask)
        save(upsampled_mask, original_img, blurred_img_numpy, index=i)
        best_prob = outputs[0, category]
