import cv2
import glob
import os
import numpy as np
import torch
# Apply sobel filter on Image data


def sobel_filter_load_from_dir(rgb_img_dir):
    file_name = data_path.split('/')[-1]

    img_color = cv2.imread(rgb_img_dir, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)

    del img_sobel_x
    del img_sobel_y
    return img_sobel

def sobel_filter(img, file_name):

    # (C, H, W) -> (H, W, C)
    img = img.detach().cpu().numpy().squeeze(0)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    # gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray * 255
    #assert img_gray.shape[2] == 1, f'channel should be gray scale but your image channel is {img.shape[1]}'
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
    #img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
    #img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    img_sobel_norm = img_sobel / 255.0

    img_sobel_norm = torch.Tensor(img_sobel_norm) # torch.from_numpy
    img_sobel_norm = img_sobel_norm.unsqueeze(0).expand(1, 3, img_sobel_norm.shape[-1], img_sobel_norm.shape[-1])
    
    assert img_sobel_norm.shape[1] == 3, 'channel is not 3 for rgb space'
    


    #cv2.imwrite(f'./original_image_{file_name}.png', img)
    #cv2.imwrite(f'./sobel_gray_image_{file_name}.png', img_gray)
    #cv2.imwrite(f'./sobel_filterint_image_{file_name}.png', img_sobel)
    del img_gray
    del img_sobel_x
    del img_sobel_y
    del img_sobel
    del img
    return img_sobel_norm