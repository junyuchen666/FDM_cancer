# Created by Junyu at 11/11/2023
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

cont_len_min, cont_len_max = 10, 250
nr_contours = 10
pixel_size = 0.36111

path_1 = r'D:/Guo/VIFFI/Images/'  # 所有长条图片的文件夹地址
# Cropped image output directory
output_dir = r'D:/Guo/VIFFI/cropped'
# Cluster image output directory
output_dir_ch1 = r'D:/Guo/VIFFI/cluster_ch1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_ch1):
    os.makedirs(output_dir_ch1)

crop_w = 88  # width of image (88pixel)

val_ch0_mat = []
val_ch1_mat = []
area_ch1_mat = []
number_mat = []

flag = 'DEBUG'

i = 0
#%%
if __name__ == "__main__":
    for item in os.listdir(path_1):  # Load all the images in the folder
        i += 1
        # Debug mode only uses 10 images
        if flag == 'DEBUG':
            if i > 10:
                break
        try:
            if (item == ".DS_Store"):
                print(" find file .Ds_Store")
                os.remove(path_1 + item)
            else:
                impath_1 = path_1 + item
        except:
            print('error')

        img = Image.open(impath_1)  # open image
        print(impath_1)

        original_img_h = img.height
        original_img_w = img.width
        print('original_img_h', original_img_h, 'original_img_w', original_img_w)
        src = np.array(img)

        src = np.array(src)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
        print(minVal, maxVal, minLoc, maxLoc)
        if maxLoc[0] > 150:
            crop_src = img.crop((maxLoc[0] - 50, 0, maxLoc[0] + 50, 100))

            crop_src_ch0 = img.crop((maxLoc[0] - 155, 0, maxLoc[0] - 55, 100))

            src_ch0 = np.array(crop_src_ch0)
            src_ch1 = np.array(crop_src)

            # Normalize the image using cv2.convertScaleAbs
            min_val = np.amin(src_ch0)
            max_val = np.amax(src_ch0)
            src_ch0 = cv2.convertScaleAbs(src_ch0, alpha=255.0 / (maxVal - minVal),
                                        beta=-minVal * 255.0 / (maxVal - minVal))

            min_val = np.amin(src_ch1)
            max_val = np.amax(src_ch1)
            src_ch1 = cv2.convertScaleAbs(src_ch1, alpha=255.0 / (maxVal - minVal),
                                        beta=-minVal * 255.0 / (maxVal - minVal))

            outpath = output_dir + '/' + item + '_ch1.png'
            if flag == 'RUN':
                cv2.imwrite(outpath, src_ch1)
            # plt.imshow(src_ch1)
            # plt.show()

            # outpath = 'cropped_ch0/' + item +'_ch0.png'
            # cv2.imwrite(outpath,src_ch0)
            # plt.imshow(src_ch0)
            # plt.show()

            median_src_ch1 = cv2.medianBlur(src_ch1, 3)
            # get the mask image
            thresh_ch1 = cv2.threshold(median_src_ch1, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # plt.imshow(thresh_ch1)
            # plt.show()
            mean_val_ch1 = cv2.mean(median_src_ch1, mask=thresh_ch1)

            contours_ch1, _ = cv2.findContours(thresh_ch1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # get the lengths of all contours (number of contour coordinates)
            cnt_lengths = np.array([len(cnt) for cnt in contours_ch1])

            # Filter out small stuff (debris, noise) & very large stuff
            ind = np.where((cnt_lengths > cont_len_min) & (cnt_lengths < cont_len_max))[0]

            if len(ind) == 0:  # if no contour was found
                # area_ch1_mat.append(0)
                continue

            # Keep the contours that are in sensible length region
            contours = list(np.array(contours_ch1, dtype=object)[ind])
            # cnt_lengths = list(np.array(cnt_lengths)[ind])
            # Set how many contours should be returned: if available, return nr_contours,
            # otherwise just return as many contours as are available
            iterations = min(len(contours), nr_contours)
            contours = contours[0:iterations]
            cnt = contours[0].astype('int32')
            # img1 = cv2.drawContours(src, contours , -1, (0,0,255), 1)
            M = cv2.moments(cnt)
            # print(M)
            area = cv2.contourArea(cnt)
            print(area)
            src_ch1_2 = src_ch1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # 6次开操作去除噪点小块
            mb = cv2.morphologyEx(thresh_ch1, cv2.MORPH_OPEN, kernel, iterations=2)
            # 3次膨胀操作，是原始图像-确定背景
            sure_bg = cv2.dilate(mb, kernel, iterations=3)
            # plt.imshow(sure_bg)
            # plt.show()
            # 距离变换函数，用于计算前景对象的中心，获取前景图像（前景子图像相互连接）
            dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)  # 最后一个值掩膜尺寸，若前面的是DIST_L1或DIST_C时，强制为3
            # 归一化，将图像相同的归为一个类别
            dist_output = cv2.normalize(dist, 0, 10, cv2.NORM_MINMAX)
            # *30是为了增加亮度
            min_val = np.amin(dist_output)
            max_val = np.amax(dist_output)
            dist_output = cv2.convertScaleAbs(dist_output, alpha=255.0 / (maxVal - minVal),
                                          beta=-minVal * 255.0 / (maxVal - minVal))

            # Plot dist_output

            ret, sure_fg = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)

            # ret,surface = cv2.threshold(dist,dist.max()*0.6,255,cv2.THRESH_BINARY)
            # plt.imshow(surface)
            surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
            unknown = cv2.subtract(sure_bg, surface_fg)

            # Concatenate the images
            img_concat = np.concatenate((dist_output, surface_fg, unknown), axis=1)

            # Plot the concatenated image
            plt.figure()
            plt.imshow(img_concat, cmap='gray')
            plt.show()