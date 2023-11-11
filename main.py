# Created by Junyu at 10/16/2023

# %%
# Import packages
print("Loading packages.")
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import pandas as pd

print("Packages loaded.")
# Debug flag
DEBUG = False

# Read image files
file_path = r'E:\Guo\0813\all_v2'
file_paths = glob.glob(os.path.join(file_path, '*.png'))
save_path = file_path + '_masked'
if not DEBUG:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

if DEBUG:
    file_paths = file_paths[0:10]
file_paths.sort()
empty_list = []
#%%
# Read image
for i in range(0, len(file_paths)):
# for i in range(0, 1):
    # Subtract the filename
    filename = os.path.split(file_paths[i])[1][:-4]
    img = cv2.imread(file_paths[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_draw = img.copy()
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
    # if DEBUG:
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img, cmap='gray')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(blur, cmap='gray')
    #     plt.show()
    # Contour detection
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Remove the background contour
    contours = contours[1:]
    # Remove large contours
    contours = [contours[i] for i in range(len(contours)) if cv2.contourArea(contours[i]) < 2000]
    # Remove small contours
    contours = [contours[i] for i in range(len(contours)) if cv2.contourArea(contours[i]) > 50]

    # Remove child contours in the hierarchy in one line
    contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]

    # pick the contour with the largest area
    if len(contours) == 0:
        empty_list.append(filename)
        continue
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)

    mask = np.zeros(img.shape, np.uint8)
    empty = np.zeros(img_draw.shape, np.uint8)

    cv2.drawContours(empty, [hull], -1, (0, 255, 0), 1)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    # Generate mask
    # mask = cv2.drawContours(mask, contours, -1, 255, -1)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Draw contours
    # img_contour = cv2.drawContours(empty, contours, -1, (0, 255, 0), 1)

    # Draw contours
    # if DEBUG:
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img_contour)
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()

    # Mask the image
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    if DEBUG:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.imshow(img_masked, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.imshow(empty)
        plt.show()
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img_masked = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # img_show = cv2.hconcat([img, img_masked, img_contour])
        # cv2.imshow('image', img_show)
        # cv2.waitKey()
    # Save the image
    cv2.imwrite(save_path + r'\{}.png'.format(filename), img_masked)
    # Save empty image names as csv

df = pd.DataFrame(empty_list, columns=['filename'])
df.to_csv(save_path + r'\empty_list.csv', index=False)
