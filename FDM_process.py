# Created by Junyu at 11/19/2023
#%%
# Import packages
print("Loading packages.")
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
print("Packages loaded.")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def preprocess_images(images):
    processed_images = []
    for img in images:
        processed_images.append(img.flatten())
    return processed_images

#%%
# Debug flag
DEBUG = False
img_dir = r'D:\Guo\0813\all_v2_masked'

images = load_images_from_folder(img_dir)
processed_images = preprocess_images(images)
X = np.array(processed_images)
X_scaled = scale(X)

# K-means clustering
print("K-means clustering...")
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaled)
labels = kmeans.labels_
print("K-means clustering finished.")

#%%
# Plot the K-means clustering result
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
plt.title('K-means clustering result')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar()
plt.show()

#%%
cluster_1_images = [img for img, label in zip(images, labels) if label == 1]
for i, image in enumerate(cluster_1_images):
    plt.subplot(1, len(cluster_1_images), i + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.show()