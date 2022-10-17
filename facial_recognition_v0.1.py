# %%
import cv2
from matplotlib import pyplot as plt
import pandas as pd

"""Image dataset source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"""

# %%
# Testing loading data from dataset

image_file = './sample_data/Img/'
annotation_file = './sample_data/Anno/'

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'
image_list = ['000001.jpg']



# Loading Identity
identity = pd.read_csv(identity_file, sep=" ", header = None,names=['image', 'image_id'])
bbox = pd.read_csv(bbox_file, delim_whitespace=True)

# Loading Image
image_name = image_list[0]
image_path = image_file + image_name
img = cv2.imread(image_path)
print(f'Loaded image is of a type: {type(img)} with {img.shape} dimensions.')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



# [TODO] Define columns for image: bbox coordinates (starting x,y, ending x,y)
# Visualising bounding box
startX = bbox[bbox['image_id'] == image_name]['x_1'].values[0]
startY = bbox[bbox['image_id'] == image_name]['y_1'].values[0]
endX = startX + bbox[bbox['image_id'] == image_name]['width'].values[0]
endY = startY + bbox[bbox['image_id'] == image_name]['height'].values[0]
cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)       
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%
#Cropping and saving boounding box
crop_img = img[startY:endY, startX:endX]
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

# %%
