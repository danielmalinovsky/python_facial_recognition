# %%
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import tensorflow as tf
import keras as keras
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from itertools import compress

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Input, Subtract,concatenate, Flatten, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow import float32
from tensorflow import stack
from tensorflow import convert_to_tensor

#import cvlib as cv


#resnet 50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# User defined functions
import src.proprietary_functions as src

"""Image dataset source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"""


# %%
# Setting filepaths
image_file = './sample_data/Img/'
annotation_file = './sample_data/Anno/'
export_file = './export'

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'

# Train/test split variables
random_seed = 123
test_size = 0.2
validation_size = 0.2

# Bbox parameters
method1_min_neighbours = 13

# Setting column names
image_id_col = 'image_id'
bbox_col_names = {
    'x_start' : 'x_1',
    'y_start' : 'y_1',
    'width' : 'width',
    'height' : 'height',
    'x_end' : '',
    'y_end' : ''}

identity_file = annotation_file + 'identity_CelebA.txt'
bbox_file = annotation_file + 'list_bbox_celeba.txt'



# Loading dataset metadata
identity = pd.read_csv(identity_file, sep=" ", header = None,names=['image', 'image_id'])
bbox = pd.read_csv(bbox_file, delim_whitespace=True)


#%% Filtering faces that appear at least 20 times
labels_annot = pd.DataFrame(identity.image_id.value_counts(ascending=True)).query('image_id > 20').index.tolist()
identity_filtered = identity[identity['image_id'].isin(labels_annot)]

#%% [SPRINT 2] Train/test split of the annotations
imgs = identity_filtered['image']
labels = identity_filtered['image_id']

temp_imgs, test_imgs, _, __ = train_test_split(imgs, labels,
                                               test_size = test_size,
                                               random_state = random_seed,        
                                               stratify = labels)
train_imgs, valid_imgs, _, __ = train_test_split(temp_imgs, _,
                                               test_size = validation_size/(1-test_size),
                                               random_state = random_seed,        
                                               stratify = _)

#%% 
# Safe train/test split

if not os.path.exists(export_file):
    os.makedirs(export_file)

if not os.path.exists(export_file + '/setting'):
    os.makedirs(export_file + '/setting')

if export_file != '':
    train_imgs.to_csv(export_file + '/setting/train_imgs.csv', index = False)
    valid_imgs.to_csv(export_file + '/setting/valid_imgs.csv', index = False)
    test_imgs.to_csv(export_file + '/setting/test_imgs.csv', index = False)

#%%
# [SPRINT 2] Random selection
random.seed(random_seed)
random_pics = random.choices(identity_filtered['image'].values, k=10)
bbox_filtered = bbox[bbox['image_id'].isin(random_pics)] #generate bboxes

# %% [SPRINT 2] BBox engine demonstration

img = cv2.imread(image_file+random_pics[0])
bbox_generated = src.bbox_engine(img, method = 0)

cv2.rectangle(img, (bbox_generated['x_1'],bbox_generated['y_1']), (bbox_generated['x_end'],bbox_generated['y_end']), (0,255,0), 2)       
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%
# [SPRINT 2 demonstration]
# %% Generating Bboxes

bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])

for pic in random_pics:

    img = cv2.imread(image_file+pic)
    bbox_coordinates = src.bbox_engine(img, method = 1, m1_min_neighbors= method1_min_neighbours)
    bbox_coordinates['image_id'] = pic
    bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)


# %% Bbox verification

image_lens = pd.DataFrame(columns= ['image_id', 'len'])

for pic, in zip(random_pics):

    image_path = image_file + pic
    img = cv2.imread(image_path)
    len_dict = {'image_id' : pic, 'len' : len(img)}
    image_lens = image_lens.append(len_dict, ignore_index = True)

bbox_verification_df = src.bbox_verification(bbox_filtered, bbox_generated, image_lens)

# %% Large bbox mismatch plot
large_diff_images = bbox_verification_df[bbox_verification_df['relative_diff'] > 0.5]['image_id']

for i in range(len(large_diff_images)):
    src.bbox_show_compare(large_diff_images.iloc[i], image_file, 'image_id', bbox_col_names, bbox_filtered, bbox_generated)

#%%
# [SPRINT 2] Plotting 10 random pictures
fig_1, axs_1 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_1 in zip(random_pics, axs_1.ravel()):

    image_path = image_file + pic
    img = cv2.imread(image_path)

    ax_1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_1.title.set_text(pic)

fig_1.tight_layout()
plt.show()

#%%
# [SPRINT 2] Generating bboxes for 10 random pictures
fig_2, axs_2 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_2 in zip(random_pics, axs_2.ravel()):

    crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = True)

    ax_2.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    ax_2.title.set_text(pic)

fig_2.tight_layout()
plt.show()

#%%
# [SPRINT 2] Cropping 10 random pictures
fig_3, axs_3 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_3 in zip(random_pics, axs_3.ravel()):

    crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = False)

    ax_3.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    ax_3.title.set_text(pic)

fig_3.tight_layout()
plt.show()

#%%
# [SPRINT 2] Resizing of 10 random cropped pictures.
fig_4, axs_4 = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 35))

for pic, ax_4 in zip(random_pics, axs_4.ravel()):

    resize_crop_img = src.face_crop(pic, image_file, 'image_id',
                             bbox_col_names, bbox_generated,
                             show_bbox = False,
                             resize = True)
    
    ax_4.imshow(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))
    ax_4.title.set_text(pic)

fig_4.tight_layout()
plt.show()


# %%
# [Sprint 3] Task 1 - Dist per person

plt.figure(figsize=(20,6))
pd.DataFrame(identity.image_id.value_counts(ascending=True)).groupby('image_id').size().plot.bar()
plt.show()

# %%
# [Sprint 3] Task 2 - Graph distribution of atributes

attbs = pd.read_csv("./data/Anno/list_attr_celeba.txt", delim_whitespace=True)
dist_var_df = pd.DataFrame(columns = ['var', 'dist_1_rel', 'dist_0_rel'])
for col in attbs.columns:

    dist_1_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[1,col]
    dist_0_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[-1,col]

    dist_var_df = pd.concat((dist_var_df, pd.DataFrame([col, dist_1_rel, dist_0_rel]).transpose().rename(columns = {0:'var', 1:'dist_1_rel', 2:'dist_0_rel'})))

dist_var_df
# %% Plotting distributions

sorted_vars = dist_var_df.sort_values(by = "dist_0_rel")
sorted_vars = sorted_vars.set_index(sorted_vars['var'])

sorted_vars.plot(kind='bar', stacked=True)
plt.legend(['Yes', 'No'], loc = 'upper left')

# %% [Sprint 3] Task 2 - Correlation Matrix (PETER)

corr_df = src.corr_atrbs(attbs)

# %%
#positive associations
corr_df.query('coef > 0.3')

# %%
#negative associations
corr_df[corr_df['coef'] < - 0.3]

# %%
# Correlation matrix

src.corr_matrix(attbs)

# %%
# Plotting correlation matrix

src.corr_matrix(attbs, True)

# %%
#relative distributions of attributes
dist_var_df = pd.DataFrame(columns = ['var', 'dist_1_rel', 'dist_0_rel'])

for col in attbs.columns:

    dist_1_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[1,col]
    dist_0_rel = pd.DataFrame(attbs[col].value_counts(normalize = True)).loc[-1,col]

    dist_var_df = pd.concat((dist_var_df, pd.DataFrame([col, dist_1_rel, dist_0_rel]).transpose().rename(columns = {0:'var', 1:'dist_1_rel', 2:'dist_0_rel'})))

dist_var_df

# %%
#joining annotations with attributes
df_joined = identity_filtered.merge(attbs.reset_index().rename(columns = {'index':'image'}), on = 'image')
df_joined

x = src.balanced_pairs(df_joined, 200, ['Wearing_Necklace', 'Heavy_Makeup', 'Wearing_Lipstick'])

# %% [Sprint 3] Task 3 - Generate balance pairs (PETER)

src.balanced_pairs(df_joined, 10, ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive'])

# %%
# generating balanced pairs

src.balanced_pairs(df_joined, 15, ['Wearing_Necklace', 'Straight_Hair', 'Blond_Hair'])

# %% [Sprint 3] Task 4 - Resnet 50 with keras application

img_from_array = Image.fromarray(cv2.cvtColor(resize_crop_img, cv2.COLOR_BGR2RGB))

img_resnet = img_from_array.resize((224,224))
img_resnet

#%%
#ResNet50 model
model = ResNet50(weights='imagenet')

x_test = image.img_to_array(img_resnet)
x_test = np.expand_dims(x_test, axis=0)
x_test = preprocess_input(x_test)

preds = model.predict(x_test)
print('Predicted:', decode_predictions(preds, top=3)[0])

# %% [Sprint 3] Task 5 - Resnet Binary classification

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %% Generated list of cropped images using bbox_engine

croplist1 = []
bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])
for i,pic in zip(x.index, x['pic_1']):
    bbox_coordinates = src.bbox_engine_img_input(pic, './data/Img/img_celeba/') #Dan, please, change

    #Generation of bounding boxes for the pic_1, if the bounding boxes are available.
    if bbox_coordinates != None:
        bbox_coordinates['image_id'] = pic
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['height']].values[0]
        img =  cv2.imread('./data/Img/img_celeba/' + pic)
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        crop_img = convert_to_tensor(crop_img, dtype=float32)
        croplist1.append(crop_img)
    #If the bounding boxes are not available, input 0 array.
    else:
        print(pic,'... no bounding boxes detected')
        zz = np.zeros(shape = (224, 224, 3))
        zz = convert_to_tensor(zz, dtype=float32)
        croplist1.append(zz)

croplist2 = []
bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])
for i,pic in zip(x.index, x['pic_2']):
    bbox_coordinates = src.bbox_engine_img_input(pic, './data/Img/img_celeba/') #Dan, please, change

    #Generation of bounding boxes for the pic_1, if the bounding boxes are available.
    if bbox_coordinates != None:
        bbox_coordinates['image_id'] = pic
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['height']].values[0]
        img =  cv2.imread('./data/Img/img_celeba/' + pic)
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        crop_img = convert_to_tensor(crop_img, dtype=float32)
        croplist2.append(crop_img)

    #If the bounding boxes are not available, input 0 array.
    else:
        print(pic,'... no bounding boxes detected')
        zz = np.zeros(shape = (224, 224, 3))
        zz = convert_to_tensor(zz, dtype=float32)
        croplist2.append(zz)

print(len(croplist1))
print(len(croplist2))

# %% Filtering image pairs with bboxes and converting numpy arrays of images into tensorflow objects

#Filtering only those pairs, which have bounding boxes for both pictures.
inds_to_keep = []
for i in range(len(croplist1)):
    if (np.sum(croplist1[i].numpy()) == 0) or (np.sum(croplist2[i].numpy()) == 0):
        inds_to_keep.append(False)
    else:
        inds_to_keep.append(True)

#tf.stack -> convert numpy arrays into tensorflow objects.
no_classes_1 = len(x.loc[inds_to_keep,'image_id_1'].values)
no_classes_2 = len(x.loc[inds_to_keep,'image_id_2'].values)
croplist1 = stack(np.asarray(list(compress(croplist1, inds_to_keep))))
croplist2 = stack(np.asarray(list(compress(croplist2, inds_to_keep))))
labels = x.loc[inds_to_keep,'label'].values
labels = stack(labels)

print(croplist1.shape)
print(croplist2.shape)
print(labels.shape)

# %% Preparing branches

#%%
base_model = ResNet50(weights = 'imagenet', input_shape = (224,224,3), include_top = False)
left_input = Input(shape=(224,224,3))
right_input = Input(shape=(224,224,3))
#%%
l = base_model(left_input)
r = base_model(right_input)
#%%
subtracted = Subtract()([l, r])
subtracted = Flatten()(subtracted)
preds = Dense(1, activation='sigmoid')(subtracted)
model = keras.models.Model(inputs=[left_input, right_input], outputs=preds)

model.summary()
#%%
model.compile(optimizer ='adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
#%%
modelicek = model.fit([croplist1, croplist2], labels,  epochs=5, validation_split=0.2, shuffle = True) 

#%%
### [SPRINT 4] Create data processing pipeline

#creating a dataframe with paths to the images with their labels
temp = pd.DataFrame(train_imgs).merge(identity_filtered, on = 'image')
temp['image'] = './data/Img/img_celeba/' + temp['image']

#separating the images and labels and then creating a TF dataset.
pics = temp['image'].values
labels = temp['image_id'].values
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((pics, labels))
#%%
#Displaying 1st observation of the dataset
for d in dataset.take(1):
    display(d)

#%%
#printing the last observation, particularly the last image path as a tensor
print(d[0]) 
#%%
#printing the last image path as numpy (byte string) by converting it from tensor
print(d[0].numpy())
#%%
#decoding the byte string to utf-8 -> therefore, we would assume that this logic will work when mapping a custom function.
print(d[0].numpy())
#%%
#Unfortunately, it doesn't work. According to several sources, we would extract the string path by numpy() method to the tensor. However, it shows an error that tensor doesnt have a numpy object.
#we tried another approach such as using tf.io.read_file() and tf.image_decode_jpeg() which would be input to the bounding box generator from cv2 library. Though, this neither doesnt work as well.

def process_images(filename, label):
    
    bbox_col_names = {
                        'x_start' : 'x_1',
                        'y_start' : 'y_1',
                        'width' : 'width',
                        'height' : 'height',
                        'x_end' : '',
                        'y_end' : ''}
        
    def bbox_engine(pic, m1_scale_factor = 1.1, m1_min_neighbors = 13):
        h = tf.io.read_file(pic)
        i = tf.image.decode_jpeg(h)
        img = np.array(i)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image = img, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
        while len(faces) < 1:
            m1_min_neighbors -= 1
            faces = face_cascade.detectMultiScale(image = img, scaleFactor =m1_scale_factor, minNeighbors = m1_min_neighbors)
            if m1_min_neighbors < 0:
                break
    
        if len(faces) > 0:

            bbox = {
                    "x_1" : faces[0][0],
                    "y_1" : faces[0][1],
                    "width" : faces[0][2],
                    "height" : faces[0][3],
                    'x_end' : faces[0][0] + faces[0][2],
                    'y_end' : faces[0][1] + faces[0][3]}

            return bbox
    # read actual file from path to a Tensor
    
    bbox_coordinates = bbox_engine(filename)
    bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])

    #Generation of bounding boxes for the pic_1, if the bounding boxes are available.
    if bbox_coordinates != None:
        bbox_coordinates['image_id'] =tf.io.read_file(filename)
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] ==tf.io.read_file(filename)][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == tf.io.read_file(filename)][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] ==tf.io.read_file(filename)][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] ==tf.io.read_file(filename)][bbox_col_names['height']].values[0]
        img =  cv2.imread(tf.io.read_file(filename))
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        crop_img =tf.convert_to_tensor(crop_img, dtype=tf.float32)

    #If the bounding boxes are not available, input 0 array.
    else:
        crop_img = np.zeros(shape = (224, 224, 3))
        crop_img = tf.convert_to_tensor(crop_img, dtype=tf.float32)

    return crop_img, label
dataset = dataset.map(process_images)


# %%
## [SPRINT 4] Evaluate training progress

#%%
base_model = ResNet50(weights = 'imagenet', input_shape = (224,224,3), include_top = False)
left_input = Input(shape=(224,224,3))
right_input = Input(shape=(224,224,3))
#%%
l = base_model(left_input)
r = base_model(right_input)
#%%
subtracted = Subtract()([l, r])
subtracted = Flatten()(subtracted)
preds = Dense(1, activation='sigmoid')(subtracted)
model = keras.models.Model(inputs=[left_input, right_input], outputs=preds)

model.summary()
#%%
model.compile(optimizer ='adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
#%%
train_sample = set(random.choices(identity_filtered['image'].values, k=700))
valid_sample = set(random.choices(list(set(identity_filtered['image'].values).difference(train_sample)), k = 300))
#%%

tr = identity_filtered[identity_filtered['image'].isin(train_sample)]
vl = identity_filtered[identity_filtered['image'].isin(valid_sample)]
#%%

tr_s = tr.merge(attbs.reset_index().rename(columns = {'index':'image'}), on = 'image')

tr_crop = []
bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])
for pic in tr_s['image']:
    bbox_coordinates = src.bbox_engine_img_input(pic, './data/Img/img_celeba/') #Dan, please, change

    #Generation of bounding boxes for the pic_1, if the bounding boxes are available.
    if bbox_coordinates != None:
        bbox_coordinates['image_id'] = pic
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['height']].values[0]
        img =  cv2.imread('./data/Img/img_celeba/' + pic)
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        crop_img = convert_to_tensor(crop_img, dtype=float32)
        tr_crop.append(crop_img)
    #If the bounding boxes are not available, input 0 array.
    else:
        print(pic,'... no bounding boxes detected')
        zz = np.zeros(shape = (224, 224, 3))
        zz = convert_to_tensor(zz, dtype=float32)
        tr_crop.append(zz)

vl_s = vl.merge(attbs.reset_index().rename(columns = {'index':'image'}), on = 'image')

vl_crop = []
bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])
for pic in vl_s['image']:
    bbox_coordinates = src.bbox_engine_img_input(pic, './data/Img/img_celeba/') #Dan, please, change

    #Generation of bounding boxes for the pic_1, if the bounding boxes are available.
    if bbox_coordinates != None:
        bbox_coordinates['image_id'] = pic
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] == pic][bbox_col_names['height']].values[0]
        img =  cv2.imread('./data/Img/img_celeba/' + pic)
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        crop_img = convert_to_tensor(crop_img, dtype=float32)
        vl_crop.append(crop_img)
    #If the bounding boxes are not available, input 0 array.
    else:
        print(pic,'... no bounding boxes detected')
        zz = np.zeros(shape = (224, 224, 3))
        zz = convert_to_tensor(zz, dtype=float32)
        vl_crop.append(zz)
#%%

print(len(tr_crop))
print(len(vl_crop))

# %% Filtering image pairs with bboxes and converting numpy arrays of images into tensorflow objects

inds_to_keep_tr = [False if np.sum(tr_crop[i].numpy()) == 0 else True for i in range(len(tr_crop))]
inds_to_keep_vl = [False if np.sum(vl_crop[i].numpy()) == 0 else True for i in range(len(vl_crop))]

#tf.stack -> convert numpy arrays into tensorflow objects.
X_train = stack(np.asarray(list(compress(tr_crop, inds_to_keep_tr))))
X_valid = stack(np.asarray(list(compress(vl_crop, inds_to_keep_vl))))
y_train = stack(tr_s.loc[inds_to_keep_tr,'image_id'].values)
y_valid = stack(vl_s.loc[inds_to_keep_vl,'image_id'].values)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
#%%


#%%
#Acurracy plot function for both sets
plt.plot(modelicek.history['acc'])
plt.plot(modelicek.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Interpration - TBD (Dan pls)

#Loss function plot for both sets
plt.plot(modelicek.history['loss'])
plt.plot(modelicek.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Interpration - TBD (Dan pls)





















#%%
## [SPRINT 4] team photos

photo_initials = ['PN','DM','PH','RP','NM']

bbox_generated = pd.DataFrame(columns= ['image_id', 'x_1', 'y_1', 'width', 'height', 'x_end', 'y_end'])
data = []

#cropping photos
for init_name in photo_initials:
    for num in ['1','2','3','4']:
        bbox_coordinates = src.bbox_engine_img_input(f'{init_name}_{num}.jpg', './team_photos/')
        bbox_coordinates['image_id'] = f'{init_name}_{num}.jpg'
        bbox_generated = bbox_generated.append(bbox_coordinates, ignore_index = True)
        startX = bbox_generated[bbox_generated['image_id'] == f'{init_name}_{num}.jpg'][bbox_col_names['x_start']].values[0]
        startY = bbox_generated[bbox_generated['image_id'] == f'{init_name}_{num}.jpg'][bbox_col_names['y_start']].values[0]
        endX = startX + bbox_generated[bbox_generated['image_id'] == f'{init_name}_{num}.jpg'][bbox_col_names['width']].values[0]
        endY = startY + bbox_generated[bbox_generated['image_id'] == f'{init_name}_{num}.jpg'][bbox_col_names['height']].values[0]
        img =  cv2.imread( './team_photos/'+ f'{init_name}_{num}.jpg')
        crop_img = cv2.resize(img[startY:endY, startX:endX], (224, 224))
        #temp.append(crop_img)
        #crop_img = convert_to_tensor(crop_img, dtype=float32)
        data.append(crop_img)

#%%
photo_names = [i+'_'+j for i in photo_initials for j in ['1','2','3','4']]
#%%
#train test split
annotations = [num for num, _ in enumerate(photo_initials)]
labels = [annot for annot in annotations for i in range(0,4)]
train_ind = []
test_ind = []
for i, phot in enumerate(photo_names):
    if phot.split('_')[1] in (['3','4']):
        test_ind.append(i)
    else:
        train_ind.append(i)
#%%
data_train = [data[i] for i in train_ind]
y_train = [labels[i] for i in train_ind]
data_test = [data[i] for i in test_ind]
y_test = [labels[i] for i in test_ind]


y_train = np.array(y_train)
y_test = np.array(y_test)
data_test = np.array(data_test)
data_train = np.array(data_train)
# %%
#function from Viktor
def sort_labels_by_classes(labs):
    result = []
    for i in range(len(photo_initials)):
        #  np.where returns the indices of elements in an input array where the given condition is satisfied
        result.append(np.where(labs == i)[0])
    return result
#%%
train_classes = sort_labels_by_classes(y_train)
test_classes = sort_labels_by_classes(y_test)
# %%

#function from Viktor
def create_triplets(data, labels):
    triplets_data = []
    class_count = len(photo_initials)
    # go per each of cloth class
    for i in range(len(labels)):
        # class for processing
        class_label_length = len(labels[i])
        # go for each of item in current cloth class
        for j in range(class_label_length - 1):
            # get the positive pair - n and n+1 item from current label
            idx1, idx2 = labels[i][j], labels[i][j + 1]
            # random generate increment from 1-9 to add to current class index
            inc = random.randrange(1, class_count)
            # add increment to class index and apply modulo by class count to get random negative class label index
            negative_label_index = (i + inc) % class_count
            # take random item from other label items to create a negative pair
            negative_sample = random.choice(labels[negative_label_index])
            # save negative pair and set label to 0
            triplets_data.append([data[idx1], data[idx2], data[negative_sample]])
    # numpy arrays are easier to work with, so type list into it
    return np.array(triplets_data)

# %%
X_train = create_triplets(data_train, train_classes)
X_test = create_triplets(data_test, test_classes)
# %%
def show_image(image):
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.grid(False)
    plt.show()

triplet = 3
# show images at this index
show_image(X_train[triplet][0])
show_image(X_train[triplet][1])
show_image(X_train[triplet][2])
# %%
def initialize_base_network():
    input = Input(shape=(224,224,3))
    x = Flatten()(input)
    x = Dense(1200, activation='relu')(x)
    x = Dense(1200, activation='relu')(x)
    x = Dense(1200, activation='relu')(x)
    return Model(inputs=input, outputs=x)
#%%
embedding = initialize_base_network()
tf.keras.utils.plot_model(embedding, show_shapes=True)
# %%
import tensorflow.keras.backend as K
class SiameseNet(tf.keras.layers.Layer):
    # set the backbone model in constructor
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, feat):
        # get feature vectors from anchor
        feats = self.model(feat[0])
        # from positive image
        pfeats = self.model(feat[1])
        # and from negative image
        nfeats = self.model(feat[2])
        # concatenate vectors to a matrix
        result = tf.stack([feats, pfeats, nfeats])
        return result
class TripletLoss(tf.keras.layers.Layer):
    # margin is settable hyperparameter in constructor
    def __init__(self, margin):
        self.margin = margin
        super().__init__()
        
    # function calculating distance between features
    def distance(self, x, y):
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, K.epsilon()))
    

    def call(self, features):
        # get anchor-positive distance
        pos = self.distance(features[0], features[1])
        # anchor-negative distance
        neg = self.distance(features[0], features[2])
        # difference between anchor positive and anchor negative distances
        loss = pos - neg
        # get overall loss
        return tf.maximum(loss + self.margin, 0.0)
# %%
def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
#%%
# anchor branch
image_input = Input(shape=(224,224,3), name='image_input')
# positive image branch
positive_input = Input(shape=(224,224,3), name='positive_input')
# negative image branch
negative_input = Input(shape=(224,224,3), name='negative_input')

siamese = SiameseNet(embedding)([image_input, positive_input, negative_input])
loss = TripletLoss(margin=1.0)(siamese)
model = Model(inputs=[image_input, positive_input, negative_input], outputs=loss)
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = identity_loss)
tf.keras.utils.plot_model(model, show_shapes=True)
# %%
# we don't need labels, everything is handled inside triplet loss layer, so we just set labels to 1, they will not be used anyway
history = model.fit([X_train[:,0], X_train[:,1], X_train[:,2]], np.ones(X_train.shape[0]), batch_size=128, verbose=1, validation_data=([X_test[:,0], X_test[:,1], X_test[:,2]], np.ones(X_test.shape[0])), epochs=20)
# %%
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.grid()
    plt.legend()
plot_metrics(metric_name='loss', title="Loss", ylim=0.2)



#%%
def create_pairs(data, labels):
    pairs_data = []
    pairs_labels = []
    ids = []
    class_count = len(photo_initials)

    # go per each of cloth class
    for i in range(len(labels)):
        # class for processing
        class_label_length = len(labels[i])
        # go for each of item in current cloth class
        for j in range(class_label_length - 1):
            # get the positive pair - n and n+1 item from current label
            idx1, idx2 = labels[i][j], labels[i][j + 1]
            # save to list and set label to 1
            pairs_data.append([data[idx1], data[idx2]])
            pairs_labels.append(1.0)
            ids.append([i, i])

            # random generate increment from 1-9 to add to current class index
            inc = random.randrange(1, class_count)
            # add increment to class index and apply modulo by class count to get random negative class label index
            negative_label_index = (i + inc) % class_count
            # take random item from other label items to create a negative pair
            negative_sample = random.choice(labels[negative_label_index])
            # save negative pair and set label to 0
            pairs_data.append([data[idx1], data[negative_sample]])
            pairs_labels.append(0.0)
            ids.append([i, negative_label_index])

    # numpy arrays are easier to work with, so type list into it
    return np.array(pairs_data), np.array(pairs_labels), np.array(ids)
# %%
X_test, Y_test, ids = create_pairs(data_test, test_classes)
left_pair = X_test[:,0]
left_pair_pred = embedding.predict(left_pair)
right_pair = X_test[:,1]
right_pair_pred = embedding.predict(right_pair)
positive_left_pred = left_pair_pred[0::2]
positive_right_pred = right_pair_pred[0::2]
positive_distances = np.linalg.norm(positive_left_pred - positive_right_pred, axis=1)
negative_left_pred = left_pair_pred[1::2]
negative_right_pred = right_pair_pred[1::2]
negative_distances = np.linalg.norm(negative_left_pred - negative_right_pred, axis=1)

#%%
pd.Series(positive_distances).describe()
#%%










#%%
fig = plt.figure()
ax = fig.add_axes([0,0,1, 1])
ax.boxplot([positive_distances, negative_distances])
plt.xticks([1, 2], ['Positive', 'Negative'])
ax.grid()
plt.show()


#%% 
def compute_accuracy(left_pred, right_pred, y_true):
    y_pred = np.linalg.norm(left_pair_pred - right_pair_pred, axis=1)
#     # 1 for the same - distance is smaller than 3.0, 0 for the different
    pred = y_pred < 7.0
    return np.mean(pred == y_true)

test_accuracy = compute_accuracy(embedding.predict(X_test[:,0]), embedding.predict(X_test[:,1]), Y_test)
print(f'Test accuracy: {test_accuracy*100:.2f}%')
# %%
from sklearn.metrics import confusion_matrix
preds = (np.linalg.norm(left_pair_pred - right_pair_pred, axis=1) < 7.0)
confusion_matrix(Y_test, preds)
# %%

