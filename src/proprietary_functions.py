import cv2
#import cvlib as cv
import cvlib as cv
from matplotlib import pyplot as plt
import pandas as pd

def face_crop(image_name, image_file, image_id_col_name, bbox_col_names, bbox_df, show_bbox = False, resize = False):
# Loading Image
    image_path = image_file + image_name
    img = cv2.imread(image_path)

    # Setting bounding box coordinates
    startX = bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['x_start']].values[0]
    startY = bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['y_start']].values[0]
    endX = startX + bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['width']].values[0]
    endY = startY + bbox_df[bbox_df[image_id_col_name] == image_name][bbox_col_names['height']].values[0]

    if show_bbox == True:
        cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
        output_img = img
        return output_img
    else:
        # Cropping and saving boounding box
        crop_img = img[startY:endY, startX:endX]
        output_img = crop_img

        if resize == True:
            output_img = cv2.resize(crop_img, (300, 300))
            return output_img
        else:
            return output_img

def bbox_engine(image, image_id = '', method = 0, m1_scale_factor = 1.1, m1_min_neighbors = 6):
    if method == 0:
        faces, confidences = cv.detect_face(image)
        for face in faces:
            #(startX,startY) = face[0],face[1]
            #(endX,endY) = face[2],face[3]

            bbox = {
        #       "image_id" : image_id,
                "x_1" : face[0],
                "y_1" : face[1],
                "width" : face[2] - face[0],
                "height" : face[3] - face[1],
                'x_end' : face[2],
                'y_end' : face[3]}
        return bbox

    if method == 1:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image = image, scaleFactor = m1_scale_factor, minNeighbors = m1_min_neighbors)
        #for face in faces:
            #(startX,startY) = face[0],face[1]
            #(endX,endY) = face[2],face[3]

        bbox = {
    #       "image_id" : image_id,
            "x_1" : faces[0][0],
            "y_1" : faces[0][1],
            "width" : faces[0][2],
            "height" : faces[0][3],
            'x_end' : faces[0][0] + faces[0][2],
            'y_end' : faces[0][1] + faces[0][3]}
    return bbox

def bbox_verification( bbox_annotation, bbox_generated, image_len):

    calc_df = bbox_annotation.merge(bbox_generated, on= "image_id", how='left', suffixes= ['_anno', '_generated'])
    calc_df = calc_df.merge(image_len, on = 'image_id', how = 'left')

    result_df = pd.DataFrame(columns= ['image_id'])

    result_df['image_id'] = calc_df['image_id']
    result_df['image_len'] = calc_df['len']
    result_df['x_1_diff'] = ((calc_df['x_1_anno'] - calc_df['x_1_generated'])**2)**(1/2)
    result_df['y_1_diff'] =  ((calc_df['y_1_anno'] - calc_df['y_1_generated'])**2)**(1/2)
    result_df['width_diff'] = ((calc_df['width_anno'] - calc_df['width_generated'])**2)**(1/2)
    result_df['height_diff'] = ((calc_df['height_anno'] - calc_df['height_generated'])**2)**(1/2)
    result_df['total_diff'] = (result_df['x_1_diff'] + result_df['y_1_diff'] + result_df['width_diff'] + result_df['height_diff'])
    result_df['relative_diff'] = (result_df['total_diff']/result_df['image_len'])

    return result_df

def bbox_show_compare(image_name, image_file, image_id_col_name, bbox_col_names, bbox_df_anno, bbox_df_generated):
# Loading Image

    fig_1, axs_1 = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 7))

    #axs_1.ravel()

    crop_img = face_crop(image_name, image_file, image_id_col_name,
                                bbox_col_names, bbox_df_anno,
                                show_bbox = True)
    axs_1[0].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    crop_img2 = face_crop(image_name, image_file, image_id_col_name,
                                bbox_col_names, bbox_df_generated,
                                show_bbox = True)
    axs_1[1].imshow(cv2.cvtColor(crop_img2, cv2.COLOR_BGR2RGB))

    fig_1.tight_layout()
    plt.show()