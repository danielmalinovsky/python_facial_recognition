import cv2

def face_crop(image_name, image_file, image_id_col_name, bbox_col_names, bbox_df, show_bbox = False ):
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
        return img
    else:
        # Cropping and saving boounding box
        crop_img = img[startY:endY, startX:endX]
        return crop_img