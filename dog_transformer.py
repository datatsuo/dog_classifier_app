# this code define a function which add dog ears and nose to
# a human image

import cv2
import numpy as np
from PIL import Image
import os
from werkzeug import secure_filename

image_folder = './static/images/'

# extract pre-trained face/eye/nose detector
# for face
cascade_path_face = './haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path_face)
# for eyes
cascade_path_eye = './haarcascades/haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(cascade_path_eye)
# for nose
cascade_path_nose = './haarcascades/haarcascade_mcs_nose.xml'
nose_cascade = cv2.CascadeClassifier(cascade_path_nose)

# detect face, eyes and nose in the human image,
# and then add dog ears and dog nose to the image
def add_dog_parts(img_path):

    # read the image
    img = cv2.imread(img_path)
    # convert BGR image to gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray_img)
    for (x,y,w,h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        count = 0
        # align such that left eye first and then right eye
        if(eyes[0][0]> eyes[1][0]):
            eyes = np.array([eyes[1], eyes[0]])
        for (ex,ey,ew,eh) in eyes:
            if(count == 0): # for left eye
                # load the left dog ear
                img_ear= Image.open('./dog_parts_images/ear_right.jpg')
                dev_ex = -int(ey/8)
            elif(count == 1): # for right eye
                # lead the right dog ear
                img_ear= Image.open('./dog_parts_images/ear_left.jpg')
                dev_ex = +int(ey/8)

            # resize the dog ear image and extract the background image
            # to put it
            w_ear, h_ear = img_ear.size
            w_ear_resize = ew
            h_ear_resize = int(ew/w_ear*h_ear)
            img_ear_resize = img_ear.resize((w_ear_resize, h_ear_resize))
            img_ear_resize.save('./dog_parts_images/ear_resize.jpg')
            img_ear_resize = cv2.imread("./dog_parts_images/ear_resize.jpg")

            max_y = max(0, y+int(ey/2)-h_ear_resize)
            min_x = min(img.shape[1], x+ex+w_ear_resize+dev_ex)
            target_background = img[max_y:y+int(ey/2), x+ex+dev_ex:min_x]
            h1, w1, _ = target_background.shape
            img_ear_resize = img_ear_resize[0:h1, 0:w1]

            # masking and merge the dog ear image to the background image
            img_ear_resize_gray = cv2.cvtColor(img_ear_resize, cv2.COLOR_BGR2GRAY)
            img_maskg = cv2.threshold(img_ear_resize_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
            img_mask = cv2.merge((img_maskg,img_maskg, img_maskg))
            img_src2m = cv2.bitwise_and(img_ear_resize, img_mask)
            img_maskn = cv2.bitwise_not(img_mask)
            img_src1m = cv2.bitwise_and(target_background, img_maskn)
            img[max_y:y+int(ey/2), x+ex+dev_ex:min_x] = cv2.bitwise_or(img_src1m, img_src2m)

            count += 1

        # detect nose
        nose = nose_cascade.detectMultiScale(roi_gray)
        for (nx,ny,nw,nh) in nose:
            # load dog nose image
            img_nose= Image.open('./dog_parts_images/nose.jpg')

            # resize the dog nose image and extract the background image
            # to put it
            w_nose, h_nose = img_nose.size
            w_nose_resize = nw
            h_nose_resize = int(nh/w_nose*h_nose)
            img_nose_resize = img_nose.resize((w_nose_resize, h_nose_resize))
            img_nose_resize.save('./dog_parts_images/nose_resize.jpg')
            img_nose_resize = cv2.imread("./dog_parts_images/nose_resize.jpg")

            max_y = max(0, y+int(ny)-h_nose_resize)
            min_x = min(img.shape[1], x+nx+w_nose_resize)
            target_background = img[max_y+nh:y+int(ny)+nh, x+nx:min_x]
            h1, w1, _ = target_background.shape
            img_ear_resize = img_nose_resize[0:h1, 0:w1]

            # masking and merge the dog nose image to the background image
            img_nose_resize_gray = cv2.cvtColor(img_nose_resize, cv2.COLOR_BGR2GRAY)
            img_maskg = cv2.threshold(img_nose_resize_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
            img_mask = cv2.merge((img_maskg,img_maskg, img_maskg))
            img_src2m = cv2.bitwise_and(img_nose_resize, img_mask)
            img_maskn = cv2.bitwise_not(img_mask)
            img_src1m = cv2.bitwise_and(target_background, img_maskn)
            img[max_y+nh:y+int(ny)+nh, x+nx:min_x]= cv2.bitwise_or(img_src1m, img_src2m)

        # save the image with dog ears and nose
        #img_added_path = os.path.join(image_folder, secure_filename('transformed.jpg'))
        img_added_path = img_path[:-4]+"_transformed.jpg"
        cv2.imwrite(img_added_path, img)

        return img_added_path
