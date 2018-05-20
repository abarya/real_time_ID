import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np

import matplotlib.image as mpimg

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def crop_image(image,r=0.75):
	#ratio defines how much part of image needs to be cropped
	#4 images will be created from a single image
	h = image.shape[0]
	w = image.shape[1]
	images = []
	roi_cl = [[0,0],[0,w*(1-r)],[h*(1-r),0],[h*(1-r),w*(1-r)]]
	roi_cr = [[h*r,w*r],[h*r,w],[h,w*r],[h,w]]
	roi_cl = [(int(x[0]),int(x[1])) for x in roi_cl]
	roi_cr = [(int(x[0]),int(x[1])) for x in roi_cr]
	for i in range(4):
		images.append(image[roi_cl[i][0]:roi_cr[i][0],roi_cl[i][1]:roi_cr[i][1],:])
	return images


def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    # img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


def pre_processing(image):
    transformed_img_list = []
    img_list  = crop_image(image)
    for j in range(len(img_list)):
        for i in range(4):
            transformed_img_list.append(transform_image(img_list[j],0,10,5,brightness=1))
    return transformed_img_list

# image = mpimg.imread('/home/arya/Documents/real_time_ID/VIPeR/cam_a/000_45.bmp')
# plt.imshow(image);
# plt.axis('off');

# gs1 = gridspec.GridSpec(4,4)
# gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
# plt.figure(figsize=(4,4))

# img  = crop_image(image)

# for j in range(len(img)):
# 	for i in range(4):
# 	    ax1 = plt.subplot(gs1[i])
# 	    ax1.set_xticklabels([])
# 	    ax1.set_yticklabels([])
# 	    ax1.set_aspect('equal')
# 	    print img[j].shape
# 	    image = transform_image(img[j],0,10,5,brightness=1)
# 	    cv2.imshow("im",image)
# 	    cv2.waitKey(0)
# 	    plt.subplot(4,4,j*4+i)
# 	    plt.imshow(image)
# 	    plt.axis('off')

# plt.show()