import numpy as np
import cv2


def change_hue_value(image, segmentation_mask, color=[0,0,255], alpha=.8, negative=False):
    image_color = np.copy(image)
    if not negative:
        image_color[(segmentation_mask==1.0)] = [0,0,255]
    else:
        image_color[~(segmentation_mask==1.0)] = [0,0,255]
    image_color_w = cv2.addWeighted(image_color, 1-alpha, image, alpha, 0, image_color)
        
    return image_color_w.astype(np.uint8)
    

def change_brightness_value(image, mask,  value=30, negative=False):
    ## value only can assume < 255
    img = np.copy(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    if not negative:
        v[(v > lim) & (mask==1.0)] = 255
        v[(v <= lim) & (mask==1.0)] += value
    else:
        v[(v > lim) & (mask==0.0)] = 255
        v[(v <= lim) & (mask==0.0)] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img.astype(np.uint8)

def change_contrast_value(img, mask, alpha=1.5, beta=0, negative=False):
    # Contrast control (1.0-3.0)
    0 # Brightness control (0-100)
    image = np.copy(img)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if not negative:
        image = image + (adjusted * mask.reshape(mask.shape[0],mask.shape[1] ,1))
    else:
        mask = mask < 1
        image = image + (adjusted * mask.reshape(mask.shape[0],mask.shape[1] ,1))
    
    return image.astype(np.uint8)


def change_texture(img, texture, mask, alpha=0.8, negative=False):
    result_image = np.copy(img)
    texture_resized = cv2.resize(texture, (result_image.shape[1], result_image.shape[0]))
    if not negative:
        result_image = result_image * (1 - mask.reshape(mask.shape[0],mask.shape[1] ,1)) + (texture_resized * mask.reshape(mask.shape[0],mask.shape[1] ,1))
    else:
        mask = mask < 1
        result_image = result_image * (1 - mask.reshape(mask.shape[0],mask.shape[1] ,1)) + (texture_resized * mask.reshape(mask.shape[0],mask.shape[1] ,1))

    result_image = cv2.addWeighted(result_image.astype(np.uint8), 1-alpha, img, alpha, 0, result_image.astype(np.uint8))
    return result_image.astype(np.uint8)

def add_noise(img, mask, scale=80, negative=False):
    image = np.copy(img)
    noise = np.random.normal(loc=0, scale=scale, size=image.shape)
    
    if not negative:
        image[mask==1] += noise[mask==1].astype('uint8')
    else:
        image[mask==0] += noise[mask==0].astype('uint8')
    return image.astype('uint8')

def crop_mask(img, mask, negative=False, region=True):
    box_image = np.copy(img)
    if region:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

        if not negative:
            exp_mask = np.zeros_like(img[:, :, 0])
            exp_mask[y:y+h, x:x+w] = 1
            print(exp_mask.shape)
        else:
            exp_mask = np.ones_like(img[:, :, 0])
            exp_mask[y:y+h, x:x+w] = 0

        box_image = img * exp_mask[:, :, np.newaxis]
    else:
        if not negative:
            box_image[~(mask==1.0)] = [0,0,0]
        else:
            box_image[(mask==1.0)] = [0,0,0]
        
    
    return box_image