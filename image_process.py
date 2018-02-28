import cv2


def resize(im,new_w,new_h,interpolation=cv2.INTER_CUBIC):
    im_resize = im.copy()
    im_resize = cv2.resize(im_resize, (new_w, new_h), interpolation=interpolation)
    return  im_resize
