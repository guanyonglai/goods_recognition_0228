import os
import cv2

# #################
# plot
# #################
colors = ['r','g','b','c','k','w']
def plot_bb_on_img(im,bb,bb_cls,thickness=2):
    h, w, c = im.shape

    for idx, box in enumerate(bb):

        b_w = box[2] * w
        b_h = box[3] * h
        c_x = box[0] * w
        c_y = box[1] * h

        x1 = int(max([0, (c_x - 0.5 * b_w)]))
        x2 = int(min([w, (c_x + 0.5 * b_w)]))
        y1 = int(max([0, (c_y - 0.5 * b_h)]))
        y2 = int(min([h, (c_y + 0.5 * b_h)]))

        # choose a color
        color = colors[bb_cls[idx]]

        # get class real name
        name = 'cls'

        # plot
        cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=thickness)
        cv2.putText(im, '%s' % name, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness=thickness)

    return im