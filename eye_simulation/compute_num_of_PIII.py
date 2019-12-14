import numpy as np
import cv2
import os

def p3_in_image(index, root):
    img123 = cv2.imread(os.path.join(root, 'shot_%05d_noPIV.png' % (index)), cv2.IMREAD_GRAYSCALE)
    img12 = cv2.imread(os.path.join(root, 'shot_%05d_noPIIIPIV.png' % (index)), cv2.IMREAD_GRAYSCALE)

    # diff operation
    diff = cv2.absdiff(img123, img12)
    # get color < 20 parts
    diff = 1 - (diff > 150) * 1.0

    # opening
    kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    num_pixels = np.sum(diff < 0.5)
    if num_pixels > 20:
        return True
    else:
        return False

    
def main():
    root = "/playpen/connylu/eye_simulation/"
    num_vert, num_hori = 6, 6 # each led has 36 images

    num_images = len([name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name))]) // 2
    num_leds = num_images // (num_vert * num_hori)
    num_p3 = np.zeros(num_leds)
    for i in range(num_images):
        if p3_in_image(i, root):
            num_p3[i // (num_vert*num_hori)] += 1

    print(num_p3)
    print("Max: %d-th led shows %d 3rd purkinje images." % (np.argmax(num_p3), max(num_p3)))

main()
