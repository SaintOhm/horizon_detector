import os
import random
import cv2
import skimage.measure
import numpy as np
from math import atan2, cos, sin, pi, degrees, radians


def show_img(img: np.array, name: str = 'N/A'):
    # Вывод изображения для отслеживания процесса и дебага
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, img)

def crop_and_scale(img: np.array):
    # Сначала обрезка изображения до квадрата
    width, height = img.shape[1::-1]
    if width > height:
        margin = (width - height)/2
        img = img[:, int(margin):int(width - margin)]
    else:
        margin = (height - width)/2
        img = img[int(margin):int(height - margin), :]
    
    # Теперь уменьшение
    img = cv2.resize(img, (100, 100))
    
    return img


LOWER = np.array([ 90,   0,  50])
UPPER = np.array([125, 255, 255])

LOWER_BLUE = np.array([ 90,   0,  50])
UPPER_BLUE = np.array([125, 255, 255])
LOWER_RED1 = np.array([  0,   0, 175])
UPPER_RED1 = np.array([ 30,  35, 255])
LOWER_RED2 = np.array([170,   0, 175])
UPPER_RED2 = np.array([179,  35, 255])

def HSV_filter(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(img, LOWER_BLUE, UPPER_BLUE)
    red  = cv2.bitwise_or(
        cv2.inRange(img, LOWER_RED1, UPPER_RED1),
        cv2.inRange(img, LOWER_RED2, UPPER_RED2)
    )
    return cv2.bitwise_or(blue, red)

#def find_pitch_and_roll(frame: np.array, debug_mode: bool = False):
def find_pitch_and_roll(path: str, debug_mode: bool = False):

    frame = crop_and_scale(cv2.imread(f'images\{path}'))
    #frame = crop_and_scale(frame)
    bgr2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue_filtered_gray = cv2.add(
        bgr2gray,
        cv2.inRange(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
            LOWER, UPPER
        )
    )

    # Отфильтровать синее небо
    #lower = np.array([100,   0,  50])
    #upper = np.array([125, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, LOWER, UPPER)

    blur_img = cv2.bilateralFilter(blue_filtered_gray, 9, 60, 50)
    _, bw_img = cv2.threshold(blue_filtered_gray, 250, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(image=blur_img, threshold1=200, threshold2=250)

    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)     
    sky = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
    for i in sky:
        x, y = i[0][0], i[0][1]
        if x == 0 or x == 99 or y == 0 or y == 99:
            bw_img[y][x] = (0, 255, 255)
        else:
            bw_img[y][x] = (0, 0, 255)
    
    show_img(frame, 'Original')
    show_img(bgr2gray, 'Gray')
    show_img(hsv, 'HSV')
    show_img(hsv_mask, 'HSV mask')
    show_img(blue_filtered_gray, 'Blue filter gray')
    show_img(blur_img, 'Blur')
    show_img(bw_img, 'B&W')
    show_img(edges, 'Edges')
    #show_img(HSV_filter(frame), 'Filter')
    cv2.imwrite(f'debug_images\{path}', cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(HSV_filter(frame))))
    return ''

os.system('cls')
for name in os.listdir('images'):
    print(name)
    print(find_pitch_and_roll(
        #cv2.imread(f'images\{name}')
        name
    ))
    if cv2.waitKey(0) == 27: break # esc для выхода
    cv2.destroyAllWindows()