import os
import cv2
import numpy as np
#from math import atan2, cos, sin, pi, degrees, radians

# CONSTANTS
#https://www.rapidtables.com/convert/color/rgb-to-hsv.html
#https://www.selecolor.com/en/hsv-color-picker/
LOWER_YELL = np.array([ 15,  50, 155])
UPPER_YELL = np.array([ 22, 205, 255])
LOWER_BLUE = np.array([ 90,   0, 100])
UPPER_BLUE = np.array([125, 255, 255])
LOWER_RED1 = np.array([  0,   0, 175])
UPPER_RED1 = np.array([ 30,  35, 255])
LOWER_RED2 = np.array([170,   0, 175])
UPPER_RED2 = np.array([179,  35, 255])


def showImg(img: np.array, name: str = 'N/A'):
    # Вывод изображения для отслеживания процесса и дебага
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, img)

def cropNscale(img: np.array):
    # Сначала обрезка изображения до квадрата
    width, height = img.shape[1::-1]
    if width > height:
        margin = (width - height)/2
        img = img[:, int(margin):int(width - margin)]
    else:
        margin = (height - width)/2
        img = img[int(margin):int(height - margin), :]
    
    # Теперь уменьшение
    return cv2.resize(img, (100, 100))

def fitlineRANSAC(points: list):
    points = np.array(points)
    n_points = len(points)
    best_model = (0, 0) # k, b in y=kx+b
    best_inliers_count = 0
    for _ in range(50):
        p1, p2 = points[np.random.choice(n_points, 2, replace=False)]
        if p1[0] == p2[0]: continue
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - k[p1[0]]
        n_inliers = 0
        if n_inliers > best_inliers_count:
            best_inliers_count = n_inliers

    return best_model

def HSVfilter(img: np.array):
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yell = cv2.inRange(img, LOWER_YELL, UPPER_YELL)
    blue = cv2.inRange(img, LOWER_BLUE, UPPER_BLUE)
    red  = cv2.bitwise_or(
        cv2.inRange(img, LOWER_RED1, UPPER_RED1),
        cv2.inRange(img, LOWER_RED2, UPPER_RED2)
    )
    return cv2.bitwise_or(blue, red), cv2.bitwise_not(yell)

#def find_pitch_and_roll(frame: np.array, debug_mode: bool = False):
def find_pitch_and_roll(path: str, debug_mode: bool = False):

    frame = cropNscale(cv2.imread(f'images\{path}'))
    #frame = crop_and_scale(frame)
    mask_blue, mask_fields = HSVfilter(frame)
    blue_filtered_gray = cv2.add(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask_blue)

    blur_img = cv2.bilateralFilter(blue_filtered_gray, 9, 60, 50)
    _, bw_img = cv2.threshold(blue_filtered_gray, 250, 255, cv2.THRESH_OTSU)
    bw_img = cv2.bitwise_and(bw_img, mask_fields)
    dilated_edges = cv2.dilate(
        cv2.Canny(image=blur_img, threshold1=200, threshold2=250),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)     
    sky = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    edge_points = []     # Лежат на краях изображения, поэтому точно не горизонт
    non_edge_points = [] # Лежат на краях водоёмов, засветов и т.д.
    for p in sky:
        x, y = p[0]
        if x == 0 or x == 99 or y == 0 or y == 99:
            edge_points.append(p[0])
        else:
            non_edge_points.append(p[0])

    avg = np.average(edge_points, axis=0).astype('uint8') if edge_points else []
    
    non_edge_points = [p for p in non_edge_points if dilated_edges[p[1]][p[0]]]
    
    if debug_mode:
        bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)
        if len(avg) > 0:
            bw_img[avg[1]][avg[0]] = (0, 255, 255)
        for pos in edge_points:
            bw_img[pos[1]][pos[0]] = (0, 255, 255)
        for pos in non_edge_points:
            bw_img[pos[1]][pos[0]] = (0,   0, 255)
        
        showImg(frame, 'Original')
        showImg(cv2.bitwise_or(mask_blue, cv2.bitwise_not(mask_fields)), 'Mask')
        showImg(blue_filtered_gray, 'Blue filter gray')
        showImg(blur_img, 'Blur')
        showImg(bw_img, 'B&W')
        showImg(dilated_edges, 'Edges')
        cv2.imwrite(f'debug_images\{path}', cv2.bitwise_and(frame, frame, mask=mask_fields))
    return ''

os.system('cls')
for name in os.listdir('images'):
    print(name)
    print(find_pitch_and_roll(
        #cv2.imread(f'images\{name}'),
        name,
        debug_mode = True
    ))
    if cv2.waitKey(0) == 27: break # esc для выхода
    cv2.destroyAllWindows()
