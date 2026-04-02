import cv2
import numpy as np
from math import inf


# CONSTANTS
FOV = 60
LOWER_YELL = np.array([ 15,  50, 155])
UPPER_YELL = np.array([ 22, 205, 255])
LOWER_BLUE = np.array([ 90,   0, 100])
UPPER_BLUE = np.array([125, 255, 255])
LOWER_RED1 = np.array([  0,   0, 175])
UPPER_RED1 = np.array([ 30,  35, 255])
LOWER_RED2 = np.array([170,   0, 175])
UPPER_RED2 = np.array([179,  35, 255])

def HSVfilter(img:np.ndarray):
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yell = cv2.inRange(img, LOWER_YELL, UPPER_YELL)
    blue = cv2.inRange(img, LOWER_BLUE, UPPER_BLUE)
    red  = cv2.bitwise_or(
        cv2.inRange(img, LOWER_RED1, UPPER_RED1),
        cv2.inRange(img, LOWER_RED2, UPPER_RED2)
    )
    return cv2.bitwise_or(blue, red), yell

def showImg(img:np.ndarray, name:str=''):
    # Вывод изображения для отслеживания процесса и дебага
    w, h = img.shape[1::-1]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, int(300 * w/h), 300)
    cv2.imshow(name, img)

def ransacPolyfit(points:list, threshold:float=2.5):
    points = np.asarray(points)
    n_points = len(points)
    max_inliers = 4
    best_mask = None

    for _ in range(20):
        p1, p2 = points[np.random.choice(n_points, 2, replace=False)]
        if np.array_equal(p1, p2): continue 
        
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        # d = |(p2-p1) x (p1-p0)| / |p2-p1|
        distances = np.abs(np.cross(line_vec, p1 - points)) / line_len
        inliers_mask = distances < threshold
        n_inliers = np.sum(inliers_mask)
        
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_mask = inliers_mask

    k, b = None, None
    if best_mask is not None:
        try:
            inliers = points[best_mask]
            k, b = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
        except np.linalg.LinAlgError:
            # Линия горизонта вертикальная
            k, b = inf, np.average(inliers[:, 0])
    return k, b

def findPitchRoll(frame:np.ndarray, debug_mode:bool=False) -> tuple[None, None] | tuple[np.float64, np.float64]:
    pitch, roll = None, None
    w, h = frame.shape[1::-1]
    c_x, c_y = w//2, h//2

    # Сжимаем изображение, чтобы уменьшить число точек
    # Не обрезаем, чтобы сохранить изображение полностью
    resized = cv2.resize(frame, (100, 100))
    
    # Получаем маски неба/воды и полей (поля часто засвечивают)
    mask_blue, mask_fields = HSVfilter(resized)
    # Переводим изображение в оттенки серого с наложением маски,
    # чтобы высветлить небо и сделать его более явным
    blue_filtered_gray = cv2.add(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), mask_blue)
    # Делаем изображение бинарным
    _, bw_img = cv2.threshold(blue_filtered_gray, 250, 255, cv2.THRESH_OTSU)
    # В солнечный день поля засвечивают, поэтому закрываем их маской
    bw_img = cv2.bitwise_and(bw_img, cv2.bitwise_not(mask_fields))

    # Ищем контур неба
    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0: return None, None

    # Предполагаем, что самый большой контур = небо
    sky = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Сортируем точки
    edge_points = []     # Лежат на краях изображения, поэтому точно не горизонт
    non_edge_points = [] # Лежат на горизонте и краях водоёмов, засветов и т.д.
    for p in sky:
        x, y = p[0]
        if x == 0 or x == 99 or y == 0 or y == 99:
            edge_points.append(p[0])
        else:
            non_edge_points.append(p[0])

    non_edge_points = non_edge_points[::5]
    edge_points = edge_points[::10]

    # Точки на краях изображения используем для того, чтобы определить ориентацию
    avg = np.average(edge_points, axis=0).astype('int8') if edge_points else []
    if len(avg) == 0:
        # Нельзя понять, где небо
        return pitch, roll

    # Получаем уравнение прямой по точкам
    k, b = ransacPolyfit(non_edge_points)
    if k is None:
        # Горизонт не найден
        return pitch, roll
    elif k is inf:
        # Горизонт строго вертикальный
        b = b * w/100
        roll = 90 if b < avg[0] else -90
        pitch = (c_x - b) / h * FOV * np.sign(roll)
        return pitch, roll
    
    # Масштабируем коэффициенты под оригинал
    k, b = k * h/w, b * h/100
    
    is_sky_down = 0 if k*avg[0]+b > avg[1] else 1
    # Проверяем, куда смотрит нос. Подобрал через таблицу истинности
    is_nose_up = (c_y > k*c_x+b) ^ is_sky_down
    # Проекция центра изображения на горизонт
    proj = [c_x, c_y] - np.array([k, -1]) * (k*c_x-c_y+b) / (k**2 + 1)
    # Число пикселей между проекцией и центром
    dist_to_horizon = np.linalg.norm(proj - np.array([c_x, c_y]))
    # Определяем тангаж через FOV
    pitch = dist_to_horizon/h*FOV * ((-1)**is_nose_up)
    # Определяем крен с поправкой на то, где небо
    roll = -np.rad2deg(np.atan(k)) + 180 * is_sky_down * np.sign(k)
    
    if debug_mode:
        bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)

        bw_img[50][50] = (0, 255, 0)
        bw_img[avg[1]][avg[0]] = (0, 255, 0)
        for pos in edge_points:
            bw_img[pos[1]][pos[0]] = (255, 0, 0)
        for pos in non_edge_points:
            bw_img[pos[1]][pos[0]] = (0, 0, 255)
        
        cv2.line(frame, (0, int(b)), (w, int(k*(w-1)+b)), (0,0,255), 5)
        cv2.line(frame, (c_x, c_y), proj.astype('int16'), (0,0,255), 5)
        showImg(bw_img, 'Binary')
        showImg(frame, 'Original')
        #return pitch, roll, frame, bw_img
    return pitch, roll
