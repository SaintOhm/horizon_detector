import os
import random
import cv2
import skimage.measure
import numpy as np
from math import atan2, cos, sin, pi, degrees, radians

def show_img(img, name: str):
    # Вывод изображения для отслеживания процесса и дебага
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, img)

def my_gray_blur(img, brush_size: int, func):
    # пройтись по изображению кистью brush_size функцией func
    # аналогичен skimage.measure.block_reduce, но сохраняет разрешение
    brush_size -= brush_size % 2

    if brush_size < 2:
        return img
    
    for x in range(0, 100, brush_size):
        for y in range(0, 100, brush_size):
            img[y:y+brush_size, x:x+brush_size] = func(img[y:y+brush_size, x:x+brush_size])
    return img

def crop_and_scale(img):
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
    
def find_pitch_and_roll(path: str):

    original_img = cv2.imread(path)

    # Для увеличения быстродействия необходимо обрезать изображение до квадрата и уменьшить разрешение
    processed_img = crop_and_scale(original_img)
    show_img(processed_img, 'Processed')

    # Переводим в оттенки серого. Подготовка к отделению неба и земли через бинаризацию по Оцу
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    show_img(gray_img, 'Gray')

    # Добавляем блюр для сглаживания шумов. Выбран билатеральный метод для сохранения границ объектов
    blur_img = cv2.bilateralFilter(gray_img, 9, 60, 50)
    show_img(blur_img, 'Blur')
    
    # Сама бинаризация.
    # Изображение содержит белые "пятна" или неровности в местах солнца, водоёмов и т.п.
    # и чёрные в местах облаков
    _, bw_img = cv2.threshold(blur_img, 250, 255, cv2.THRESH_OTSU)

    # Находим белые контуры
    # cv2.CHAIN_APPROX_SIMPLE не подходит, потому что при крене близком к 0, он будет возвращать мало точек,
    # которые могут быть потеряны при дальнейшей обработке
    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Если контуров нет, то скорее всего камера смотрит на землю или отключилась
    # Положение горизонта необходимо предсказать из прошлого положения и данных акселерометра
    if len(contours) == 0:
        print('Небо не найдено')
        #predict_horizon()
        return None, None

    # Предполагаем, что пятно с самым длинным контуром = небо
    sky = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)

    edge_points = []     # Лежат на краях изображения, поэтому точно не горизонт
    non_edge_points = [] # Лежат на краях водоёмов, полей, засветов, облаков и т.д.
    for i in sky:
        x, y = i[0][0], i[0][1]
        if x == 0 or x == 99 or y == 0 or y == 99:
            edge_points.append(i[0])
            bw_img[y][x] = (0, 255, 255)
        else:
            non_edge_points.append(i[0])
            bw_img[y][x] = (0, 0, 255)
    show_img(bw_img, 'B&W')


    # Горизонт находится на границе какого-то объекта. Помогает отфильтровать облака
    edges = cv2.Canny(image=gray_img, threshold1=200, threshold2=250)
    show_img(edges, 'Edges')
    # Находим чёткие границы по серому изображению
    # Проходим кистью 2х2. Ч/б картинка 50х50
    edges = skimage.measure.block_reduce(edges, (2, 2), np.max)
    show_img(edges, 'Gray edges skimage')

    show_img(
        skimage.measure.block_reduce(
            cv2.Canny(image=blur_img, threshold1=200, threshold2=250),
            (2, 2), np.max
        ),
        'Blur edges skimage'
    )
    
    blur_skimage = skimage.measure.block_reduce(blur_img, (2, 2), np.min)
    show_img(blur_skimage, 'Blur skimage')
    new_edges = cv2.resize(cv2.Canny(image=blur_skimage, threshold1=200, threshold2=250), (100, 100), interpolation=cv2.INTER_NEAREST)
    show_img(new_edges, 'Blur skimage edges')

    gray_skimage = skimage.measure.block_reduce(gray_img, (2, 2), np.min)
    show_img(gray_skimage, 'Gray skimage')
    gray_skimage_blur = cv2.bilateralFilter(gray_skimage, 5, 40, 50)
    show_img(gray_skimage_blur, 'Gray skimage blur')
    show_img(cv2.Canny(image=gray_skimage, threshold1=200, threshold2=250), 'Gray skimage blur edges')

    # Затемняет всё изображение, но тёмное сильнее, так что границы более явные
    new_gray_skimage = ((gray_skimage / 255)**2 * 255).astype(np.uint8)
    new_gray_skimage_blur = cv2.bilateralFilter(new_gray_skimage, 5, 50, 50)
    show_img(new_gray_skimage_blur, 'My gray skimage blur')
    show_img(cv2.Canny(image=new_gray_skimage_blur, threshold1=200, threshold2=250), 'My gray skimage blur edges')

    return path



os.system('cls')
for name in os.listdir('images'):
    print(find_pitch_and_roll(
        f'images\{name}'
    ))
    if cv2.waitKey(0) == 27: break # esc для выхода
    cv2.destroyAllWindows()
