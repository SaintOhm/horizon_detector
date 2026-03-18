import os
import random
import cv2
import numpy as np



def show_img(img, name: str):
    # Вывод изображения для отслеживания процесса и дебага
    cv2.namedWindow(f'{name} image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(f'{name} image', img)

def crop_and_scale(frame):
    # Сначала обрезка изображения до квадрата
    width, height = frame.shape[1::-1]
    if width > height:
        margin = (width - height)/2
        frame = frame[:, int(margin):int(width - margin)]
    else:
        margin = (height - width)/2
        frame = frame[int(margin):int(height - margin), :]
    
    # Теперь уменьшение
    frame = cv2.resize(frame, (100, 100))
    
    return frame
    
def find_pitch_and_roll(path: str):

    original_img = cv2.imread(path)
    show_img(original_img, 'Original')

    # Для увеличения быстродействия необходимо обрезать изображение до квадрата и уменьшить разрешение
    processed_img = crop_and_scale(original_img)
    show_img(processed_img, 'Processed')

    # Переводим в оттенки серого. Подготовка к отделению неба и земли через бинаризацию по Оцу
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    show_img(gray_img, 'Gray')

    # Добавляем блюр для сглаживания шумов. Выбран билатеральный метод для сохранения границ объектов
    blur_img = cv2.bilateralFilter(gray_img, 9, 50, 50)
    show_img(blur_img, 'Blur')

    # Сама бинаризация.
    # Изображение содержит белые "пятна" или неровности в местах солнца, водоёмов и т.п.
    # и чёрные в местах облаков
    _, bw_img = cv2.threshold(blur_img, 250, 255, cv2.THRESH_OTSU)
    show_img(bw_img, 'B&W')

    # Находим белые контуры
    # cv2.CHAIN_APPROX_SIMPLE не подходит, потому что при крене близком к 0, он будет возвращать мало точек,
    # которые могут быть потеряны при дальнейшей обработке
    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Если контуров нет, то скорее всего камера смотрит на землю или отключилась
    # Положение горизонта необходимо предсказать из прошлого положения и данных акселерометра
    if len(contours) == 0:
        print('Небо не найдено. Предсказать положение по прошлым данным и ускорениям')
        #predict_horizon()
        return None, None

    # Предполагаем, что пятно с самым длинным контуром = небо
    sky = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    return 'Work in progress'



os.system('cls')
print(find_pitch_and_roll(
    f'images\{random.choice(os.listdir('images'))}'
))

while cv2.waitKey(10) != 27: # esc
    pass
cv2.destroyAllWindows()