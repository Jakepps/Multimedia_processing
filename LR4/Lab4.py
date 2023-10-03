import cv2
import numpy as np


def svertka(img, kernel):
    kernel_size = len(kernel)

    x_start = kernel_size // 2
    y_start = kernel_size // 2

    matr = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    # проходим через внутренние пиксели изображения и выполняем операцию свертки между изображением и ядром.
    # Каждый пиксель изображения умножается на соответствующее значение в ядре, а затем суммируется
    for i in range(x_start, len(matr)-x_start):
        for j in range(y_start, len(matr[i])-y_start):

            # операция свёртки
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k][j + l] * kernel[k + (kernel_size//2)][l + (kernel_size//2)]
            matr[i][j] = val

    return matr


def get_angle_number(x, y):
    tg = y/x if x != 0 else 999

    if (x < 0):
        if (y < 0):
            if (tg > 2.414):
                return 0
            elif (tg < 0.414):
                return 6
            elif (tg <= 2.414):
                return 7
        else:
            if (tg < -2.414):
                return 4
            elif (tg < -0.414):
                return 5
            elif (tg >= -0.414):
                return 6
    else:
        if (y < 0):
            if (tg < -2.414):
                return 0
            elif (tg < -0.414):
                return 1
            elif (tg >= -0.414):
                return 2
        else:
            if (tg < 0.414):
                return 2
            elif (tg < 2.414):
                return 3
            elif (tg >= 2.414):
                return 4


i = 0

def main(path, standard_deviation, kernel_size, bound_path):
    global i
    i += 1

    # 1
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgBlurByCV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    cv2.imshow(path, imgBlurByCV2)

    # 2
    # Операторы собеля
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    img_Gx = svertka(img, Gx)
    img_Gy = svertka(img, Gy)

    #общий градиент
    matr_gradient = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_gradient[i][j] = img[i][j]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_gradient[i][j] = np.sqrt(img_Gx[i][j] ** 2 + img_Gy[i][j] ** 2)

    #углы градиента для каждого пикселя
    #каждый пиксель заменяется на значение угла градиента в соответствии с его градиентами по горизонтали и вертикали.
    img_angles = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_angles[i][j] = get_angle_number(img_Gx[i][j], img_Gy[i][j])

    # вывод матриц
    img_gradient_to_print = img.copy()
    max_gradient = np.max(matr_gradient)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #нормализация значения градиента относительно макс значения и масшабирования в диапозоне от 0 до 255
            img_gradient_to_print[i][j] = (float(matr_gradient[i][j]) / max_gradient) * 255
    cv2.imshow('img gradient to print ' + str(i), img_gradient_to_print)

    img_angles_to_print = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_angles_to_print[i][j] = img_angles[i][j] / 7 * 255
    cv2.imshow('img angles to print ' + str(i), img_angles_to_print)

    # 3
    img_border_no_filter = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            angle = img_angles[i][j]
            gradient = matr_gradient[i][j]
            #находится ли текущий пиксель на грани изображения(на границе матрицы)
            if (i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1):
                img_border_no_filter[i][j] = 0
            else:
                x_shift = 0
                y_shift = 0
                if (angle == 0 or angle == 4):
                    x_shift = 0
                elif (angle > 0 and angle < 4):
                    x_shift = 1
                else:
                    x_shift = -1

                if (angle == 2 or angle == 6):
                    y_shift = 0
                elif (angle > 2 and angle < 6):
                    y_shift = -1
                else:
                    y_shift = 1

                #явл ли текущий пиксель локльным максимумом градиента в направлении угла
                is_max = gradient >= matr_gradient[i + y_shift][j + x_shift] and gradient >= matr_gradient[i-y_shift][j-x_shift]
                img_border_no_filter[i][j] = 255 if is_max else 0

    cv2.imshow('img border no filter ' + str(i), img_border_no_filter)

    # 4
    #для понимания когда лок максимум градиента будут считаться границами
    lower_bound = max_gradient / bound_path
    upper_bound = max_gradient - max_gradient / bound_path
    img_border_filter = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradient = matr_gradient[i][j]
            #является лок максимумом
            if (img_border_no_filter[i][j] == 255):
                if (gradient >= lower_bound and gradient <= upper_bound):
                    flag = False
                    #проверим соседние пиксели текусщего пикселя
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (flag):
                                break
                            if (img_border_no_filter[i+k][j+l] == 255 and matr_gradient[i+k][j+l] >= lower_bound):
                                flag = True
                                break
                    if (flag):
                        img_border_filter[i][j] = 255
                elif (gradient > upper_bound):
                    img_border_filter[i][j] = 255

    cv2.imshow('img border filter ' + str(i), img_border_filter)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# 5
main('test.jpg', 10, 3, 15)
main('test.jpg', 10, 3, 6)
main('test.jpg', 100, 11, 15)
