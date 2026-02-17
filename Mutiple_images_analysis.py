import copy
import cv2
import numpy as np
import pandas as pd
import os
from One_image_test import get_all_image_path, GetPixelsRef
from tqdm import tqdm

# parameters for each image
h_coefficient = 1
height_all_for_all = []
height_all_for_one = []
height_full = []
height_tape = 0
counts = 0
dot = []
rotated_dot = []
flag = 1
revise_signal = False
revise_factor_for_camera_position = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

WIN_NAME = 'draw_rect'


class Rect(object):
    def __init__(self):
        self.tl = (0, 0)
        self.br = (0, 0)

    def regularize(self):
        """
        make sure tl = TopLeft point, br = BottomRight point
        """
        pt1 = (min(self.tl[0], self.br[0]), min(self.tl[1], self.br[1]))
        pt2 = (max(self.tl[0], self.br[0]), max(self.tl[1], self.br[1]))
        self.tl = pt1
        self.br = pt2


class DrawRects(object):
    def __init__(self, image, color, thickness=1):
        self.original_image = image
        self.image_for_show = image.copy()
        self.color = color
        self.thickness = thickness
        self.rects = []
        self.current_rect = Rect()
        self.left_button_down = False

    @staticmethod
    def __clip(value, low, high):
        """
        clip value between low and high

        Parameters
        ----------
        value: a number
            value to be clipped
        low: a number
            low limit
        high: a number
            high limit

        Returns
        -------
        output: a number
            clipped value
        """
        output = max(value, low)
        output = min(output, high)
        return output

    def shrink_point(self, x, y):
        """
        shrink point (x, y) to inside image_for_show

        Parameters
        ----------
        x, y: int, int
            coordinate of a point

        Returns
        -------
        x_shrink, y_shrink: int, int
            shrinked coordinate
        """
        height, width = self.image_for_show.shape[0:2]
        x_shrink = self.__clip(x, 0, width)
        y_shrink = self.__clip(y, 0, height)
        return (x_shrink, y_shrink)

    def append(self):
        """
        add a rect to rects list
        """
        self.rects.append(copy.deepcopy(self.current_rect))

    def pop(self):
        """
        pop a rect from rects list
        """
        rect = Rect()
        if self.rects:
            rect = self.rects.pop()
        return rect

    def reset_image(self):
        """
        reset image_for_show using original image
        """
        self.image_for_show = self.original_image.copy()

    def draw(self):
        """
        draw rects on image_for_show
        """
        for rect in self.rects:
            cv2.rectangle(self.image_for_show, rect.tl, rect.br,
                          color=self.color, thickness=self.thickness)

    def draw_current_rect(self):
        """
        draw current rect on image_for_show
        """
        cv2.rectangle(self.image_for_show,
                      self.current_rect.tl, self.current_rect.br,
                      color=self.color, thickness=self.thickness)


def onmouse_draw_rect(event, x, y, flags, draw_rects):
    if event == cv2.EVENT_LBUTTONDOWN:
        # pick first point of rect
        print('pt1: x = %d, y = %d' % (x, y))
        draw_rects.left_button_down = True
        draw_rects.current_rect.tl = (x, y)
        if flag:
            dot.append([x, y])
        else:
            rotated_dot.append([x, y])
    if draw_rects.left_button_down and event == cv2.EVENT_MOUSEMOVE:
        # pick second point of rect and draw current rect
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        draw_rects.reset_image()
        draw_rects.draw()
        draw_rects.draw_current_rect()
    if event == cv2.EVENT_LBUTTONUP:
        # finish drawing current rect and append it to rects list
        draw_rects.left_button_down = False
        draw_rects.current_rect.br = draw_rects.shrink_point(x, y)
        print('pt2: x = %d, y = %d' % (draw_rects.current_rect.br[0],
                                       draw_rects.current_rect.br[1]))
        if flag:
            dot.append([draw_rects.current_rect.br[0], draw_rects.current_rect.br[1]])
        else:
            rotated_dot.append([draw_rects.current_rect.br[0], draw_rects.current_rect.br[1]])
        draw_rects.current_rect.regularize()
        draw_rects.append()
    if (not draw_rects.left_button_down) and event == cv2.EVENT_RBUTTONDOWN:
        # pop the last rect in rects list
        draw_rects.pop()
        draw_rects.reset_image()
        draw_rects.draw()


def get_xy_form_one_image(Image):
    """ get the center if image need to rotate,
        parameter:  angle to change
    """

    height, width = Image.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1)
    rotated_image = cv2.warpAffine(Image, M=rotate_matrix, dsize=(width, height))

    draw_rects_r = DrawRects(rotated_image, (0, 255, 0), 2)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects_r)
    while True:
        cv2.imshow(WIN_NAME, draw_rects_r.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


def drawMyContours(winName, image, contours, draw_on_blank):
    # cv2.drawContours(image, contours, index, color, line_width)
    # 输入参数：
    # image:与原始图像大小相同的画布图像（也可以为原始图像）
    # contours：轮廓（python列表）
    # index：轮廓的索引（当设置为-1时，绘制所有轮廓）
    # color：线条颜色，
    # line_width：线条粗细
    # 返回绘制了轮廓的图像image
    if draw_on_blank:  # 在白底上绘制轮廓
        temp = np.ones(image.shape, dtype=np.uint8) * 255
        cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
    else:
        temp = image.copy()
        cv2.drawContours(temp, contours, -1, (0, 0, 255), 2)
    # cv2.imshow(winName, temp)
    # cv2.waitKey()


def img_height_get(image):
    # resize image
    # print(image.shape)
    height, width, channel = image.shape
    try:
        image = cv2.resize(image, (int(1 * width), int(h_coefficient * height)), interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        print("Please try again")
        exit()
    # cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    # # cv2.imshow("original", image)
    # # cv2.waitKey()

    # extract red part
    lower_red = np.array([160, 60, 60])
    upper_red = np.array([180, 255, 255])

    lower_red2 = np.array([0, 60, 60])
    upper_red2 = np.array([10, 255, 255])  # there is two ranges of red

    # change to hsv model
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_r = cv2.inRange(hsv, lower_red, upper_red)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask_r + mask_r2
    mask = cv2.resize(mask, (int(3 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("red_part", mask)
    # cv2.waitKey()

    # image processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    mask = cv2.erode(mask, kernel, 18)
    mask = cv2.dilate(mask, kernel)
    # mask = cv2.resize(mask, (int(5 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (11, 3), 0)
    mask = cv2.medianBlur(mask, 5)
    # cv2.imshow("image_process", mask)
    # cv2.waitKey()

    # find coutours
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # print("find", len(contours), "contours")
    final = cv2.drawContours(image, contours, 0, 255, cv2.FILLED)
    # cv2.imshow("final",final)
    # cv2.waitKey()

    # draw contour
    drawMyContours("countours", image, contours, True)

    # get_height of one tube
    # get x,y
    print("\n------------for single tube------------")
    global counts,revise_signal
    if len(contours) > 1:
        h = []
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            h0 = (1 / h_coefficient) * (abs(box[0][1] - box[2][1]) + abs(box[1][1] - box[3][1])) / 2
            # h.append(round(h0/height_full[counts],3))
            h.append(round(h0, 3))
            # print(box)
        print(round(max(h) / height_tape, 4))
        if revise_signal:
            if 65<max(h) / height_tape<70:
                revise_factor_for_camera_position[counts] = 67/int(max(h) / height_tape)
        if max(h) / height_tape > 2.5:
            height_all_for_one.append((max(h) / height_tape)*revise_factor_for_camera_position[counts])
        else:
            height_all_for_one.append("0")

    elif len(contours) == 0:
        print("this pipe is empty")
        height_all_for_one.append("0")
    else:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)
        h = round((1 / h_coefficient) * (abs(box[0][1] - box[2][1]) + abs(box[1][1] - box[3][1])) / 2, 3)
        print(round(h / height_tape, 4))
        if revise_signal:
            if 65<h / height_tape<70:
                revise_factor_for_camera_position[counts] = 67/int(h / height_tape)
        if h / height_tape > 2.5:
            height_all_for_one.append((h / height_tape)*revise_factor_for_camera_position[counts])
        else:
            height_all_for_one.append("0")

    counts += 1
    counts %= 11
    if counts == 10:
        revise_signal = False


def rotate_image(imgpath):
    img = cv2.imread(imgpath)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1)
    rotated_image = cv2.warpAffine(img, M=rotate_matrix, dsize=(width, height))
    return rotated_image


def one_image_processing(filename):
    img = cv2.imread(filename)
    rotated_image = rotate_image(filename)
    base_cropped_img = []
    base_rotated_cropped_img = []
    height_base = []
    height_base_rotated = []

    for i in range(int(len(dot) / 2)):
        base_cropped_img.append(img[dot[2 * i][1]:dot[2 * i + 1][1], dot[2 * i][0]:dot[2 * i + 1][0]])
        height_base.append(dot[2 * i + 1][1] - dot[2 * i][1])

    for i in range(int(len(rotated_dot) / 2)):
        base_rotated_cropped_img.append(rotated_image[rotated_dot[2 * i][1]:rotated_dot[2 * i + 1][1],
                                        rotated_dot[2 * i][0]:rotated_dot[2 * i + 1][0]])
        height_base_rotated.append(rotated_dot[2 * i + 1][1] - rotated_dot[2 * i][1])

    # store 11 tubes coordinates
    cropped_image = base_cropped_img + base_rotated_cropped_img

    global height_full
    height_full = height_base + height_base_rotated

    for i in range(len(cropped_image)):
        img_height_get(cropped_image[i])

    global height_all_for_one
    print("------------For one image:---------------")
    print("The result is :", height_all_for_one)
    height_all_for_all.append(height_all_for_one)
    height_all_for_one = []


def to_csv(store_path):
    global height_all_for_all
    print(store_path)
    df = pd.DataFrame(height_all_for_all)
    df.to_csv(store_path, mode="a")
    height_all_for_all = []


def data_to_csv(data_path, storepath):
    first_file_name, full_name = get_all_image_path(data_path)
    # print(first_file_name)
    global height_tape
    height_tape = GetPixelsRef(first_file_name).get_pixel_num_per_cm()

    base_image = cv2.imread(first_file_name)
    rotated_base_image = rotate_image(first_file_name)
    get_xy_form_one_image(base_image)
    global flag
    flag = 0
    get_xy_form_one_image(rotated_base_image)

    for i in tqdm(range(len(full_name))):
        one_image_processing(full_name[i])

    to_csv(storepath)


# the image file name need to be change
image_file_name = "5"

Base_path = 'C:\\Users\\ZHENGJ\\PycharmProjects\\Image_analysis'
datapath = os.path.join("images", image_file_name)
Store_path = os.path.join(Base_path, "results/1heights_%s.csv" % datapath.split("\\")[-1])

if __name__ == "__main__":
    # height,width=base_image.shape[:2]
    # center=(width/2,height/2)
    # rotate_matrix=cv2.getRotationMatrix2D(center=center,angle=-2,scale=1)
    # base_image= cv2.warpAffine(base_image,M=rotate_matrix,dsize=(width,height))

    data_to_csv(datapath, Store_path)
