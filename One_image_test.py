import cv2
import numpy as np
import os
from get_rec_coordinates import get_rectangles_xy

Base_path = 'C:\\Users\\ZHENGJ\\PycharmProjects\\Image_analysis'


def get_all_image_path(data_path):
    full_path = os.path.join(Base_path, data_path)
    name = os.listdir(full_path)
    #print(len(name))
    # name.sort(key=lambda x: int(x.split('.')[0].split('_')[1]+x.split('.')[0].split('_')[2]))
    name.sort(key=lambda x: int(x.split('.')[0].strip('G')))

    full_name = []
    for i in range(len(name)):
        full_name.append(os.path.join(full_path,name[i]))

    return full_name[0],full_name

class GetPixelsRef:
    def __init__(self,imagepath):
        self.h_coefficient =1
        self.height_for_green_tape = []
        self.image = cv2.imread(imagepath)

    def img_height_get(self,image):
        # resize image
        # print(image.shape)
        height, width, channel = image.shape

        try:
            image = cv2.resize(image, (int(1 * width), int(self.h_coefficient * height)), interpolation=cv2.INTER_CUBIC)
        except cv2.error:
            print("Please try again")
            exit()
        # cv2.namedWindow('original',cv2.WINDOW_NORMAL)
        # cv2.imshow("original", image)
        # cv2.waitKey()

        #extract green tape
        lower_green = np.array([35,45,46])
        upper_green = np.array([77,214,214])

        # change to hsv model
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask_r = cv2.inRange(hsv, lower_green, upper_green)
        mask_r = cv2.inRange(hsv, lower_green, upper_green)
        mask = mask_r
        mask=cv2.resize(mask, (int(5 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("green_part", mask)
        # cv2.waitKey()

        #image processing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 4))
        mask = cv2.erode(mask, kernel, 17)
        mask = cv2.dilate(mask, kernel)
        # mask = cv2.resize(mask, (int(5 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.GaussianBlur(mask, (11, 3), 0)
        mask = cv2.medianBlur(mask,5)
        # cv2.imshow("image_process", mask)
        # cv2.waitKey()

        #find coutours
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        #print("find", len(contours), "contours")
        #final=cv2.drawContours(image,contours,0,255,cv2.FILLED)
        # cv2.imshow("final",final)
        # cv2.waitKey()

        #get_height of one tube
        # get x,y
        if len(contours) > 1:
            h = []
            for i in range(len(contours)):
                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                h0 = round((1 / self.h_coefficient) * (abs(box[0][1] - box[2][1]) + abs(box[1][1] - box[3][1])) / 2, 3)
                h.append(round(h0,3))
            height = max(h)
        elif len(contours) == 0:
            print("this pipe is empty")
            height = 0
        else:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            height = round((1 / self.h_coefficient) * (abs(box[0][1] - box[2][1])+abs(box[1][1]-box[3][1]))/2,3)

        return height

    @staticmethod
    def rotate_image(imgpath):
        img = cv2.imread(imgpath)
        height,width=img.shape[:2]
        center=(width/2,height/2)
        rotate_matrix=cv2.getRotationMatrix2D(center=center,angle=-2,scale=1)
        rotated_image= cv2.warpAffine(img,M=rotate_matrix,dsize=(width,height))
        return rotated_image

    def get_pixel_num_per_cm(self):
        img = self.image

        rec_coordinates = get_rectangles_xy(img)
        height_base = []

        base_cropped_img = []
        for coordinates in rec_coordinates:
            base_cropped_img.append(img[coordinates[0][1]:coordinates[1][1],coordinates[0][0]:coordinates[1][0]])

        for i in range(len(base_cropped_img)):
            height_base.append(self.img_height_get(base_cropped_img[i]))

        revised_factor = 0.982
        pixels_per_cm = (0.9*max(height_base)/63.5+0.1*min(height_base)/30)*revised_factor

        print("for one cm, pixels are: ",pixels_per_cm)
        Justify = input("Please ensure the pixels per cm is correct(40.0-45.0), if good enter yes.\n")

        if Justify == "yes":
            pass
        else:
            exit("Please try again")

        return pixels_per_cm






