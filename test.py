import os

Base_path='/Users/zhengj/PycharmProjects/pythonProject/Image_analysis'
datapath='images/High Temp 3 cm 12th june'

def get_all_image_path(data_path):
    full_path = os.path.join(Base_path, data_path)
    name = os.listdir(full_path)
    print(len(name))
    # name.sort(key=lambda x: int(x.split('.')[0].split('_')[1]+x.split('.')[0].split('_')[2]))
    name.sort(key=lambda x: int(x.split('.')[0].strip('G')))

    full_name = []
    for i in range(len(name)):
        full_name.append(os.path.join(full_path,name[i]))

    return full_name[0],full_name


if __name__ == "__main__":
    print(get_all_image_path(datapath))
