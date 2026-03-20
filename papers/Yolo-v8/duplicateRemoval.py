import os

rootimgs = '/nvme1/liuyundong/xianyinggunYoloV8/datasets/labels/train20230304/'
rootxmls = '/nvme1/liuyundong/xianyinggunYoloV8/huashangdatasets/images'
allusedxmls = []
file_imgs = os.listdir(rootimgs)
file_xmls = os.listdir(rootxmls)
for file_name in file_xmls:
    file_name = file_name[:-4] + '.txt'
    # print(file_name)
    allusedxmls.append(file_name)

for file_name in allusedxmls:
    path = rootimgs + file_name
    try:
        os.remove(path)
    except:
        print(path,'没有这个')

    print(file_name)
    # if file_name not in allusedxmls:
    #     path = rootimgs + file_name
    #     os.remove(path)