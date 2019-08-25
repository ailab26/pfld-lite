import os
import cv2
import mxnet as mx
import numpy as np

def make_pfld_record(output=None, listName=None, imageFolder=None, dest_size=98):
    record = mx.recordio.MXRecordIO(output, 'w')
    File = open(listName, 'r')
    line = File.readline()
    idx =  0
    while line:
        idx += 1
        print(idx)
        info = line.split(' ')
        filename = info[0].split('/')[-1]
        image = cv2.imread(os.path.join(imageFolder, filename))
        image = cv2.resize(image, (dest_size, dest_size))
        lmks = []
        for i in range(0, 98):
            x = float(info[i*2 + 1]) 
            y = float(info[i*2 + 2])
            lmks.append(x)
            lmks.append(y)
        categories = []
        for i in range(0, 6):
            categories.append(
                float(info[1 + 98*2 + i])
            )
        angles = []
        for i in range(0, 3):
            angles.append(
                float(info[1 + 98*2 + 6 + i])
            )
        label  = lmks + categories + angles

        header = mx.recordio.IRHeader(0, label, i, 0)
        packed_s = mx.recordio.pack_img(header, image)
        record.write(packed_s)

        line = File.readline()

    if File is not None:
        File.close()
    record.close()

if __name__ == '__main__':
    train_record_name = './datas/pfld_train_data.rec'
    valid_record_name = './datas/pfld_valid_data.rec'

    train_file = './datas/train_data/list.txt'
    train_folder = './datas/train_data/imgs/'
    valid_file = './datas/test_data/list.txt'
    valid_folder = './datas/test_data/imgs/'
    
    image_size = 96
    make_pfld_record(train_record_name, train_file, train_folder, image_size)
    make_pfld_record(valid_record_name, valid_file, valid_folder, image_size)
