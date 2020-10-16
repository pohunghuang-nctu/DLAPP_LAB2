#!/opt/conda/bin/python
import cv2
import sys
import numpy as np
import os


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    img_folder = os.path.join(input_path, 'img')
    anno_folder = os.path.join(input_path, 'anno')
    # check if img and annotation matched. 
    img_list = []
    anno_list = []
    for item in os.listdir(img_folder):
        item_path = os.path.join(img_folder, item)
        if os.path.isfile(item_path):
            anno_filename = os.path.splitext(item)[0] + '.txt'
            anno_path = os.path.join(anno_folder, anno_filename)
            if os.path.isfile(anno_path):
                img_list.append(item_path)
                anno_list.append(anno_path)
            else:
                print('Missing annotation for %s' % item)
    # print(img_list)
    # print(anno_list)

    # read jpg and crop accroding to txt file
    for i in range(0, len(img_list)):
        print('precessing img:%s' % img_list[i])
        img = cv2.imread(img_list[i])
        img_name, ext = os.path.splitext(os.path.basename(img_list[i]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('shape: [%d, %d]' % (gray.shape[0], gray.shape[1]))
        src_h = gray.shape[0]
        src_w = gray.shape[1]
        class_counter = {}
        with open(anno_list[i], 'r') as afile:
            for line in afile:
                line = line.strip()
                elements = line.split(' ')
                class_id = int(elements[0]) + 1
                crop_folder = os.path.join(
                    output_path, str(class_id))
                # if folder of class not exist, create it.
                if not os.path.isdir(crop_folder):
                    os.mkdir(crop_folder)
                c_x = float(elements[1]) * src_w
                c_y = float(elements[2]) * src_h
                w = float(elements[3]) * src_w
                h = float(elements[4]) * src_h
                LT_x = np.uint64(np.ceil(c_x - (w/2)))
                LT_y = np.uint64(np.ceil(c_y - (h/2)))
                RB_x = np.uint64(np.floor(c_x + (w/2)))
                RB_y = np.uint64(np.floor(c_y + (h/2)))
                print('(%d, %d) - (%d, %d)' % (LT_x, LT_y, RB_x, RB_y))
                crop = gray[LT_y:RB_y, LT_x:RB_x]
                crop_128 = cv2.resize(crop, (128, 128))
                if class_id not in class_counter:
                    class_counter[class_id] = 1
                else:
                    class_counter[class_id] += 1
                crop_name = '%s_%d_%d%s' % (img_name, class_id, class_counter[class_id], ext)
                crop_path = os.path.join(crop_folder, crop_name)
                cv2.imwrite(crop_path, crop_128)
                

if __name__ == '__main__':
    main()


