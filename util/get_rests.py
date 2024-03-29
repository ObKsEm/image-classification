import os
import cv2
import random

source_dir = '/home/lichengzhi/image-classification/voc/VOCdevkit/VOC2012/JPEGImages'
output_dir = '/home/lichengzhi/image-classification/data/classification/rests'


def main():
    num = 0
    rests_num = 10000
    for r, dirs, files in os.walk(source_dir):
        for file in files:
            if num < rests_num:
                img = cv2.imread(os.path.join(r, file))
                if img is not None:
                    num += 1
                    cv2.imwrite(os.path.join(output_dir, file), img)
    print("Get %d images.\n" % num)


if __name__ == "__main__":
    main()
