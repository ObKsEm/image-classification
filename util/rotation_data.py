import os
import cv2
import random

train_dir = '/home/lichengzhi/image-classification/data/rotations/train'
valid_dir = '/home/lichengzhi/image-classification/data/rotations/valid'
test_dir = '/home/lichengzhi/image-classification/data/rotations/test'
base_dir = '/home/lichengzhi/image-classification/data/rotations/augmentation'
test_ratio = 0.05


def image_rotate(img, rotation):
    ret = img
    if rotation == 90:
        ret = cv2.transpose(ret)
        ret = cv2.flip(ret, 1)
    elif rotation == 180:
        ret = cv2.flip(ret, 0)
        ret = cv2.flip(ret, 1)
    elif rotation == 270:
        ret = cv2.transpose(ret)
        ret = cv2.flip(ret, 0)
    return ret


def main():
    for r, dirs, files in os.walk(base_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                for i in range(0, 4):
                    rotate = i * 90
                    rd = random.random()
                    if rd < test_ratio:
                        cv2.imwrite(os.path.join(test_dir, "%d/%s" % (rotate, file)), image_rotate(img, rotate))
                    elif test_ratio <= rd < 2 * test_ratio:
                        cv2.imwrite(os.path.join(valid_dir, "%d/%s" % (rotate, file)), image_rotate(img, rotate))
                    else:
                        cv2.imwrite(os.path.join(train_dir, "%d/%s" % (rotate, file)), image_rotate(img, rotate))
                    # print("%d data: %d train, %d valid, %d test\n" % (rotate, train, valid, test))


if __name__ == "__main__":
    main()
