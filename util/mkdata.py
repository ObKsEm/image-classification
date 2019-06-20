import os
import cv2
import random

shop_dir = '/home/lichengzhi/image-classification/data/classification/shop'
shelf_dir = '/home/lichengzhi/image-classification/data/classification/shelf'
rests_dir = '/home/lichengzhi/image-classification/data/classification/rests'
train_dir = '/home/lichengzhi/image-classification/data/train'
valid_dir = '/home/lichengzhi/image-classification/data/valid'
test_dir = '/home/lichengzhi/image-classification/data/test'
test_ratio = 0.05


def main():

    shop_img = list()
    shelf_img = list()
    rests_img = list()

    for r, dirs, files in os.walk(shop_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                shop_img.append(img)
    for r, dirs, files in os.walk(shelf_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                shelf_img.append(img)
    for r, dirs, files in os.walk(rests_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                rests_img.append(img)
    train = 0
    test = 0
    valid = 0
    for img in shop_img:
        rd = random.random()
        if rd < test_ratio:
            test += 1
            cv2.imwrite(os.path.join(test_dir, "shop/%d.jpg" % test), img)
        elif test_ratio <= rd < 2 * test_ratio:
            valid += 1
            cv2.imwrite(os.path.join(valid_dir, "shop/%d.jpg" % valid), img)
        else:
            train += 1
            cv2.imwrite(os.path.join(train_dir, "shop/%d.jpg" % train), img)
    print("shop data: %d train, %d valid, %d test\n" % (train, valid, test))

    train = 0
    test = 0
    valid = 0
    for img in shelf_img:
        rd = random.random()
        if rd < test_ratio:
            test += 1
            cv2.imwrite(os.path.join(test_dir, "shelf/%d.jpg" % test), img)
        elif test_ratio <= rd < 2 * test_ratio:
            valid += 1
            cv2.imwrite(os.path.join(valid_dir, "shelf/%d.jpg" % valid), img)
        else:
            train += 1
            cv2.imwrite(os.path.join(train_dir, "shelf/%d.jpg" % train), img)
    print("shelf data: %d train, %d valid, %d test\n" % (train, valid, test))

    train = 0
    test = 0
    valid = 0
    for img in rests_img:
        rd = random.random()
        if rd < test_ratio:
            test += 1
            cv2.imwrite(os.path.join(test_dir, "rests/%d.jpg" % test), img)
        elif test_ratio <= rd < 2 * test_ratio:
            valid += 1
            cv2.imwrite(os.path.join(valid_dir, "rests/%d.jpg" % valid), img)
        else:
            train += 1
            cv2.imwrite(os.path.join(train_dir, "rests/%d.jpg" % train), img)
    print("rests data: %d train, %d valid, %d test\n" % (train, valid, test))


if __name__ == "__main__":
    main()
