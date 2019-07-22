import os

import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

img_transforms = transforms.Compose([
        # transforms.CenterCrop((224, 224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class_to_idx = dict({'0': 0, '180': 1, '270': 2, '90': 3})
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))


def rotate(img, angel):
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if angel == 1:
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
    elif angel == 2:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    elif angel == 3:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)

    return image


def main():
    source_dir = "/home/lichengzhi/image-classification/data/rotations/demo"
    target_dir = "/home/lichengzhi/image-classification/data/rotations/adjustment"
    model_dir = "/home/lichengzhi/image-classification/work_dir/rotation/classifier11.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torchvision.models as models
    model = models.resnet101(pretrained=False).to(device)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=fc_features, out_features=4).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    for r, dirs, files in os.walk(source_dir):
        for file in files:
            print("Image: %s" % file)
            img_path = os.path.join(r, file)
            img = Image.open(img_path)
            if img is not None:
                if img.mode is not "RGB":
                    img = img.convert("RGB")
                image_tensor = img_transforms(img).float()
                image_tensor = image_tensor.unsqueeze_(0)
                image_tensor.to(device)
                input = Variable(image_tensor).to(device)
                with torch.no_grad():
                    output = model(input)
                softmax = F.softmax(output).cpu().numpy()[0]
                print(softmax)
                result = output.data.cpu().numpy().argmax()
                print(idx_to_class[result])
                ret_img = rotate(img, result)
                cv2.imwrite(os.path.join(target_dir, file), ret_img)
            else:
                print("Invalid image :%s" % file)


if __name__ == "__main__":
    main()
