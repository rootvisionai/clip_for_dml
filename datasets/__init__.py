import torchvision
from PIL import Image


# import dataset
def img_load(path):
    im = Image.open(path)
    if len(list(im.split())) == 1:
        im = im.convert('RGB')
    return im

class Data(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        record = self.samples[index]
        img_path = record[0]
        image = img_load(img_path)
        # label_str = self.classes[record[1]]
        label_int = record[1]
        if self.transform is not None:
            image = self.transform(image)
        return image, label_int

    def nb_classes(self):
        return len(self.classes)

def get_transform(cfg, train=False):
    if train:
        TransformTrain = torchvision.transforms.Compose([
            torchvision.transforms.Resize((cfg.preprocessing.image_size, cfg.preprocessing.image_size)),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(hue=0.1, brightness=0.3, saturation=0.2),
                torchvision.transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 5))
            ], p=cfg.preprocessing.color_and_blur),
            torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=cfg.preprocessing.perspective),
            torchvision.transforms.ToTensor()
        ])
        return TransformTrain
    else:
        TransformEval = torchvision.transforms.Compose([
            torchvision.transforms.Resize((cfg.preprocessing.image_size, cfg.preprocessing.image_size)),
            torchvision.transforms.ToTensor()
        ])
        return TransformEval