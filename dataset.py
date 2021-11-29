import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import utils


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 image_size=128,
                 normalize=True):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
            
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size, interpolation=0),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
                self.mask_transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size, interpolation=0),
                    transforms.ToTensor()
                ])
                self.sobel_transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size, interpolation=0),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                    
                ])
            else:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size, interpolation=0),
                    transforms.ToTensor()
                    
                ])
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s" % meta_file)
        self.num = len(lines)
        self.metas = []
        self.classifier = None
        for line in lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0], int(line_split[1])))
            else:
                self.metas.append((line_split[0], -1))
        print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/'+ 'image' + '/' +  self.metas[idx][0] + '.png'
        mask_filename = self.root_dir + '/' + 'mask' + '/' + self.metas[idx][0] + '.png'

        # # crfill
        # sobel_filename = self.root_dir + '/' + '4_inpainted_crfill' + '/' + self.metas[idx][0] + '.png'
        # 4_inpainted_outputs
        # /home/haneollee/dgm3/DGP-inpainting/data/others/

        # edge-connect
        sobel_filename = self.root_dir + '/' + '3_inpainted_outputs_EC' + '/' + self.metas[idx][0] + '_merged' + '.png'

        cls = self.metas[idx][1]
        img = default_loader(filename)
        mask_img = default_loader(mask_filename)
        sobel_img = default_loader(sobel_filename)

        # transform
        if self.transform is not None:
            img = self.transform(img)
            mask_img = self.mask_transform(mask_img)
            sobel_img = self.sobel_transform(sobel_img)

        return img, cls, self.metas[idx][0], mask_img, sobel_img
