import glob
import random
import os
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = 'A', 'label'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        # Convert grayscale images to rgb
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_names(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B, "name": image_A_name}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset_names_mask(Dataset):
    def __init__(self, root, transforms_=None, transforms1_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.transform1 = transforms.Compose(transforms1_)
        self.unaligned = unaligned
        self.root = root
        self.mode = "train"
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])
        mask_A = Image.open(os.path.join(self.root, "%s/mask" % self.mode) + '/' + image_A_name).convert('L')

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_mask = self.transform1(mask_A)
        item_mask = torch.where(item_mask > 0.5, torch.tensor(1.), torch.tensor(0.))
        return {"A": item_A, "B": item_B, "name": image_A_name, "mask_A": item_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset_names2(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])

        # image_B_name = ""
        # if self.unaligned:
        #     image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        # else:
        image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')
        file_root, image_B_name = os.path.split(self.files_B[index % len(self.files_B)])

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B, "name_A": image_A_name, "name_B": image_B_name}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset_image_name(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])

        item_A = self.transform(image_A)
        return {"A": item_A, "name": image_A_name}

    def __len__(self):
        return len(self.files_A)


class ImageDataset_label_mask(Dataset):
    def __init__(self, root, transforms_=None, transforms1_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.transform1 = transforms.Compose(transforms1_)
        self.unaligned = unaligned
        self.root = root
        self.mode = "train"
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        self.label_A = img2label_paths(self.files_A)

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        label_A = np.loadtxt(self.label_A[index % len(self.label_A)])
        if label_A.size == 5:
            label_A = np.array([label_A])

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])
        mask_A = Image.open(os.path.join(self.root, "%s/mask" % self.mode) + '/' + image_A_name).convert('L')

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_mask = self.transform1(mask_A)
        item_mask = torch.where(item_mask > 0.5, torch.tensor(1.), torch.tensor(0.))
        labels_out = torch.zeros((label_A.shape[0], 6))
        labels_out[:, 1:] = torch.from_numpy(label_A)
        return item_A, item_B, labels_out, item_mask, self.files_A[index % len(self.files_A)]

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    @staticmethod
    def collate_fn(batch):
        item_A, item_B, label, item_mask, path = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return {"A": torch.stack(item_A, 0), "B": torch.stack(item_B, 0), "Mask": torch.stack(item_mask, 0), "Label": torch.cat(label, 0), "Path": path}

class ImageDataset_label_mask_normal(Dataset):
    def __init__(self, root, transforms_=None, transforms1_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.transform1 = transforms.Compose(transforms1_)
        self.unaligned = unaligned
        self.root = root
        self.mode = "train"
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        self.label_A = img2label_paths(self.files_A)
        self.files_normal = "./datasets/normal.png"

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        label_A = np.loadtxt(self.label_A[index % len(self.label_A)])
        if label_A.size == 5:
            label_A = np.array([label_A])
        image_normal = Image.open(self.files_normal).convert('L')

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])
        mask_A = Image.open(os.path.join(self.root, "%s/mask" % self.mode) + '/' + image_A_name).convert('L')

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')
        
        item_normal = self.transform(image_normal)
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_mask = self.transform1(mask_A)
        item_mask = torch.where(item_mask > 0.5, torch.tensor(1.), torch.tensor(0.))
        labels_out = torch.zeros((label_A.shape[0], 6))
        labels_out[:, 1:] = torch.from_numpy(label_A)
        return item_A, item_B, labels_out, item_mask, self.files_A[index % len(self.files_A)], item_normal

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    @staticmethod
    def collate_fn(batch):
        item_A, item_B, label, item_mask, path, item_normal = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return {"A": torch.stack(item_A, 0), "B": torch.stack(item_B, 0), "Mask": torch.stack(item_mask, 0), "Label": torch.cat(label, 0), "Path": path, "Normal": torch.stack(item_normal, 0)}

class ImageDataset_names_B(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        file_root, image_B_name = os.path.split(self.files_B[index % len(self.files_B)])

        item_B = self.transform(image_B)

        return {"B": item_B, "name_B": image_B_name}

    def __len__(self):
        return len(self.files_B)


class ImageDataset_mask(Dataset):
    def __init__(self, root, transforms_=None, transforms1_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.transform1 = transforms.Compose(transforms1_)
        self.unaligned = unaligned
        self.root = root
        self.mode = "train"
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        # self.label_A = img2label_paths(self.files_A)

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        # label_A = np.loadtxt(self.label_A[index % len(self.label_A)])
        # if label_A.size == 5:
        #     label_A = np.array([label_A])

        file_root, image_A_name = os.path.split(self.files_A[index % len(self.files_A)])
        mask_A_path = os.path.join(self.root, "%s/mask" % self.mode) + '/' + image_A_name
        # if os.path.exists(mask_A_path):
        mask_A = Image.open(os.path.join(self.root, "%s/mask" % self.mode) + '/' + image_A_name).convert('L')
        # print(mask_A_path)
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        # if os.path.exists(mask_A_path):
            # print("mask transform ", mask_A_path)
        item_mask = self.transform1(mask_A)
        item_mask = torch.where(item_mask > 0.5, torch.tensor(1.), torch.tensor(0.))
        # else:
        #     item_mask = torch.zeros_like(item_A)
        # labels_out = torch.zeros((label_A.shape[0], 6))
        # labels_out[:, 1:] = torch.from_numpy(label_A)
        # return item_A, item_B, item_mask, self.files_A[index % len(self.files_A)]
        return {"A": item_A, "B": item_B, "Mask": item_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    # @staticmethod
    # def collate_fn(batch):
    #     item_A, item_B, item_mask, path = zip(*batch)  # transposed

    #     return {"A": torch.stack(item_A, 0), "B": torch.stack(item_B, 0), "Mask": torch.stack(item_mask, 0), "Path": path}
