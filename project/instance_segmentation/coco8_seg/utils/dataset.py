import io
import torch
import numpy as np
from typing import Any, Tuple
from torchvision import datasets
from pycocotools.coco import COCO
from torchvision import tv_tensors
from torchvision.transforms import v2
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_convert

def collate_fn(batch):
    return tuple(zip(*batch))

class InstanceSegmentation(datasets.CocoDetection):
    def __init__(self, root, annFile, transform):
        super(InstanceSegmentation, self).__init__(root, annFile, transform)
        self.coco = COCO(annFile)
        self.dataset = self.coco.dataset
        self.v2_transform = transform
        self.annFile = annFile

    def __getitem__(self, index) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.v2_transform is not None:
            height, width = image.height, image.width
            original_bounding_box = torch.tensor([t['bbox'] for t in target])
            bounding_box_xyxy = box_convert(original_bounding_box, in_fmt = 'xywh', out_fmt = 'xyxy')
            tv_tensor_bounding_box_xyxy = tv_tensors.BoundingBoxes(bounding_box_xyxy, format = "XYXY", canvas_size = (height, width))
            label = torch.tensor([t['category_id'] for t in target])
            mask_np = np.array([self.coco.annToMask(t) for t in target])
            mask = torch.tensor(mask_np)
            transform_image, transform_bounding_box, transform_mask, transform_label = self.v2_transform(image, tv_tensor_bounding_box_xyxy, mask, label)

        dic = {
            "boxes": [],
            "labels": [],
            "image_id": -1,
            "masks": []
        }

        if not all(x == target[0]["image_id"] for x in [t["image_id"] for t in target]):
            raise ValueError("Not all values are equal in the list.")
        else:
            dic["image_id"] = target[0]["image_id"] 

        dic["boxes"] = transform_bounding_box
        dic["labels"] = transform_label
        dic["masks"] = transform_mask

        return transform_image, dic

    def __len__(self) -> int:
        return len(self.ids)

    def getImgIds(self, imgIds = [], catIds = []):
        return self.coco.getImgIds(imgIds = [], catIds = [])

    def getCatIds(self, catNms = [], supNms = [], catIds = []):
        return self.coco.getCatIds(catNms = [], supNms = [], catIds = [])

    def loadAnns(self, ids = []):
        return self.coco.loadAnns(ids = ids)

    def getAnnIds(self, imgIds = [], catIds = [], areaRng = [], iscrowd = None):
        return self.coco.getAnnIds(imgIds = imgIds, catIds = catIds, areaRng = areaRng, iscrowd = iscrowd)

def training_loader(rootPath, annFilePath):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale = True),
        v2.ToDtype(torch.float32, scale = True),
    ])

    with redirect_stdout(io.StringIO()):
        training_dataset = InstanceSegmentation(
            root = rootPath, 
            annFile = annFilePath,
            transform = transform
        )

    training_dataloader = DataLoader(
        training_dataset, 
        batch_size = 1,
        shuffle = True,
        num_workers = 0,
        collate_fn = collate_fn
    )

    return training_dataloader, training_dataset

def validation_loader(rootPath, annFilePath):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale = True),
        v2.ToDtype(torch.float32, scale = True),
    ])

    with redirect_stdout(io.StringIO()):
        validation_dataset = InstanceSegmentation(
            root = rootPath, 
            annFile = annFilePath,
            transform = transform
        )

    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size = 1,
        shuffle = True,
        num_workers = 0,
        collate_fn = collate_fn
    )

    return validation_dataloader, validation_dataset