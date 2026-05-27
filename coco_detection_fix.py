from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
import torch

def create_fixed_cocodetection_dataset(root: str, annFile: str, transforms = None) -> CocoDetection:
    if transforms is not None:
        transforms = T.Compose([T.PILToTensor(), transforms])
    dataset = CocoDetection(root=root, annFile=annFile, transforms=transforms)
    dataset = wrap_dataset_for_transforms_v2( # absolutely clutch because format is slightly different from expected
        dataset, target_keys=["boxes", "labels", "masks", "image_id", "area", "iscrowd"]
    )

    empty_tensor = torch.zeros((0, 4), dtype=torch.float32)

    class FixifyingWrapper(type(dataset), torch.utils.data.Dataset):
        def __getitem__(self, idx: int):
            image, potentially_bad_dict = super().__getitem__(idx)

            canvas_size = tuple(F.get_size(image))

            potentially_bad_dict.setdefault("boxes", tv_tensors.BoundingBoxes(
                empty_tensor, format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=canvas_size
            ))
            potentially_bad_dict.setdefault("masks", torch.zeros((0, *canvas_size), dtype=torch.uint8))
            potentially_bad_dict.setdefault("labels", torch.zeros(0, dtype=torch.int64))
            potentially_bad_dict.setdefault("area", torch.zeros(0, dtype=torch.float32))
            potentially_bad_dict.setdefault("iscrowd", torch.zeros(0, dtype=torch.uint8))
            return image, potentially_bad_dict

    dataset.__class__ = FixifyingWrapper

    return dataset