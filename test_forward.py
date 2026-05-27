import torch
import torchvision

import utils

from coco_detection_fix import create_fixed_cocodetection_dataset
from get_transform import get_transform

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = create_fixed_cocodetection_dataset(
    root='data/weeds/train',
    annFile='data/weeds/train/_annotations.coco.json',
    transforms=get_transform(train=True)
)

g = torch.Generator()
g.manual_seed(42)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn,
    generator=g
)

# For Training
images, targets = next(iter(data_loader))
# print(f"{images = }, targets = {targets}")
# print(f"{type(targets[0]) = }")
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
# print(f"{targets = }")
output = model(images, targets)  # Returns losses and detections
print(output)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
print(predictions[0])
