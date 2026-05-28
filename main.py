from pathlib import Path
from typing import Literal

import torch
from ultralytics import YOLO

from util import add_subparser

# nano segmentation model
MODEL = 'yolov8n-seg.pt'

def train(
    datayaml: Path,
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    initial_learning_rate: float = 0.01,
    device: Literal['cpu', 'cuda'] = 'cpu',
    save_period: int = 10,
    workers: int = 4,
):
    if type(datayaml) != Path:
        datayaml = Path(datayaml)
    assert datayaml.is_file() and (datayaml.suffix.lower() == '.yaml' or datayaml.suffix.lower() == '.yml'), \
        "datayaml must be a YAML file"

    model = YOLO('yolov8n-seg.pt')

    results = model.train(
        data=datayaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=initial_learning_rate,
        device=device,
        workers=workers,
        pretrained=True,
        save_period=save_period,
    )

    return model, results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    add_subparser(subparsers, train, 'train')

    args = vars(parser.parse_args())

    target_function = args.pop('func')
    args.pop('command')

    target_function(**args)


