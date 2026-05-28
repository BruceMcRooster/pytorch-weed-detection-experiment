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

def predict(
    weights: Path,
    input_file: Path,
    conf: float = 0.5,
):
    if type(weights) != Path:
        weights = Path(weights)
    if type(input_file) != Path:
        input_file = Path(input_file)

    assert weights.is_file() and (weights.suffix.lower() == '.pt' or weights.suffix.lower() == '.pth'), \
        "weights must be a pytorch model weight save file"

    model = YOLO(weights)

    results = {}

    if input_file.is_dir():
        print(f"Detected input_file is a directory, iterating over files")

    # deduplicate logic to keep it in sync
    for file in input_file.iterdir() if input_file.is_dir() else [input_file]:
        if file.suffix != '.jpg' and file.suffix != '.jpeg' and file.suffix != '.png':
            print(f"Skipping {file}")
            continue
        result = model(file, save=True, conf=conf)
        results[file] = result

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    add_subparser(subparsers, train, 'train')
    add_subparser(subparsers, predict, 'predict')

    args = vars(parser.parse_args())

    target_function = args.pop('func')
    args.pop('command')

    if target_function == train:
        model, results = target_function(**args)
        print("\n\n=====FINAL RESULTS=====\n\n")
        print(results)
    elif target_function == predict:
        results = target_function(**args)
        print("\n\n=====PREDICTION RESULTS=====\n\n")
        for file, result in results.items():
            print(f"\n====={file}=====\n")
            print(result)


