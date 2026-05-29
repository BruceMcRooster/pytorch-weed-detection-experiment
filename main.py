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
    quiet: bool = False,
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
        verbose=(not quiet)
    )

    return model, results

def predict(
    weights: Path,
    input_file: Path,
    conf: float = 0.5,
    quiet: bool = False
):
    if type(weights) != Path:
        weights = Path(weights)
    if type(input_file) != Path:
        input_file = Path(input_file)

    assert weights.is_file() and (weights.suffix.lower() == '.pt' or weights.suffix.lower() == '.pth'), \
        "weights must be a pytorch model weight save file"

    model = YOLO(weights)

    if not quiet and input_file.is_dir():
        print("Detected directory as input, will iterate through all files")

    return model(input_file, save=True, conf=conf, verbose=not quiet)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    add_subparser(subparsers, train, 'train')
    add_subparser(subparsers, predict, 'predict')

    args = vars(parser.parse_args())

    target_function = args.pop('func')
    args.pop('command')

    quiet = False
    if 'quiet' in args:
        quiet = args['quiet']

    if target_function == train:
        model, results = target_function(**args)
        if not quiet:
            print("\n\n=====FINAL RESULTS=====\n\n")
            print(results)
    elif target_function == predict:
        results = target_function(**args)
        if not quiet:
            print("\n\n=====PREDICTION RESULTS=====\n\n")
            total_infer_time = 0.0
            for result in results:
                print(f"\n====={result.path}=====\n")
                print(result)
                total_infer_time += result.speed['inference']
            print("\n\n=====CUMULATIVE RESULTS=====\n\n")
            print(f"Average inference time: {total_infer_time / len(results):.3f}ms")
