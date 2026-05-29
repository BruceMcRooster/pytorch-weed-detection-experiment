import sys

if len(sys.argv) != 2:
    print("Usage: uv run convert-yoloe-to-coreml.py <input.pt>")

in_file = sys.argv[1]

from ultralytics import YOLOE

from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter
if BlobWriter is not None:
    print(f"Got the BlobWriter, it's {BlobWriter}")
else:
    print("BlobWriter is none")
    quit

model = YOLOE(in_file)

model.export(format='coreml')
