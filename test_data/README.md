# Inference demo data

`truck.jpg` is the official SAM notebook example:
https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg

`truck_boxes.json` supplies one box prompt around a wheel, in COCO `[x, y, width, height]` format.
It does not contain ground-truth segmentation masks and is used only to demonstrate inference.
For real SA-1B inputs, pass `--image_path` and `--annotation_path`; COCO RLE `segmentation`
annotations remain supported and will be used to sample foreground points as in the original demo.
