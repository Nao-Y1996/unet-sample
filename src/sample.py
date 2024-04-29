import os

from pycocotools import coco

from const.const_values import ROOT

# Load annotations
coco = coco.COCO(os.path.join(ROOT, 'datasets/annotations/instances_val2017.json'))

