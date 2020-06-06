import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

import os 

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


path_data_base = '../data'
## Add v5 data
json_annotation = f'{path_data_base}/tk1957-v7/train/annotations.json'
image_path_base = f'{path_data_base}/tk1957-v7/train'
register_coco_instances(f"TKDatav7-train", {}, json_annotation, image_path_base)

## Add v4 eval data
json_annotation = f'{path_data_base}/tk1957-v7/val/annotations.json'
image_path_base = f'{path_data_base}/tk1957-v7/val'
register_coco_instances(f"TKDatav7-val", {}, json_annotation, image_path_base)



cfg = get_cfg()
cfg.merge_from_file("../outputs/tkmodel7/mask_cascade_rcnn/config.yaml")
cfg.MODEL.WEIGHTS = os.path.join('../outputs/tkmodel7/mask_cascade_rcnn', "model_0039999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("HJDataset_test", )
cfg.TEST.DETECTIONS_PER_IMAGE  = 200
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  

cfg.DATASETS.TRAIN = ("TKDatav7-train",)
cfg.DATASETS.TEST = ("TKDatav7-val",)

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("TKDatav7-val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "TKDatav7-val")

res = inference_on_dataset(predictor.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
print(res)