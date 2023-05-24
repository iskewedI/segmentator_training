from matplotlib import pyplot as plt
import numpy as np
import cv2
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
# from google.colab import drive
import os

from main import paths, files

# Train model
# python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\ssd_custom_model --pipeline_config_path=Tensorflow\workspace\models\ssd_custom_model\pipeline.config --num_train_steps=2000

# Get metrics (loss, precision)
# python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\ssd_custom_model --pipeline_config_path=Tensorflow\workspace\models\ssd_custom_model\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\ssd_custom_model

# Open tensorboard
# Training metrics
# tensorboard --logdir="Tensorflow\workspace\models\ssd_custom_model\train"

# Evaluation metrics
# tensorboard --logdir="Tensorflow\workspace\models\ssd_custom_model\eval"

# Freezing (saving) the graph
# python Tensorflow\models\research\object_detection\exporter_main_v2.py  --input_type=image_tensor --pipeline_config_path=Tensorflow\workspace\models\ssd_custom_model\pipeline.config --trained_checkpoint_dir=Tensorflow\workspace\models\ssd_custom_model --output_directory=Tensorflow\workspace\models\ssd_custom_model\export
