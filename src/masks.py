from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import cv2
import numpy as np
import PIL.Image as Image
import tensorflow as tf


@tf.function
def detect_fn(model, image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections


def get_masks(img, checkpoint, pipeline_config, label_map):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(
        label_map)

    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(detection_model, input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    # image_np_with_detections = image_np.copy()

    boxes = detections['detection_boxes']
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .7
    # # iterate over all objects found
    coordinates = []
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            class_id = int(
                detections['detection_classes'][i] + label_id_offset)

            coordinates.append({
                "box": boxes[i],
                "class_name": category_index[class_id]["name"],
                "score": scores[i]
            })

    color_by_class = {"upper_clothes": (
        255, 255, 0), "lower_clothes": (0, 255, 255)}

    image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')

    im_width, im_height = image_pil.size

    classes_detected = []
    masks = []

    img_mask = img.copy()

    masks_together = [{
        "mask": img.copy(),
        "name": "",
        "coords": [(), ()]
    }]

    masks_together[0]["mask"][:] = 0

    for coordinate in coordinates:
        class_name = coordinate["class_name"]

        classes_detected.append(class_name)

        box = tuple(coordinate["box"].tolist())
        ymin, xmin, ymax, xmax = box

        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                      int(ymin * im_height), int(ymax * im_height))

        start_point = (left, top)
        end_point = (right, bottom)

        # cv2.rectangle(img, start_point, end_point, color=color_by_class[class_name], thickness=3)

        # img = cv2.putText(img, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_by_class[class_name], 3)

        mask = img.copy()
        mask[:] = 0

        num_channels = 1 if len(mask.shape) == 2 else mask.shape[2]

        # TODO: delete. Just for tests now.
        if (class_name == "upper_clothes"):
            # Masks separated by class name
            cv2.rectangle(mask, start_point, end_point, color=(
                255,) * num_channels, thickness=-1)
            masks.append({"name": class_name, "mask": mask,
                         "coords": [start_point, end_point]})

        # Both of the masks together in the same img.
        cv2.rectangle(masks_together[0]["mask"], (left, top), (right, bottom), color=(
            255,) * num_channels, thickness=-1)

        # Set the max cords to the max coordinates, so it will contain every shape detected.
        cords = masks_together[0]["coords"]
        masks_together[0]["coords"] = [max(cords[0], (left, top)), min(
            cords[1], (right, bottom)) if len(cords[1]) > 0 else (right, bottom)]

        masks_together[0]["name"] = class_name

        # To show original image with the mask on front
        cv2.rectangle(img_mask, (left, top), (right, bottom),
                      color=(255,) * num_channels, thickness=-1)

    return [masks, masks_together, classes_detected, img_mask]
