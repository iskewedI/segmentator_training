from absl import flags
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import re
import tensorflow as tf
import xml.etree.ElementTree as XMLET

from image_transf.img_fn import rotate_img_bb
from image_transf.transformations import generate_cross_line
from data.types_fn import tuple_to_str

flags = tf.compat.v1.app.flags

tf.compat.v1.flags.DEFINE_string(
    'images_dir', None, 'Path to the dir containing the images.')
tf.compat.v1.flags.DEFINE_string(
    'out_dir', None, 'Dir in which save every result img and XML bounding box.')

FLAGS = flags.FLAGS

transformations = {
    "blur": [
        {
            "size": 27,
            "sigma": (1, 1.5)
        },
            {
            "size": 45,
            "sigma": (1.5, 2)
        }
    ],
    "solarize": [50, 100, 225],
    "posterize": [],
    "posterize": [6],
    # TODO: make it work with 90, 270, etc. How can I get the new min and max rotated coordinates?
    # "rotation": [180]
    "rotation": []
}


def save_img_xml(img, image_name, xml_tree, transformation_name, out_dir):
    new_img_name = f"{image_name}_{transformation_name}.jpg"
    new_xml_name = f"{image_name}_{transformation_name}.xml"

    out_path_img = os.path.abspath(os.path.join(out_dir, new_img_name))
    out_path_xml = os.path.abspath(os.path.join(out_dir, new_xml_name))

    xml_root = xml_tree.getroot()

    xml_root.find("./filename").text = new_img_name
    xml_root.find("./path").text = out_path_img

    print("Saving img and xml on...", out_path_img)
    xml_tree.write(out_path_xml)
    img.save(out_path_img, "JPEG")


def gen_transformations(image_name, image_path, xml_path, xml_tree):
    img = Image.open(image_path)

    # Greyscale
    grayscaled = transforms.Grayscale()(img)
    save_img_xml(grayscaled, image_name, xml_tree, "gc", FLAGS.out_dir)

    # TODO: adjust the bounding boxes after cropping!
    # cropped = transforms.CenterCrop((img.size[0] - 100, img.size[1] - 100))(img)
    # save_img_xml(cropped, image_name, xml_tree, "crp", FLAGS.out_dir)

    for blur_t in transformations["blur"]:
        size = blur_t["size"]
        sigma = blur_t["sigma"]

        name = f"{size}_{tuple_to_str(sigma)}"

        blurred_image = transforms.GaussianBlur(size, sigma)(img)
        save_img_xml(blurred_image, image_name, xml_tree, name, FLAGS.out_dir)

    r_inverted = transforms.RandomInvert(1)(img)
    save_img_xml(r_inverted, image_name, xml_tree, "rinv", FLAGS.out_dir)

    for solarize_px in transformations["solarize"]:
        name = f"sol_{solarize_px}"
        solarized = transforms.RandomSolarize(solarize_px, 1)(img)
        save_img_xml(solarized, image_name, xml_tree, name, FLAGS.out_dir)

    for posterize_ch in transformations["posterize"]:
        name = f"pos_{solarize_px}"
        posterized = transforms.RandomSolarize(posterize_ch, 1)(img)
        save_img_xml(posterized, image_name, xml_tree, name, FLAGS.out_dir)

    for rotation_d in transformations["rotation"]:
        name = f"rot_{rotation_d}"
        img_rotated = None

        xml_root = xml_tree.getroot()
        bounding_boxes = xml_root.findall("./object")
        for bb in bounding_boxes:
            xmin = bb.find("./bndbox/xmin")
            ymin = bb.find("./bndbox/ymin")
            xmax = bb.find("./bndbox/xmax")
            ymax = bb.find("./bndbox/ymax")

            rotated_img, rotated_bb = rotate_img_bb(
                image_path,
                [int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)],
                rotation_d
            )

            r_xmin, r_ymin = [min(rotated_bb[:,0]), min(rotated_bb[:,1])]
            r_xmax, r_ymax = [max(rotated_bb[:,0]), max(rotated_bb[:,1])]

            xmin.text = str(int(r_xmin))
            ymin.text = str(int(r_ymin))
            xmax.text = str(int(r_xmax))
            ymax.text = str(int(r_ymax))

            # pt1 = (int(r_xmin), int(r_ymin))
            # pt2 = (int(r_xmax), int(r_ymax))

            # cv2.rectangle(rotated_img, pt1, pt2, color=(255, 0, 0), thickness=1)
            # plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))

            # plt.show()

            img_rotated = Image.fromarray(rotated_img)

        save_img_xml(img_rotated, image_name, xml_tree, name, FLAGS.out_dir)

    crossed_arr = generate_cross_line(image_path)
    save_img_xml(Image.fromarray(crossed_arr), image_name, xml_tree, "line", FLAGS.out_dir)

def main(args):
    print("--- Analyzing flags...")
    assert FLAGS.images_dir, '`images_dir` missing.'
    assert FLAGS.out_dir, '`out_dir` missing.'

    dir = os.listdir(FLAGS.images_dir)
    if (len(dir) is 0):
        return

    if (not os.path.exists(FLAGS.out_dir)):
        print("Creating out dir => ", FLAGS.out_dir)
        os.mkdir(FLAGS.out_dir)

    data_arr = []

    for filename in dir:
        if (re.search(".xml", filename)):
            # XML file
            file_path = os.path.join(FLAGS.images_dir, filename)

            file_tree = XMLET.parse(file_path)
            file_root = file_tree.getroot()

            img_path = file_root.find("./path")
            if (img_path is None):
                print("Couldn't find relative image for => ", filename)
                return

            img_name = file_root.find("./filename")
            if (img_path is None):
                print("Couldn't find relative image name for => ", filename)
                return

            data_arr.append({
                "img_path": img_path.text,
                "img_name": img_name.text,
                "xml_name": filename,
                "xml_path": file_path,
                "xml_parsed": file_tree
            })

    for i, data in enumerate(data_arr):
        img_name = f"{data['img_name'][0:10]}{i}".rsplit('.', maxsplit=1)[0]

        gen_transformations(
            img_name,
            data["img_path"],
            data["xml_path"],
            data["xml_parsed"])


if __name__ == "__main__":
    print("--- Starting script...")

    tf.compat.v1.app.run()
