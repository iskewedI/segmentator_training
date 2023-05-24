import base64
import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as XMLET
import random
import os
import PIL.Image as Image

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    return _b64encode(buffered.getvalue())

def MAT2b64(mat):
    return base64.b64encode(cv2.imencode('.jpg', mat)[1]).decode()

def b642img(b64str) -> Image:
    imgdata = base64.b64decode(b64str)
    return Image.open(io.BytesIO(imgdata))

def b642MAT(b64str) -> np.ndarray:
    imgbytes = base64.b64decode(b64str)
    return np.array(Image.open(io.BytesIO(imgbytes)))

def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """

    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def rotate_img_bb(image_path, bb_coords, angle):
    img = cv2.imread(image_path)

    #Get all 4 coordinates of the box
    bb = np.array((
        (bb_coords[0],bb_coords[1]),
        (bb_coords[2],bb_coords[1]),
        (bb_coords[2],bb_coords[3]),
        (bb_coords[0],bb_coords[3])
    ))

    center = (img.shape[0]//2,img.shape[1]//2) #Get the center of the image

    rotMat = cv2.getRotationMatrix2D(center,angle,1.0) #Get the rotation matrix, its of shape 2x3

    img_rotated = cv2.warpAffine(img,rotMat,img.shape[1::-1]) #Rotate the image

    bb_rotated = np.vstack((bb.T,np.array((1,1,1,1)))) #Convert the array to [x,y,1] format to dot it with the rotMat
    bb_rotated = np.dot(rotMat,bb_rotated).T #Perform Dot product and get back the points in shape of (4,2)

    # print(bb_rotated2)
    # print(bb_rotated)
    # view_bb_img(img, bb)
    # view_bb_img(img_rotated, bb_rotated)

    return [img_rotated, bb_rotated]

def view_bb_img(img, bb):
    # bb[:, 0] # Every tuple first value (tuple[0])

    # bb[0, 0] # First column first value (tuples[0][0])
    # bb[0, 1] # First column second value (tuples[0][0])

    # bb[:,1] # Every tuple second value (tuple[1])

    plt.imshow(img)
    plt.plot(
        np.append(bb[:,0],bb[0,0]),
        np.append(bb[:,1],bb[0,1])
    )
    plt.show()

def show_MAT_img(MAT_img):
    cv2.imshow("image", MAT_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_and_bb(image_path, image = None):
    """ Gets the image, the relative XML file and shows with CV2 the image and the
    configured bounding boxes.
    ---------
    Parameters:
    ----------
    image_path : str
        The path containing the img. Assumed that the bounding boxes XML file will be the same
        file path but .xml.
    image : MAT
        The MAT loaded image. Optional. If not provided, it will open the image_path as a MAT.
    """
    img = image or cv2.imread(image_path)

    xml_path = f"{image_path.rsplit('.', maxsplit=1)[0]}.xml"

    xml_root = XMLET.parse(xml_path).getroot()

    bounding_boxes = xml_root.findall("./object")

    for bb in bounding_boxes:
        xmin = int(bb.find("./bndbox/xmin").text)
        ymin = int(bb.find("./bndbox/ymin").text)
        xmax = int(bb.find("./bndbox/xmax").text)
        ymax = int(bb.find("./bndbox/ymax").text)

        cv2.rectangle(img,
        (xmin, ymin),
        (xmax, ymax),
        tuple([random.randint(0, 255) for _ in range(3)]),
        thickness=3
        )


    show_MAT_img(img)

def show_all_images_bb(image_dir):
    content = os.listdir(image_dir)

    for image in content:
        image_path = os.path.join(image_dir, image)
        if (".jpg" in image_path):
            show_img_and_bb(image_path)

# show_all_images_bb("Tensorflow/workspace/images/augmented")

def b64img_draw_rect(b64img, pt1, pt2, color=(255, 255, 255), thickness=1) -> str:
    """ Loads an image as B64 string and using cv2 draw a rectangle. Returns a b64 encoded img string.
    ---------
    Parameters:
    ----------
    image_path : str
        The path containing the img. Assumed that the bounding boxes XML file will be the same
        file path but .xml.
    """
    img = convert_from_image_to_cv2(b642img(b64img))
    img[:] = 0

    cv2.rectangle(img, pt1, pt2, color, thickness)

    return MAT2b64(img)

def augment_img_mask_cords(coords, factor):
    """ Returns an augmented coords mask, by a factor (1.5, 2).
    ---------
    Parameters:
    ----------
    coords : tuple[]
        Mask coords to augment (tuple array) in form of [(top, left) (bottom right)]
    factor : int | float
        Escalar value to reduce every tuple member.
    """
    new_coords = [
        tuple(int(cord / factor) for cord in coords[0]),
        tuple(int(cord * factor) for cord in coords[1])
    ]

    return new_coords
