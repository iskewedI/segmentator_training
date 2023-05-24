import cv2
# from img_fn import show_MAT_img

def generate_cross_line(img_path):
    img = cv2.imread(img_path)

    color = (0, 0, 0)

    thickness_px = int(((img.shape[0] + img.shape[1]) / 2) / 20)

    image = cv2.line(
        img,
        (0, int(img.shape[0] / 2)),
        (img.shape[1], int(img.shape[0] / 2)),
        color,
        thickness_px
        )

    image = cv2.line(
        img,
        (int(img.shape[1] / 2), 0),
        (int(img.shape[1] / 2), img.shape[0]),
        color,
        thickness_px
    )

    # show_PIL_img(image)

    return image


# generate_cross_line("C:\\Users\\jtorn\\Desktop\\Dev\\Redes neuronales e AI\\ray_c\\Tensorflow\\workspace\\images\\augmented\\gettyimage2_gc.jpg")
