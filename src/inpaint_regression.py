from time import sleep
from absl import flags
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import imutils
from api.masks_api import encode_img_to_b64s, get_mask
from api.sd_api import interrogate
from image_transf.img_fn import MAT2b64, b642img, convert_from_image_to_cv2
from static.regression_steps import common_config, steps
from inpaint import inpaint

INPAINTING_FILL_METHODS = ['fill', 'original',
                           'latent_noise', 'latent_nothing']

flags = tf.compat.v1.app.flags

tf.compat.v1.flags.DEFINE_string(
    'sd_ckpt_name', '', 'Display name of the ckpt in SD')
tf.compat.v1.flags.DEFINE_string(
    'image_path', '', 'Path to the image to be inpainted.')
tf.compat.v1.flags.DEFINE_string(
    'text_prompt', '', 'Prompt text to generate the inpainting.')
tf.compat.v1.flags.DEFINE_string(
    'negative_prompt', '', "Text of things you don't want to see in the generation.")
tf.compat.v1.flags.DEFINE_string(
    'output_dir', '.', 'Output data directory.')

FLAGS = flags.FLAGS

def start_regression():
    print("Starting regression...")
    encode_res = encode_img_to_b64s(options={
        "image_abs_path": FLAGS.image_path,
        "width": 512,
        "height": 512
        })

    b64img = encode_res["img"]

    result = get_mask(options={"image": b64img})
    if(result is None):
        print("No masks detected")
        return

    maskb64, classes_detected, masked_imgb64 = result["maskb64"], result["classes_detected"], result["masked_imgb64"]

    classes_detected = classes_detected.split(",")

    if (len(classes_detected) == 0):
            print("No masks detected")
            return

    tags = FLAGS.text_prompt
    negative_tags = FLAGS.negative_prompt

    focused_prompt = [f'(({tag}))' for tag in tags.split(",")]
    focused_negative_prompt = [
        f'(({negative_tag}))' for negative_tag in negative_tags.split(",")]

    input_prompt = {
        "text": ",".join(focused_prompt),
        "negative": ",".join(focused_negative_prompt)
    }

    last_mask_thickness = 5

    # interrogation_res = interrogate(b64img, {"model": "deepdanbooru"})
    interrogation_res = interrogate(b64img, {"model": "clip"})
    if(interrogation_res is None):
        print("Couln't interrogate image. Exiting...")
        return

    tags = "".join(interrogation_res["caption"])

    # Data retrieved after every image processing (SD API)
    last_seed = -1
    # The last image generated in b64.
    last_imgb64 = b64img
    # Last mask generated
    last_maskb64 = maskb64

    last_masked_imgb64 = masked_imgb64

    for i, step in enumerate(steps):
        # Things that can be replaced in the fly

        config = {}
        config.update(common_config)
        config.update({
            "sd_model_checkpoint": FLAGS.sd_ckpt_name,
            "seed": last_seed if step.get("restore_seed") else -1})
        config.update(step.get("config"))

        new_mask = step.get("new_mask")

        if(new_mask is not None):
            thickness = last_mask_thickness

            mask_img = convert_from_image_to_cv2(b642img(last_maskb64))

            gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # All img black
            if(not new_mask.get("mask_filled")):
                thickness = int(thickness * new_mask.get("thickness_multiplier"))

                # Save the multiplier value
                last_mask_thickness = thickness

                mask_img[:] = 0
            else:
                thickness = -1

            # draw the contours of c
            cv2.drawContours(mask_img, [c], -1, (255, 255, 255), thickness)

            last_maskb64 = MAT2b64(mask_img)

            imgmask_img = convert_from_image_to_cv2(b642img(last_masked_imgb64))

            cv2.drawContours(imgmask_img, [c], -1, (255, 255, 255), thickness)

            last_masked_imgb64 = MAT2b64(imgmask_img)

        input_mask = {
            "mask": last_maskb64,
            "classes_detected": classes_detected,
            "img_maskb64": last_masked_imgb64
        }

        print(f"""
            --- Starting step {step.get('name')}
            - Steps: {config.get("steps")}
            - Mask blur: {config.get("mask_blur")}
            - Denoising strenght: {config.get("denoising_strength")}
            - CFG Scale: {config.get("cfg_scale")}
            - Inpainting fill: {INPAINTING_FILL_METHODS[config.get("inpainting_fill")]}

            - Using last image generated as input: {"No" if last_imgb64 is None else "Yes" }
            - Restore seed: {step.get("restore_seed")}
            - Using seed: {last_seed}
            - New mask settings: {config.get("new_mask")}
            - Interrogation tags: {tags}
            """)

        result = inpaint(
            input_prompt,
            # Pass the latest generated image or the input image (first time)
            last_imgb64,
            input_mask,
            config,
            out_dir=FLAGS.output_dir,
            id=f"{i}-{step['name']}",
            tags=tags
        )

        if(result is not None):
            last_seed = result.get("seed")
            last_imgb64 = result.get("imgb64")

        sleep(1)

def main(args):
    start_regression()

if __name__ == "__main__":
    print("--- Starting script...")
    tf.compat.v1.app.run()
