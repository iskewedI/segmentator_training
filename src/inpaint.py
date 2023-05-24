import os
import json

from image_transf.img_fn import save_encoded_image
from api.sd_api import img2img

common_prompts_per_class = {
    "upper_clothes": {
        "text": "good skin, healthy, hd, 1girl, high resolution",
        "negative": "animated, childish, 3d bad body, fat, ugly"
    },
    "lower_clothes": {
        "text": "good skin, hd, 1girl, high resolution, detailed,  trained legs",
        "negative": "animated, childish, 3d bad body, fat, ugly, highly muscular"
    }
}

default_options = {
        "CLIP_stop_at_last_layers": 2,
        "mask_blur": 6,
        "inpaint_full_res": False,
        "sampler_name": "Euler",
        "cfg_scale": 11,
        "width": 512,
        "height": 512,
    }

def getUniqueTextFromList(list_in):
    return ",".join(list(set(",".join(list_in).split(", "))))


def inpaint(input_prompt, b64img, input_mask, options, out_dir, tags=[], id=""):
    maskb64 = input_mask["mask"]
    mask_classes = input_mask["classes_detected"]
    img_maskb64 = input_mask["img_maskb64"]

    common_prompts = {
        # Avoid duplicated values converting to set.
        # Get the combination of input props + common props for each mask class
        "text": getUniqueTextFromList([",".join(common_prompts_per_class[mask_class]["text"] for mask_class in mask_classes)]),
        "negative": getUniqueTextFromList([",".join(common_prompts_per_class[mask_class]["negative"] for mask_class in mask_classes)])
    }

    prompt = {
        "text": getUniqueTextFromList([input_prompt['text'], common_prompts['text'], tags]),
        "negative": getUniqueTextFromList([input_prompt['negative'], common_prompts['negative']])
    }

    api_config = {}
    api_config.update({
        "init_images": [b64img],
        "mask": maskb64,
        "prompt": prompt["text"],
        "negative_prompt": prompt["negative"],
        })
    api_config.update(options)

    print("--- Making request to API")
    response = img2img(api_config)

    print("--- Getting JSON response")
    output_img_b64 = response['images'][0]

    out_img = f"{out_dir}/{id}_result-{str(mask_classes)}.png"
    print("--- Saving img result on => ", out_img)
    save_encoded_image(output_img_b64, out_img)

    print("--- Saving mask")
    out_mask = f"{out_dir}/mask_{os.path.basename(out_img)}"
    save_encoded_image(maskb64, out_mask)

    save_encoded_image(img_maskb64, f"{out_dir}/imgmask_{os.path.basename(out_img)}")

    info = response.get("info")

    if(info is not None):
        deserealized = json.loads(info)

        return { "seed": deserealized.get("seed"), "imgb64": response['images'][0]}
