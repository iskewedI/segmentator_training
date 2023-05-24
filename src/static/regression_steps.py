INPAINTING_FILL_METHODS = ['fill', 'original',
                           'latent_noise', 'latent_nothing']

common_config = {
    "CLIP_stop_at_last_layers": 2,
    "sampler_name": "Euler",
    "resize_mode": 1,
    "height": 1024,
    "width": 712,
    "restore_faces": False,
}

steps = [
    {
        "name": "Original mask.",
        "config": {
            "mask_blur": 6,
            "steps": 40,
            "denoising_strength": 0.75,
            "cfg_scale": 11,
            "inpainting_fill": 0,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
        },
    },
    {
        "name": "Repaint - Borders 1",
        "config": {
            "mask_blur": 6,
            "steps": 40,
            "denoising_strength": 0.65,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res_padding": 32,
            "inpaint_full_res": True,
        },
        # It will use the last generated seed.
        "restore_seed": True,
        # Replace IA generated mask
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 2,
            "mask_filled": False
        }

    },
    {
        "name": "Repaint - Borders 2",
        "config": {
            "mask_blur": 8,
            "steps": 40,
            "denoising_strength": 0.55,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res_padding":32,
            "inpaint_full_res": True,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 2,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Borders 3",
        "config": {
            "mask_blur": 10,
            "steps": 30,
            "denoising_strength": 0.45,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res_padding": 32,
            "inpaint_full_res": True,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 2,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Borders 4",
        "config": {
            "mask_blur": 10,
            "steps": 30,
            "denoising_strength": 0.45,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res_padding": 32,
            "inpaint_full_res": True,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 2,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Fill mask 1",
        "config": {
            "mask_blur": 12,
            "steps": 35,
            "denoising_strength": 0.25,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 1,
            "mask_filled": True
        }
    },
    {
        "name": "Repaint - Fill mask 2",
        "config": {
            "mask_blur": 14,
            "steps": 35,
            "denoising_strength": 0.25,
            "cfg_scale": 11,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1.02,
            "thickness_multiplier": 1,
            "mask_filled": True
        }
    },
    {
        "name": "Repaint - Entire image 1",
        "config": {
            "mask_blur": 14,
            "steps": 30,
            "denoising_strength": 0.04,
            "cfg_scale": 15,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 100,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 3,
            "thickness_multiplier": 60,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Entire image 2",
        "config": {
            "mask_blur": 12,
            "steps": 30,
            "denoising_strength": 0.03,
            "cfg_scale": 17,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 100,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1,
            "thickness_multiplier": 1,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Entire image 3",
        "config": {
            "mask_blur": 14,
            "steps": 40,
            "denoising_strength": 0.02,
            "cfg_scale": 22,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 100,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1,
            "thickness_multiplier": 1,
            "mask_filled": False
        }
    },
    {
        "name": "Repaint - Entire image 3",
        "config": {
            "mask_blur": 14,
            "steps": 40,
            "denoising_strength": 0.01,
            "cfg_scale": 25,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 100,
        },
        "restore_seed": True,
        "new_mask": {
            "mask_multiplier": 1,
            "thickness_multiplier": 1,
            "mask_filled": True
        }
    },
]
