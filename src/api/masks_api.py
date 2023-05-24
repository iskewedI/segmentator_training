import requests

base_url = "http://127.0.0.1:5000/api"

def get_mask(options):
    """
    Generate a prediction given an B64 String image.
    Options = dict with:
        -- image: B64 String image.
    -----
    Returns:
        - Dict array that contains:
        -- maskb64: B64 String of the mask (mask white color, background black)
        -- classes_detected: String array of every class detected.
        -- masked_imgb64: B64 String of the image with the painted mask.
        -- mask_coords: Array [y1, x1, y2, x2]
    """
    response = requests.post(
        url=f'{base_url}/mask', json=options)
    
    return response.json()

def encode_img_to_b64s(options):
    """
    Returns the B64 string encoded image.
    Options = dict with:
        -- image_path: String absolute path to the image target to make the conversion.
    -----
    Returns:
        -- img: B64 string converted image.
    """
    response = requests.post(
        url=f'{base_url}/img/encode', json=options)

    return response.json()

def show_b64s_image(options):
    """
    Convert and show the image using PLT from the API.
    Options = dict with:
        -- image: B64 String of the image to display with PLT.
    -----
    Returns: none.
    """
    response = requests.post(
        url=f'{base_url}/img/show', json=options)

    return response.json()
