import logging
from io import BytesIO
from typing import BinaryIO

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
file_handler = logging.FileHandler('log/app.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI()

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def perform_selfie_segmentation(image: BinaryIO, bg_image: BinaryIO) -> np.ndarray:
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        bg_img = cv2.imdecode(np.frombuffer(bg_image.read(), np.uint8), 1)

        image_height, image_width, _ = img.shape
        bg_img = cv2.resize(bg_img, (image_width, image_height))

        results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        condition = np.stack((cv2.blur(results.segmentation_mask, (5, 5)),) * 3, axis=-1) > 0.6

        out_img = np.where(condition, img, bg_img)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return out_img


def create_response_image(img: np.ndarray) -> StreamingResponse:
    pillow_img = Image.fromarray(img)
    response_img = BytesIO()
    pillow_img.save(response_img, 'JPEG')
    response_img.seek(0)

    return StreamingResponse(response_img, media_type='image/jpeg')


@app.post('/segmentation')
async def image_segmentation(image: UploadFile = File(...), bg_image: UploadFile = File(...)):
    logger.info('/segmentation endpoint called')
    if image.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(400, detail='Invalid file type')
    result_img = perform_selfie_segmentation(image.file, bg_image.file)
    return create_response_image(result_img)


@app.get('/hello')
async def hello_world():
    logger.info('/hello endpoint called')
    return {'Message': 'Hello World'}
