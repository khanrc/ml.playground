""" asyncio example
References:
    - https://soooprmx.com/archives/6882
    - https://soooprmx.com/archives/8620
"""
import re
import io
from pathlib import Path
import base64
import time

from PIL import Image
import numpy as np
import requests
import asyncio


root = Path("/app/fonts/font/assets/preprocess/pngs/char/char_001")


def read_image(path):
    image = Image.open(path)
    image.load()
    return image


def sync_read(root):
    st = time.time()
    for path in root.iterdir():
        image = read_image(path)
        print(path, image.size)

    print("Sync elapsed = {:.1f}s".format(time.time() - st))


def async_read(root):
    # PIL Image.open & np.asarray 자체가 blocking function 이라
    # async 하게 작동하지 않음.
    st = time.time()

    ###########
    async def get_image(path):
        image = read_image(path)
        print(path, image.size)
        return path, image.size

    async def test_images(root):
        queue = [get_image(path) for path in root.iterdir()]
        for f in asyncio.as_completed(queue):
            path, shape = await f

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_images(root))
    ###########

    print("Async elapsed = {:.1f}s".format(time.time() - st))


def async_read2(root):
    # https://soooprmx.com/archives/8620
    # loop.run_in_executor 를 사용해서 run loop 에 넣어주면 async 하게 실행시킬 수 있다.
    st = time.time()

    ###########
    async def get_image(path):
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, read_image, path)
        print(path, image.size)
        return path, image.size

    async def test_images(root):
        queue = [get_image(path) for path in root.iterdir()]
        for f in asyncio.as_completed(queue):
            path, shape = await f

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_images(root))
    ###########

    print("Async2 elapsed = {:.1f}s".format(time.time() - st))


if __name__ == "__main__":
    # task: read 37 images
    sync_read(root)   # 24.2s
    async_read(root)  # 25.2s
    async_read2(root) # 5.0s
