from collections import deque
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

encoding: str = 'utf-8'


class Point:
    """Координата в матрице"""
    def __init__(self, i: int, j: int):
        assert (i >= 0 & j >= 0)
        self.__i: int = i
        self.__j: int = j

    @property
    def i(self) -> int:
        return self.__i

    @property
    def j(self) -> int:
        return self.__j

    def __repr__(self):
        return '({}, {})'.format(self.i, self.j)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (self.i, self.j) == (other.i, other.j)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.__class__, self.i, self.j))


class Pixel:
    counter: int = 0

    def __init__(self, rgba: np.array):
        assert rgba.shape[0] == 4
        self.__rgba: np.array = rgba
        Pixel.counter += 1
        self.__count = Pixel.counter

    @property
    def rgba(self):
        return self.__rgba

    @property
    def count(self):
        return self.__count

    def __repr__(self):
        return '({}, {})\n'.format(self.rgba, self.count)


class GroupedPixels:
    def __init__(self):
        self.__pixels: list[Pixel] = []
        self.__arr = None

    @property
    def pixels(self):
        return self.__pixels

    def convert_pixels_to_arr(self):
        self.__arr = np.asarray([pixel.rgba for pixel in self.pixels], dtype='uint8')

    @property
    def arr(self):
        return self.__arr

    def __repr__(self):
        return '{}'.format(self.pixels)


class BruyndonckxMethod:
    def __init__(self, old_image_path: str, new_image_path: str):
        self.__empty_image_path: str = old_image_path
        self.__full_image_path: str = new_image_path
        self.__n: int = 8
        self.__diff_limit: int = 3
        self.__delta_l = 8

    @staticmethod
    def str_to_bits(message: str) -> list:
        result = []
        for num in list(message.encode(encoding=encoding)):
            result.extend([(num >> x) & 1 for x in range(7, -1, -1)])
        return result

    @staticmethod
    def bits_to_str(bits: list) -> str:
        chars = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return ''.join(chars)

    def __define_bounds_of_blocks(self, height: int, width: int) -> tuple[tuple[Point, ...]]:
        n = self.__n
        nested_list = [[tuple([i, j]) for j in range(n, width, n)] for i in range(n, height, n)]
        start_and_end_points_of_blocks = tuple([
            tuple([Point(i=elements[0] - n, j=elements[1] - n), Point(i=elements[0], j=elements[1])])
            for sub_list in nested_list for elements in sub_list])
        return start_and_end_points_of_blocks

    def __find_div_index(self, sorted_pixels: tuple[Pixel]) -> int:
        pixels_arr = np.asarray([pixel.rgba for pixel in sorted_pixels], dtype='uint8')
        diffs_by_alpha = np.diff(pixels_arr[:, 3])
        shift = 24
        div_index = np.argmax(diffs_by_alpha[shift:-shift]) + shift
        if diffs_by_alpha[div_index] > self.__diff_limit:
            div_index = len(sorted_pixels) // 2
        return div_index

    def __make_pixel_groups_by_masks(self, sorted_pixels: tuple[Pixel], div_index: int) -> dict[str: GroupedPixels]:
        d: dict[str: GroupedPixels] = {mask: GroupedPixels() for mask in ('1A', '1B', '2A', '2B')}
        for i, pixel in enumerate(sorted_pixels):
            group = np.random.choice(['1A', '1B']) if i < div_index else np.random.choice(['2A', '2B'])
            d[group].pixels.append(pixel)
        for v in d.values():
            v.convert_pixels_to_arr()
        assert (np.mean(d['1A'].arr[:, 3]) - np.mean(d['2A'].arr[:, 3]) < 0
                and np.mean(d['1B'].arr[:, 3]) - np.mean(d['2B'].arr[:, 3]))
        return d

    def __pixel_brightness_modification(self, g: dict[str: GroupedPixels], bit: int):
        delta_l = self.__delta_l
        sign = 1 if bit == 1 else -1

        g1_arr = np.concatenate([g['1A'].arr, g['1B'].arr], axis=0)
        m_g1A = np.mean(g['1A'].arr[:, 3])
        g['1A'].arr[:, 3] -= (
                    m_g1A - (np.mean(g1_arr[:, 3]) + (sign * g['1B'].arr.shape[0] * delta_l / g1_arr.shape[0]))).astype(
            np.uint8)

        m_g1B = np.mean(g['1B'].arr[:, 3])
        g['1B'].arr[:, 3] -= (
                    m_g1B - (np.mean(g1_arr[:, 3]) - (sign * g['1A'].arr.shape[0] * delta_l / g1_arr.shape[0]))).astype(
            np.uint8)

        g2_arr = np.concatenate([g['2A'].arr, g['2B'].arr], axis=0)
        m_g2A = np.mean(g['2A'].arr[:, 3])
        g['2A'].arr[:, 3] -= (
                    m_g2A - (np.mean(g2_arr[:, 3]) + (sign * g['2B'].arr.shape[0] * delta_l / g2_arr.shape[0]))).astype(
            np.uint8)

        m_g2B = np.mean(g['2B'].arr[:, 3])
        g['2B'].arr[:, 3] -= (
                    m_g2B - (np.mean(g2_arr[:, 3]) - (sign * g['2A'].arr.shape[0] * delta_l / g2_arr.shape[0]))).astype(
            np.uint8)

    def embed(self, message: str, key_generator: int):
        np.random.seed(key_generator)

        img = Image.open(self.__empty_image_path).convert('RGBA')
        image = np.asarray(img, dtype='uint8')
        image[:, :, 3] = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(int)
        img.close()

        height, width = image.shape[0], image.shape[1]

        message_bits = BruyndonckxMethod.str_to_bits(message)
        message_bits_length = len(message_bits)
        print(message_bits)
        if message_bits_length > (height // 8) * (width // 8):
            raise ValueError('Размер сообщения превышает размер контейнера!')
        message_bits = deque(message_bits)

        for start, end in self.__define_bounds_of_blocks(height, width)[:message_bits_length]:
            pixels = image[start.i: end.i, start.j: end.j, :].copy()
            pixels = pixels.reshape(-1, 4)
            assert len(pixels) == 64

            sorted_pixels: list[Pixel] = sorted([Pixel(pixel) for pixel in pixels], key=lambda obj: obj.rgba[3])
            div_index: int = self.__find_div_index(tuple(sorted_pixels))
            print(div_index)
            grpd_pixels: dict[str: GroupedPixels] = self.__make_pixel_groups_by_masks(tuple(sorted_pixels), div_index)
            self.__pixel_brightness_modification(grpd_pixels, message_bits.popleft())
            print(sorted_pixels)
            print(grpd_pixels)
            exit(0)
            # image[start.i: end.i, start.j: end.j, :] = pixels.reshape(self.__n, self.__n, 4)

        Image.fromarray(image, 'RGBA').save(self.__full_image_path, 'PNG')
        np.random.seed()

    def recover(self, key_generator: int) -> str:
        np.random.seed(key_generator)

        img = Image.open(self.__full_image_path).convert('RGBA')
        image = np.asarray(img, dtype='uint8')
        img.close()

        height, width = image.shape[0], image.shape[1]

        message_bits = []


def metrics(empty_image: str, full_image: str) -> None:
    img = Image.open(empty_image).convert('RGBA')
    empty = np.asarray(img, dtype='uint8')
    empty[:, :, 3] = (0.299 * empty[:, :, 0] + 0.587 * empty[:, :, 1] + 0.114 * empty[:, :, 2]).astype(int)
    img.close()

    img = Image.open(full_image).convert('RGBA')
    full = np.asarray(img, dtype='uint8')
    img.close()

    max_d = np.max(np.abs(empty.astype(int) - full.astype(int)))
    print('Максимальное абсолютное отклонение:\n{}'.format(max_d))

    NMSE = np.sum((empty - full) ** 2) / np.sum(empty * empty)
    print('Нормированное среднее квадратичное отклонение :\n{}'.format(NMSE))

    H, W = empty.shape[0], empty.shape[1]
    MSE = np.sum((empty - full) ** 2) / (W * H)
    print('Среднее квадратичное отклонение:\n{}'.format(MSE))


def e_probability(message: str, recovered_message: str) -> float:
    message_bits = np.asarray(BruyndonckxMethod.str_to_bits(message))
    recovered_message_bits = np.asarray(BruyndonckxMethod.str_to_bits(recovered_message))
    return round(100 * np.mean(np.abs(message_bits - recovered_message_bits[:message_bits.shape[0]])), 2)


def main():
    load_dotenv('.env')
    key: int = int(os.getenv('KEY'))

    old_image = 'input/old_image.png'
    new_image = 'output/new_image.png'

    with open('message.txt', mode='r', encoding=encoding) as file:
        message = file.read()

    bruyndonckx = BruyndonckxMethod(old_image, new_image)
    bruyndonckx.embed(message, key)
    # recovered_message = bruyndonckx.recover(key)
    # print('Ваше сообщение:\n{}'.format(recovered_message))
    #
    # print(e_probability(message, recovered_message))
    # metrics(old_image, new_image)


if __name__ == '__main__':
    main()
