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


class BruyndonckxMethod:
    def __init__(self, old_image_path: str, new_image_path: str):
        self.__empty_image_path: str = old_image_path
        self.__full_image_path: str = new_image_path
        self.__occupancy: int = 0

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

    def __find_div_index(self, sorted_pixels: np.array, *, diff_limit: int = 3) -> int:
        pixels_length = len(sorted_pixels)

        diffs = np.diff(sorted_pixels[:, 3])
        index = np.argmax(diffs[pixels_length // 4: 3 * pixels_length // 4 + 1]) + pixels_length // 4
        max_diff = np.max(diffs[pixels_length // 4: 3 * pixels_length // 4 + 1])

        if max_diff > diff_limit:
            index = pixels_length // 2 - 1
        return index

    def __set_groups_to_pixels(self, sorted_pixels, div_index: int) -> dict[np.array]:
        pixels_length = len(sorted_pixels)

        group_1 = np.random.choice(['1A', '1B'] * div_index, size=div_index, replace=False)
        group_2 = np.random.choice(['2A', '2B'] * (pixels_length - div_index), size=pixels_length - div_index,
                                   replace=False)

        all_groups = np.concatenate((group_1, group_2), axis=0)
        d = dict()
        for k, pixel in enumerate(sorted_pixels):
            d.setdefault(all_groups[k], []).append(pixel)
        for k, v in d.items():
            d[k] = np.asarray(v, dtype='uint8')

        if np.mean(d['1A']) >= np.mean(d['2A']) or np.mean(d['1B']) >= np.mean(d['2B']):
            raise ValueError
        return d

    def embed(self, message: str, key_generator: int):
        np.random.seed(key_generator)

        img = Image.open(self.__empty_image_path).convert('RGBA')
        image = np.asarray(img, dtype='uint8')
        img.close()
        image[:, :, 3] = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(int)

        height, width = image.shape[0], image.shape[1]

        message_bits = deque(BruyndonckxMethod.str_to_bits(message))
        print(message_bits)
        if len(message_bits) > (height // 8) * (width // 8):
            raise ValueError('Размер сообщения превышает размер контейнера!')

        occupancy = 0
        for i in range(8, height, 8):
            for j in range(8, width, 8):
                pixels = np.asarray(image[i - 8: i, j - 8: j, :], dtype='uint8')
                pixels = pixels.reshape(-1, 4)
                # Присвоение масок
                pixels = pixels[pixels[:, 3].argsort()]

                div_index = self.__find_div_index(pixels)
                g_p: dict = self.__set_groups_to_pixels(pixels, div_index)

                bit = message_bits.popleft()
                delta_l: int = 2
                sign: int = 1 if bit else -1

                size_1A = len(g_p['1A'])
                size_1B = len(g_p['1B'])
                size_1 = size_1A + size_1B
                size_2A = len(g_p['2A'])
                size_2B = len(g_p['2B'])
                size_2 = size_2A + size_2B

                g_p['1A'][:, 3] = g_p['1A'][:, 3] - (
                        (size_1 * np.mean(np.concatenate([g_p['1A'][:, 3], g_p['1B'][:, 3]],
                                                         axis=0)) + sign * size_1B * delta_l) // size_1)
                g_p['1B'][:, 3] = g_p['1B'][:, 3] - (
                        (size_1 * np.mean(np.concatenate([g_p['1A'][:, 3], g_p['1B'][:, 3]],
                                                         axis=0)) - sign * size_1A * delta_l) // size_1)
                g_p['2A'][:, 3] = g_p['2A'][:, 3] - (
                        (size_2 * np.mean(np.concatenate([g_p['2A'][:, 3], g_p['2B'][:, 3]],
                                                         axis=0)) + sign * size_2B * delta_l) // size_2)
                g_p['2B'][:, 3] = g_p['2B'][:, 3] - (
                        (size_2 * np.mean(np.concatenate([g_p['2A'][:, 3], g_p['2B'][:, 3]],
                                                         axis=0)) - sign * size_2A * delta_l) // size_2)

                occupancy += 1
                if not message_bits:
                    self.__occupancy = occupancy
                    Image.fromarray(image, 'RGBA').save(self.__full_image_path, 'PNG')
                    np.random.seed(0)
                    return

    def recover(self, key_generator: int) -> str:
        np.random.seed(key_generator)

        img = Image.open(self.__full_image_path).convert('RGBA')
        image = np.asarray(img, dtype='uint8')
        print(image[0, :, :])
        img.close()

        height, width = image.shape[0], image.shape[1]

        message_bits = []
        for i in range(8, height, 8):
            for j in range(8, width, 8):
                pixels = np.asarray(image[i - 8: i, j - 8: j, :], dtype='uint8')
                pixels = pixels.reshape(-1, 4)
                # Присвоение масок
                pixels = pixels[pixels[:, 3].argsort()]

                div_index = self.__find_div_index(pixels)
                g_p: dict = self.__set_groups_to_pixels(pixels, div_index)

                if np.mean(g_p['1A'][:, 3]) - np.mean(g_p['1B'][:, 3]) < 0 and np.mean(g_p['2A'][:, 3]) - np.mean(
                        g_p['2B'][:, 3]) < 0:
                    message_bits.append(0)
                elif np.mean(g_p['1A'][:, 3]) - np.mean(g_p['1B'][:, 3]) > 0 and np.mean(g_p['2A'][:, 3]) - np.mean(
                        g_p['2B'][:, 3]) > 0:
                    message_bits.append(1)
                if len(message_bits) == self.__occupancy:
                    print(message_bits)
                    recovered_message = BruyndonckxMethod.bits_to_str(message_bits)
                    np.random.seed(0)
                    return recovered_message


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

    # Универсальный индекс качества (УИК)
    # С помощью данной метрики оцениваются
    # коррелированность, изменение динамического диапазона, а также изменение
    # среднего значения одного изображения относительно другого.
    # -1 <= UQI <= 1
    # минимальному искажению изображения соответствуют
    # значения UQI ~ 1
    sigma = np.sum((empty - np.mean(empty)) * (full - np.mean(full))) / (H * W)
    UQI = (4 * sigma * np.mean(empty) * np.mean(full)) / \
          ((np.var(empty) ** 2 + np.var(full) ** 2) * (np.mean(empty) ** 2 + np.mean(full) ** 2))
    print(f'Универсальный индекс качества (УИК):\n{UQI}\n')


def e_probability(message: str, recovered_message: str) -> float:
    message_bits = np.asarray(BruyndonckxMethod.str_to_bits(message))
    recovered_message_bits = np.asarray(BruyndonckxMethod.str_to_bits(recovered_message))
    return np.mean(np.abs(message_bits - recovered_message_bits[:message_bits.shape[0]])) * 100


def main():
    load_dotenv('.env')
    key: int = int(os.getenv('KEY'))

    old_image = 'input/old_image.png'
    new_image = 'output/new_image.png'

    with open('message.txt', mode='r', encoding=encoding) as file:
        message = file.read()

    bruyndonckx = BruyndonckxMethod(old_image, new_image)
    bruyndonckx.embed(message, key)
    recovered_message = bruyndonckx.recover(key)
    print('Ваше сообщение:\n{}'.format(recovered_message))

    print(e_probability(message, recovered_message))
    # metrics(old_image, new_image)


if __name__ == '__main__':
    main()
