from collections import deque
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings(action='once')

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
    pointer: int = 0

    def __init__(self, rgba: np.array):
        assert rgba.shape[0] == 4 and rgba.dtype == np.uint8
        self.__rgba: np.array = rgba
        self.__order = Pixel.pointer
        Pixel.pointer += 1

    @property
    def rgba(self) -> np.array:
        return self.__rgba

    @rgba.setter
    def rgba(self, value):
        assert value.shape[0] == 4 and value.dtype == np.uint8
        self.__rgba = value

    @property
    def order(self) -> int:
        return self.__order

    def __repr__(self):
        return '({}, {})\n'.format(self.rgba, self.order)


class BruyndonckxMethod:
    def __init__(self, old_image_path: str, new_image_path: str):
        self.__empty_image_path: str = old_image_path
        self.__full_image_path: str = new_image_path
        self.__n: int = 8
        self.__diff_limit: int = 2
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
        pixels_arr = np.asarray([pixel.rgba for pixel in sorted_pixels], dtype=np.uint8)
        diffs_by_alpha = np.diff(pixels_arr[:, 3])
        shift = 24
        div_index = np.argmax(diffs_by_alpha[shift:-shift]) + shift
        if diffs_by_alpha[div_index] > self.__diff_limit:
            div_index = len(sorted_pixels) // 2
        return div_index

    def __make_pixel_groups_by_masks(self, div_index: int) -> dict[str: list]:
        d = {}
        for i in range(64):
            group = np.random.choice(['1A', '1B']) if i < div_index else np.random.choice(['2A', '2B'])
            d.setdefault(group, []).append(i)
        return d

    def __embed_bit_with_modification_pixels_brightness(self, sorted_block_pixels: list[Pixel], g: dict[str: list], bit: int):
        arr = np.asarray([pixel.rgba for pixel in sorted_block_pixels], dtype=np.uint8)
        delta_l = self.__delta_l
        sign = 1 if bit else -1

        g1_arr = arr[(g['1A'] + g['1B'])]
        arr[g['1A'], 3] -= (np.mean(arr[g['1A'], 3]) - (
                np.mean(g1_arr[:, 3]) + (sign * arr[g['1B']].shape[0] * delta_l / g1_arr.shape[0]))).astype(
            np.uint8)

        arr[g['1B'], 3] -= (np.mean(arr[g['1B'], 3]) - (
                np.mean(g1_arr[:, 3]) - (sign * arr[g['1A']].shape[0] * delta_l / g1_arr.shape[0]))).astype(
            np.uint8)

        g2_arr = arr[(g['2A'] + g['2B'])]
        arr[g['2A'], 3] -= (np.mean(arr[g['2A'], 3]) - (
                np.mean(g2_arr[:, 3]) + (sign * arr[g['2B']].shape[0] * delta_l / g2_arr.shape[0]))).astype(
            np.uint8)

        arr[g['2B'], 3] -= (np.mean(arr[g['2B'], 3]) - (
                np.mean(g2_arr[:, 3]) - (sign * arr[g['2A']].shape[0] * delta_l / g2_arr.shape[0]))).astype(
            np.uint8)

        # print('__________________________________')
        # print('bit: {}'.format(bit))
        # mean_1A = np.mean(arr[g['1A'], 3])
        # print('Среднее l1a: {}'.format(mean_1A))
        # mean_1B = np.mean(arr[g['1B'], 3])
        # print('Среднее l1b: {}'.format(mean_1B))
        # mean_2A = np.mean(arr[g['2A'], 3])
        # print('Среднее l2a: {}'.format(mean_2A))
        # mean_2B = np.mean(arr[g['2B'], 3])
        # print('Среднее l2b: {}'.format(mean_2B))
        # print('Сравнение:\n1A - 1B: {}\n2A - 2B: {}'.format(mean_1A - mean_1B, mean_2A - mean_2B))
        # assert (mean_1A - mean_1B > 0 and mean_2A - mean_2B > 0) or (mean_1A - mean_1B < 0 and mean_2A - mean_2B < 0)
        # print('__________________________________')

        for i, pixel in enumerate(arr):
            sorted_block_pixels[i].rgba = pixel

    def __recover_bit_from_modified_block(self, sorted_pixels: list[Pixel], g: dict[str: list]) -> int | None:
        arr = np.asarray([pixel.rgba for pixel in sorted_pixels], dtype=np.uint8)
        if (np.mean(arr[g['1A'], 3]) - np.mean(arr[g['1B'], 3]) < 0) and (np.mean(arr[g['2A'], 3]) - np.mean(
                arr[g['2B'], 3]) < 0):
            return 0
        elif (np.mean(arr[g['1A'], 3]) - np.mean(arr[g['1B'], 3]) > 0) and (np.mean(arr[g['2A'], 3]) - np.mean(
                arr[g['2B'], 3]) > 0):
            return 1
        return None

    def embed(self, message: str, key_generator: int):
        np.random.seed(key_generator)

        img = Image.open(self.__empty_image_path).convert('RGBA')
        image = np.asarray(img, dtype=np.uint8)
        image[:, :, 3] = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(int)
        img.close()

        height, width = image.shape[0], image.shape[1]

        message_bits = BruyndonckxMethod.str_to_bits(message)
        message_bits_length = len(message_bits)
        if message_bits_length > (height // 8) * (width // 8):
            raise ValueError('Размер сообщения превышает размер контейнера!')
        message_bits = deque(message_bits)

        for start, end in self.__define_bounds_of_blocks(height, width)[:message_bits_length]:
            old_block = image[start.i: end.i, start.j: end.j].copy()
            old_block = old_block.reshape(-1, 4)
            assert len(old_block) == 64

            new_block = sorted([Pixel(pixel) for pixel in old_block], key=lambda obj: obj.rgba[3])
            div_index = self.__find_div_index(tuple(new_block))
            group_index = self.__make_pixel_groups_by_masks(div_index)
            bit = message_bits.popleft()
            self.__embed_bit_with_modification_pixels_brightness(new_block, group_index, bit)
            new_block = sorted(new_block, key=lambda obj: obj.order)
            new_block = (np.asarray([pixel.rgba for pixel in new_block], dtype=np.uint8)).reshape(self.__n, self.__n, 4)
            image[start.i: end.i, start.j: end.j] = new_block[:, :]

        Image.fromarray(image, 'RGBA').save(self.__full_image_path, 'PNG')
        np.random.seed()

    def recover(self, key_generator: int) -> str:
        np.random.seed(key_generator)

        img = Image.open(self.__full_image_path).convert('RGBA')
        image = np.asarray(img, dtype=np.uint8)
        img.close()

        height, width = image.shape[0], image.shape[1]

        message_bits = []
        for start, end in self.__define_bounds_of_blocks(height, width):
            modified_block = image[start.i: end.i, start.j: end.j].copy()
            modified_block = modified_block.reshape(-1, 4)
            assert len(modified_block) == 64

            modified_block = sorted([Pixel(pixel) for pixel in modified_block], key=lambda pixel:
                                    int(0.299 * pixel.rgba[0] + 0.587 * pixel.rgba[1] + 0.114 * pixel.rgba[2]))

            div_index = self.__find_div_index(tuple(modified_block))
            group_index = self.__make_pixel_groups_by_masks(div_index)

            bit = self.__recover_bit_from_modified_block(modified_block, group_index)
            if bit is not None:
                message_bits.append(bit)
            else:
                np.random.seed()
                message = BruyndonckxMethod.bits_to_str(message_bits)
                return message


def metrics(empty_image: str, full_image: str) -> None:
    img = Image.open(empty_image).convert('RGBA')
    empty = np.asarray(img, dtype=np.uint8)
    empty[:, :, 3] = (0.299 * empty[:, :, 0] + 0.587 * empty[:, :, 1] + 0.114 * empty[:, :, 2]).astype(int)
    img.close()

    img = Image.open(full_image).convert('RGBA')
    full = np.asarray(img, dtype=np.uint8)
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
    recovered_message = bruyndonckx.recover(key)
    print('Ваше сообщение:\n{}'.format(recovered_message))

    print('Вероятность ошибки: {}'.format(e_probability(message, recovered_message)))
    metrics(old_image, new_image)


if __name__ == '__main__':
    main()
