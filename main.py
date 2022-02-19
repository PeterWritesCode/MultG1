# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

Red = (1, 0, 0)
Green = (0, 1, 0)
Blue = (0, 0, 1)


def encoder(img):
    print('Encoding image')

    # 3
    cmred = clr.LinearSegmentedColormap.from_list('myred', [(0, 0, 0), Red], N=256)
    cmgreen = clr.LinearSegmentedColormap.from_list('mygreen', [(0, 0, 0), Green], N=256)
    cmblue = clr.LinearSegmentedColormap.from_list('myblue', [(0, 0, 0), Blue], N=256)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    plt.figure()
    plt.imshow(R, cmred)
    plt.figure()
    plt.imshow(G, cmgreen)
    plt.figure()
    plt.imshow(B, cmblue)

    # 4
    print(img.shape)

    height_or, width_or, channels = img.shape

    height = height_or
    width = width_or

    while (height % 16) != 0:
        img = np.pad(img, ((0, height), (0, 0), (0, 0)), mode="edge")
        height = img.shape[0]
    while (width % 16) != 0:
        img = np.pad(img, ((0, 0), (0, width), (0, 0)), mode="edge")
        width = img.shape[1]

    plt.figure()
    plt.imshow(img)
    print(img.shape)

    decoder(img, height_or, width_or)


def decoder(img, height, width):
    print('Decoding image')

    # 4
    img = img[0:height, 0:width]

    plt.figure()
    plt.imshow(img)
    print(img.shape)

    # 3
    cmred_rev = clr.LinearSegmentedColormap.from_list('myred', [Red, (0, 0, 0)], N=256)
    cmgreen_rev = clr.LinearSegmentedColormap.from_list('mygreen', [Green, (0, 0, 0)], N=256)
    cmblue_rev = clr.LinearSegmentedColormap.from_list('myblue', [Blue, (0, 0, 0)], N=256)

    plt.figure()
    plt.imshow(img, cmred_rev)
    plt.figure()
    plt.imshow(img, cmgreen_rev)
    plt.figure()
    plt.imshow(img, cmblue_rev)


def main():
    # 1
    img = {}

    img[0] = plt.imread('peppers.bmp')
    img[1] = plt.imread('logo.bmp')
    img[2] = plt.imread('barn_mountains.bmp')

    encoder(img[1])


if __name__ == '__main__':
    plt.close('all')
    main()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
