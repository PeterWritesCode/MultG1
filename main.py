# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

colorlist = ["gold", "red"]
colorlistRed = ["black", "red"]
colorlistGreen = ["black", "green"]
colorlistBlue = ["black", "blue"]


def encoder(img):
    print('Encoding image')
    # 3

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    cmgen = clr.LinearSegmentedColormap.from_list('myclrmap', colorlist, N=256)

    plt.figure()
    plt.imshow(R, cmgen)
    plt.axis('off')
    plt.show()

    cmred = clr.LinearSegmentedColormap.from_list('myred', colorlistRed, N=256)
    cmgreen = clr.LinearSegmentedColormap.from_list('mygreen', colorlistGreen, N=256)
    cmblue = clr.LinearSegmentedColormap.from_list('myblue', colorlistBlue, N=256)

    plt.figure()
    plt.imshow(R, cmred)
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(G, cmgreen)
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(B, cmblue)
    plt.axis('off')
    plt.show()

    # 4
    print(img.shape)

    height_or, width_or, channels = img.shape

    height = height_or
    width = width_or

    while (height % 16) != 0:
        img = np.pad(img, ((0, 1), (0, 0), (0, 0)), mode="edge")
        height, width, c = img.shape
    while (width % 16) != 0:
        img = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode="edge")
        height, width, c = img.shape

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(img.shape)

    return img


def decoder(img, height, width):
    print('Decoding image')

    # 4
    img = img[0:height, 0:width]

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(img.shape)

    # 3

    cmred_rev = clr.LinearSegmentedColormap.from_list('myred', colorlistRed[::-1], N=256)
    cmgreen_rev = clr.LinearSegmentedColormap.from_list('mygreen', colorlistGreen[::-1], N=256)
    cmblue_rev = clr.LinearSegmentedColormap.from_list('myblue', colorlistBlue[::-1], N=256)

    plt.figure()
    plt.imshow(img, cmred_rev)
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(img, cmgreen_rev)
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(img, cmblue_rev)
    plt.axis('off')
    plt.show()

    return img


def main():
    # 1
    img = {}

    img[0] = plt.imread('peppers.bmp')
    img[1] = plt.imread('logo.bmp')
    img[2] = plt.imread('barn_mountains.bmp')

    h, w, c = img[2].shape

    encoder(img[2])
    decoder(img[2], h, w)


if __name__ == '__main__':
    plt.close('all')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
