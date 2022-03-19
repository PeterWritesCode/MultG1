# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
import scipy.fftpack as fft
import math as m
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc  # pip install Pillow
import matplotlib.pylab as pylab

colorlist = ["gold", "red"]
colorlistRed = ["black", "red"]
colorlistGreen = ["black", "green"]
colorlistBlue = ["black", "blue"]


def downsample_inter(img_in, num):
    w, h, c = img_in.shape
    if num == 420:
        img_out = np.zeros((w//2, h//2, 3), dtype=np.uint8)
    elif num == 422:
        img_out = np.zeros((w//2, h, 3), dtype=np.uint8)
    for i in range(3):
        img = img_in[:, :, i]
        stepX = 2
        stepY = 2
        if num == 420:  # coisa bonita
            dsImg = img[::stepY, ::stepX]

        elif num == 422:  # coisa bonita
            dsImg = img[::stepY, :]

        print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

        dsImgInterp = cv2.resize(img, None, fx=1 / stepX, fy=1 / stepY, interpolation=cv2.INTER_LINEAR)

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('image')

        fig.add_subplot(1, 2, 2)
        plt.imshow(dsImgInterp)
        if num == 420:  # coisa bonita
            plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5 interpolated')

        elif num == 422:  # coisa bonita
            plt.title('downsampled 4:2:2 sx = 0.5 interpolated')
        plt.axis('image')
        plt.show()

        img_out[:, :, i] = dsImgInterp

    return img_out


def upsample_rep(img_in, num):
    w, h, c = img_in.shape
    if num == 420:
        img_out = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)
    elif num == 422:
        img_out = np.zeros((w * 2, h, 3), dtype=np.uint8)
    for i in range(3):
        dsImg = img_in[:, :, i]
        stepX = 2
        stepY = 2
        print("\nUpsampling with repetitions\n")

        fig = plt.figure(figsize=(20, 20))

        if num == 420:  # coisa bonita
            usImg = np.repeat(dsImg, stepX, axis=1)
            l, c = usImg.shape
            usImg = np.repeat(usImg, stepY, axis=0)
        elif num == 422:  # coisa bonita
            usImg = np.repeat(dsImg, stepX, axis=1)
            l, c = usImg.shape

        fig.add_subplot(1, 2, 1)
        plt.imshow(dsImg)
        plt.title('original')
        plt.axis('image')

        fig.add_subplot(1, 2, 2)
        plt.imshow(usImg)
        plt.title('upsampled with repetitions')
        plt.axis('image')
        plt.show()

        print("dsImg size = ", dsImg.shape)
        print("usImg size = ", usImg.shape)
        img_out[:, :, i] = usImg

    return img_out


def upsample_inter(img_in, num):
    w, h, c = img_in.shape
    if num == 420:
        img_out = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)
    elif num == 422:
        img_out = np.zeros((w * 2, h, 3), dtype=np.uint8)
    for i in range(3):
        dsImg = img_in[:, :, i]
        stepX = 2
        stepY = 2

        print("Upsampling with interpolation")

        fig = plt.figure(figsize=(20, 20))
        if num == 420:
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=stepY, interpolation=cv2.INTER_LINEAR)
        elif num == 422:  # coisa bonita
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=1, interpolation=cv2.INTER_LINEAR)
        fig.add_subplot(1, 2, 1)
        plt.imshow(dsImg)
        plt.title('original')
        plt.axis('image')

        fig.add_subplot(1, 2, 2)
        plt.imshow(usImg)
        plt.title('upsampled with interpolation')
        plt.axis('image')
        plt.show()

        print()
        print("dsImg size = ", dsImg.shape)
        print("usImg size = ", usImg.shape)
        img_out[:, :, i] = usImg

    return img_out


def downsample_nointer(img_in, num):
    w, h, c = img_in.shape
    if num == 420:
        img_out = np.zeros((w//2, h//2, 3), dtype=np.uint8)
    elif num == 422:
        img_out = np.zeros((w//2, h, 3), dtype=np.uint8)
    for i in range(3):
        img = img_in[:, :, i]
        stepX = 2
        stepY = 2

        print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

        if num == 420:  # coisa bonita
            dsImg = img[::stepY, ::stepX]

        elif num == 422:  # coisa bonita
            dsImg = img[::stepY, :]

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('image')

        # fig.add_subplot(1, 2, 2)
        # plt.imshow(dsImg)
        # if num == 420:  # coisa bonita
        #     plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5')
        #
        # elif num == 422:  # coisa bonita
        #     plt.title('downsampled 4:2:2 sx = 0.5')
        # plt.axis('image')
        # plt.show()

        fig.add_subplot(1, 2, 2)
        plt.imshow(dsImg)
        if num == 420:  # coisa bonita
            plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5  no interp')

        elif num == 422:  # coisa bonita
            plt.title('downsampled 4:2:2 sx = 0.5 no interp')
        plt.axis('image')
        img_out[:, :, i] = dsImg

    return img_out


def DCT(img):
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)

    dctImg = fft.dct(fft.dct(img, norm="ortho").T, norm="ortho").T
    dctLogImg = np.log(np.abs(dctImg) + 0.0001)

    # fig = plt.figure(figsize=(20, 20))

    # fig.add_subplot(1, 3, 1)
    # plt.imshow(img, cm_grey)
    # plt.title('original')
    # plt.axis('image')

    # fig.add_subplot(1, 3, 2)
    # plt.imshow(dctImg, cm_grey)
    # plt.title('DCT')
    # plt.axis('image')

    # fig.add_subplot(1, 3, 3)
    # plt.imshow(dctLogImg, cm_grey)
    # plt.title('DCT log')
    # plt.axis('image')
    # plt.show()

    invDctImg = fft.idct(fft.idct(dctImg, norm="ortho").T, norm="ortho").T

    fig = plt.figure(figsize=(20, 20))

    # fig.add_subplot(1, 4, 1)
    # plt.imshow(img, cm_grey)
    # plt.title('original')
    # plt.axis('image')

    # fig.add_subplot(1, 4, 2)
    # plt.imshow(invDctImg, cm_grey)
    # plt.title('IDCT')
    # plt.axis('image')
    # plt.show()

    # fig = plt.figure(figsize=(5, 5))
    # diffImg = img - invDctImg
    # diffImg[diffImg < 0.000001] = 0.

    # plt.imshow(diffImg, cm_grey)
    # plt.title('original - invDCT')
    # plt.axis('image')
    # plt.show()

    return dctImg


def dctBasisImg(img, d):
    imsize = img.shape
    dct = np.zeros(imsize)

    for i in r_[:imsize[0]:d]:
        for j in r_[:imsize[1]:d]:
            dct[i:(i + d), j:(j + d)] = dct2(img[i:(i + d), j:(j + d)])

    # for l in range(0, d1):
    #     for k in range(0, d2):
    #         for i in range(0, d):
    #             for j in range(0, d):1
    #                 u = d * k + j
    #                 # if(u>=img.shape[1]):
    #                 #     u = u%d;
    #                 #print(u)
    #                 v = l * d + i
    #                 # if(v>= img.shape[0]):
    #                 #     v = v%d;
    #                 img[v, u] = m.cos((2 * j + 1) * (2 * k) * m.pi / (4 * 8)) * m.cos(
    #                     (2 * i + 1) * (2 * l) * m.pi / (4 * 8))

    return dct


def dct2(a):
    dctImg = fft.dct(fft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
    return dctImg


def DCT_block(img_in):
    for i in range(3):
        img = img_in[:, :, i]
        dct8x8 = dctBasisImg(img, 8)
        cmBW = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
        plt.figure()
        plt.imshow(dct8x8, cmap='gray', vmax=np.max(dct8x8) * 0.01, vmin=0)
        plt.title("8x8 DCTs of the image")
        # n = 64
        # height1, width1 = img.shape

        # if (height1 % n) != 0:
        #     resto = n - height1 % n
        #     print(resto, "resto", height1,width1)
        #     img = np.pad(img, ((0, resto), (0, 0)), mode="edge")
        # resto = 0
        # if (width1 % n) != 0:
        #     resto = n - width1 % n
        #     print(resto, "resto", height1, width1)
        #     img = np.pad(img, ((0, 0), (0, resto)), mode="edge")
        #     resto = 0
        # height1, width1 = img.shape
        # if(height1 < width1):
        #     resto = width1-height1
        #     img = np.pad(img, ((0, resto), (0, 0)), mode="edge")
        # elif(width1 < height1):
        #     resto = height1-width1
        #     img = np.pad(img, ((0, 0), (0, resto)), mode="edge")

        # print(img.shape, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        dct64x64 = dctBasisImg(img, 64)


def visualizacao(img):
    # 3
    img = user_colormap(img)
    img = colormap(img)
    return img


def user_colormap(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    cmgen = clr.LinearSegmentedColormap.from_list('myclrmap', colorlist, N=256)

    plt.figure()
    plt.imshow(R, cmgen)
    plt.axis('off')
    plt.show()

    return img


def colormap(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
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

    return img


def RGB2YCbCr(img, tc):
    # 5
    R = img[:, :, 0]
    floatR = R.astype(float)
    G = img[:, :, 1]
    floatG = G.astype(float)
    B = img[:, :, 2]
    floatB = B.astype(float)
    cbcr = np.empty_like(img, dtype=float)

    # Y
    cbcr[:, :, 0] = tc[0][0] * floatR + tc[0][1] * floatG + tc[0][2] * floatB
    # Cb
    cbcr[:, :, 1] = 128 + tc[1][0] * floatR + tc[1][1] * floatG + tc[1][2] * floatB
    # Cr
    cbcr[:, :, 2] = 128 + tc[2][0] * floatR + tc[2][1] * floatG + tc[2][2] * floatB

    transcol = cbcr

    colorlistgray = ["black", (0.5, 0.5, 0.5)]

    cmgray = clr.LinearSegmentedColormap.from_list('myclrmap', colorlistgray, N=256)

    # transcol = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    plt.figure()
    plt.imshow(transcol[:, :, 0], cmgray)
    print(transcol.shape)
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(transcol[:, :, 1], cmgray)
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(transcol[:, :, 2], cmgray)
    plt.axis('off')
    plt.show()

    return transcol, cbcr


def getRGB(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def invRGB(R, G, B, shape):
    inv = np.zeros(shape)
    inv[:, :, 0] = R
    inv[:, :, 1] = G
    inv[:, :, 2] = B

    return inv


def YCbCr2RGb(img, tc):

    Y = img[:, :, 0]
    Cb = img[:, :, 1]
    Cr = img[:, :, 2]

    tc_invertida = np.linalg.inv(tc)
    print(tc_invertida)

    R = Y * tc_invertida[0][0] + (Cb - 128) * tc_invertida[0][1] + (Cr - 128) * tc_invertida[0][2]
    G = Y * tc_invertida[1][0] + (Cb - 128) * tc_invertida[1][1] + (Cr - 128) * tc_invertida[1][2]
    B = Y * tc_invertida[2][0] + (Cb - 128) * tc_invertida[2][1] + (Cr - 128) * tc_invertida[2][2]

    rgb = img.astype(float)
    print(R[:8, :8])
    print("-----------------------")

    rgb = invRGB(R, G, B, img.shape)
    rgb = rgb.round()

    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0
    rgb = np.uint8(rgb)

    print(tc_invertida.T)

    plt.figure()
    plt.imshow(rgb)
    plt.show()

    return rgb


def getImage_inv(img):
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


def padding(img):
    print(img.shape)
    print("padding")
    height_or, width_or, channels = img.shape

    height = height_or
    width = width_or

    # while (height % 16) != 0:
    #     img = np.pad(img, ((0, 1), (0, 0), (0, 0)), mode="edge")
    #     height = img.shape[0]
    # while (width % 16) != 0:
    #     img = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode="edge")
    #     width = img.shape[1]
    R, G, B = getRGB(img)

    if (height % 16) != 0:
        resto = 16 - height % 16

        # rowR = R[-1, :]
        # rowG = G[-1, :]
        # rowB = B[-1, :]
        # rowR = np.repeat(rowR, resto, 0)
        # rowG = np.repeat(rowG, resto, 0)
        # rowB = np.repeat(rowB, resto, 0)

        # np.r_['-1', img[], rowR]
        # G = np.r_['-1', G, rowG ]
        # B = np.r_['-1', B, rowB]
        img = np.pad(img, ((0, resto), (0, 0), (0, 0)), mode="edge")
        resto = 0

    if (width % 16) != 0:
        resto = 16 - width % 16

        # columnR = Rc[:, -1]
        # columnG = Gc[:, -1]

        # columnB = Bc[:, -1]
        # np.repeat(columnR, resto, 1)
        # np.repeat(columnG, resto, 1)
        # np.repeat(columnB, resto, 1)

        # Rc = np.hstack([R, columnR])
        # Gc = np.hstack([G, columnG])
        # Bc = np.hstack([B, columnB])
        img = np.pad(img, ((0, 0), (0, resto), (0, 0)), mode="edge")
        resto = 0

    plt.figure()
    plt.imshow(img)
    plt.show()
    print(img.shape)

    return img


def getImageOriginal(img, height, width):
    img = img[0:height, 0:width]

    plt.figure()
    plt.imshow(img)
    print(img.shape)
    plt.axis('off')
    plt.show()

    return img


def encoder(img, tc):
    print('Encoding image')
    # 2
    img = visualizacao(img)
    # 4
    img = padding(img)
    # 5
    img, cbcr = RGB2YCbCr(img, tc)

    cbcr_inter = downsample_inter(cbcr, 420)

    cbcr = downsample_nointer(cbcr, 420)

    # y_d = DCT(img)

    DCT_block(cbcr)

    return img, cbcr, cbcr_inter


def decoder(img, cbcr, cbcr_inter, h, w, tc):
    print('Decoding image')
    cbcr_inter = upsample_inter(cbcr_inter, 420)

    cbcr = upsample_rep(cbcr, 420)
    # 5
    img = YCbCr2RGb(img, tc)
    # 4
    img = getImageOriginal(img, h, w)
    # 3
    img = getImage_inv(img)

    return img


def main():
    # 1
    img = {}

    img[0] = plt.imread('peppers.bmp')
    img[1] = plt.imread('logo.bmp')
    img[2] = plt.imread('barn_mountains.bmp')

    k = 2
    h, w, c = img[k].shape
    img_in = img[k]

    plt.figure()
    plt.imshow(img[k])
    plt.show()

    tc = np.array([[0.299, 0.587, 0.114],
                   [-0.168736, -0.331264, 0.5],
                   [0.5, -0.418688, -0.081312]])

    img_enc, cbcr, cbcr_inter = encoder(img_in, tc)
    img_dec = decoder(img_enc, cbcr, cbcr_inter, h, w, tc)
    # comparison = img[1] == img_dec
    # print(comparison.all())
    # print(img[2], " \n A \n", img_dec)


if __name__ == '__main__':
    plt.close('all')
    main()
