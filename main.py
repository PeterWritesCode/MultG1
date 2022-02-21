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

def visualizacao(img):
    # 3
    userColormap(img)
    colormap(img)

def userColormap(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    cmgen = clr.LinearSegmentedColormap.from_list('myclrmap', colorlist, N=256)

    plt.figure()
    plt.imshow(R, cmgen)
    plt.axis('off')
    plt.show()
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

def RGB2YCbCr(img):
    # 5
    R = img[:, :, 0]
    floatR = R.astype(np.float)
    G = img[:, :, 1]
    floatG = G.astype(np.float)
    B = img[:, :, 2]
    floatB = B.astype(np.float)
    cbcr = np.empty_like(img)

    # Y
    cbcr[:, :, 0] = .299 * floatR + .587 * floatG + .114 * floatB
    # Cb
    cbcr[:, :, 1] = -128 - .168736 * floatR - .331264 * floatG + .5 * floatB
    # Cr
    cbcr[:, :, 2] = -128 + .5 * floatR - .418688 * floatG - .081312 * floatB

    transcol = np.uint8(cbcr)

    colorlistgray = ["black", "gray"]

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

    return transcol


def getRGB(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B

def invRGB(R, G, B, shape):
    inv = np.zeros(shape)
    inv[:,:,0] = R
    inv[:,:,1] = G
    inv[:,:,2] = B

def YCbCr2RGb(img):
# 5
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    plt.figure()
    plt.imshow(img)
    plt.show()

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
        resto = height % 16
        rowR = R[-1, :]
        rowG = G[-1, :]
        rowB = B[-1, :]
        np.vstack[R, np.repeat(rowR, resto)]
        np.vstack[G, np.repeat(rowG, resto)]
        np.vstack[B, np.repeat(rowB, resto)]

    if (width % 16) != 0:
        resto = width % 16
        columnR = R[:, -1]
        columnG = G[:, -1]
        columnB = B[:, -1]
        np.vstack(R, np.repeat(columnR, resto))
        np.vstack(G, np.repeat(columnG, resto))
        np.vstack(B, np.repeat(columnB, resto))

    plt.figure()
    plt.imshow(img)
    plt.show()
    print(img.shape)


def getImageOriginal(img):
    img = img[0:height, 0:width]

    plt.figure()
    plt.imshow(img)
    print(img.shape)
    plt.axis('off')
    plt.show()
def encoder(img):
    print('Encoding image')
    #2
    visualizacao(img)
    #4
    padding(img)
    #5
    RGB2YCbCr(img)



def decoder(img, height, width):
    print('Decoding image')
    #5
    YCbCr2RGb(img)
    # 4
    getImageOriginal(img)
    # 3
    getImage_inv(img)

def main():
    # 1
    img = {}

    img[0] = plt.imread('peppers.bmp')
    img[1] = plt.imread('logo.bmp')
    img[2] = plt.imread('barn_mountains.bmp')

    h, w, c = img[0].shape

    img_enc = encoder(img[0])
    decoder(img_enc, h, w)


if __name__ == '__main__':
    plt.close('all')
    main()

