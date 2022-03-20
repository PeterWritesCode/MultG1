# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
import scipy.fftpack as fft
from numpy import r_


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

        # print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

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
        # print("\nUpsampling with repetitions\n")

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
        plt.title('downsampled')
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

        # print("Upsampling with interpolation")

        fig = plt.figure(figsize=(20, 20))
        if num == 420:
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=stepY,
                               interpolation=cv2.INTER_LINEAR)
        elif num == 422:  # coisa bonita
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=1,
                               interpolation=cv2.INTER_LINEAR)
        fig.add_subplot(1, 2, 1)
        plt.imshow(dsImg)
        plt.title('downsampled')
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

        # print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

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


def DCT(img_in):
    w, h, c = img_in.shape
    img_out = np.zeros((w,h, 3))
    for i in range(c):
        img = img_in[:, :, i]
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    
        dctImg = dct2(img)
        print(dctImg)
        img_out[:,:,i] = dctImg
        dctLogImg = np.log(np.abs(dctImg) + 0.0001)
    
        fig = plt.figure(figsize=(20, 20))
        fig.add_subplot(1, 2, 1)
        plt.imshow(dctImg, cm_grey)
        plt.title('DCT')
        plt.axis('image')
    
        fig.add_subplot(1, 2, 2)
        plt.imshow(dctLogImg, cm_grey)
        plt.title('DCT log')
        plt.axis('image')
        plt.show()

        
    print(img_out)
    return img_out


def iDCT(img_in):
    w, h, c = img_in.shape
    img_out = np.zeros((w,h, 3), dtype=np.uint8)
    for i in range(c):
        img = img_in[:, :, i]
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
        print(img)
        invDctImg = idct2(img)
    
        plt.figure(figsize=(20, 20))
        
        plt.imshow(invDctImg, cm_grey)
        plt.title('IDCT' + str(i))
        plt.axis('image')
        img_out[:,:,i] = invDctImg
        
    return img_out


def idctBlocks(dctImg, d):
    w, h, c = dctImg.shape
    img_out = np.zeros((w,h,3))
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    for k in range(3):
        idct = np.zeros(dctImg[:,:,k].shape)
        
        dctImg1 = dctImg[:,:,k]
        for i in r_[:w:d]:
            for j in r_[:h:d]:
                idct[i:(i + d), j:(j + d)] = idct2(dctImg1[i:(i + d), j:(j + d)])
        img_out[:,:,k] = idct
        
        
        plt.figure(figsize=(20, 20))
        plt.imshow(idct,cm_grey)
        stringTitle = "IDCT of " + str(d) + "blocks"
        plt.title(stringTitle)
    
    return img_out


def dctBasisImg(img, d):
    imsize = img.shape
    dct = np.zeros(imsize)
    dctLog = np.zeros(dct.shape)
    for i in r_[:imsize[0]:d]:
        for j in r_[:imsize[1]:d]:
            dct[i:(i + d), j:(j + d)] = dct2(img[i:(i + d), j:(j + d)])
            
    dctLog = np.log(np.abs(dct) + 0.0001)
    return dct, dctLog


def dct2(a):
    dctImg = fft.dct(fft.dct(a, norm='ortho').T, norm='ortho').T
    return dctImg


def idct2(dctImg):
    idctImg = fft.idct(fft.idct(dctImg, axis = 0, norm="ortho"),axis = 1, norm="ortho")
    return idctImg


def DCT_block(img_in):
    w, h, c = img_in.shape
    img_out = np.zeros((w,h, 3))
    img_out64 = np.zeros((w, h, 3))
    cmBW = clr.LinearSegmentedColormap.from_list(
        'greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
    for i in range(c):
        img = img_in[:, :, i]
        dct8x8, log8x8 = dctBasisImg(img, 8)
        
        fig = plt.figure(figsize=(20, 20))
        fig.add_subplot(1,2,1)
        plt.imshow(log8x8, cmBW)
        plt.title("Log of 8x8 DCT")
        
        fig.add_subplot(1,2,2)
        plt.imshow(dct8x8, cmBW, vmax=np.max(dct8x8) * 0.01, vmin=0)
        plt.title("8x8 DCTs of the image")
        
        dct64x64,log64x64 = dctBasisImg(img, 64)
        
        fig = plt.figure(figsize=(20, 20))
        fig.add_subplot(1,2,1)
        plt.imshow(log64x64, cmBW)
        plt.title("Log of 64x64 DCT")
        
        fig.add_subplot(1,2,2)
        plt.imshow(dct64x64, cmBW, vmax=np.max(dct8x8) * 0.01, vmin=0)
        plt.title("64x64 DCTs of the image")
       
        img_out[:,:,i] = dct8x8
        img_out64[:,:,i] = dct64x64
        
    return img_out, img_out64

def getMatrix(i, qFactor):
    LumMatrix =  np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                           [12, 12, 14, 19,  26,  58,  60,  55],
                           [14, 13, 16, 24,  40,  57,  69,  56],
                           [14, 17, 22, 29,  51,  87,  80,  62],
                           [18, 22, 37, 56,  68, 109, 103,  77],
                           [24, 35, 55, 64,  81, 104, 113,  92],
                           [49, 64, 78, 87, 103, 121, 120, 101],
                           [72, 92, 95, 98, 112, 100, 103,  99]])
    CromMatrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                           [18, 21, 26, 66, 99, 99, 99, 99],
                           [24, 26, 56, 99, 99, 99, 99, 99],
                           [47, 66, 99, 99, 99, 99, 99, 99],
                           [99, 99, 99, 99, 99, 99, 99, 99],
                           [99, 99, 99, 99, 99, 99, 99, 99],
                           [99, 99, 99, 99, 99, 99, 99, 99],
                           [99, 99, 99, 99, 99, 99, 99, 99]])
    
    if(qFactor == 100):
        matrix = np.ones(LumMatrix.shape)
        return matrix
    
    if(i == 0):
        matrix = LumMatrix
    else:
        matrix = CromMatrix
    
    matrix = np.round(matrix/((qFactor)/50))
    return matrix
    
    
def reverseQuantization(img,qFactor):
    w,h,c = img.shape
    img_out = np.zeros((w,h,3))
    cmBW = clr.LinearSegmentedColormap.from_list(
        'greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
    for k in range(3):
        unQuantizedImg = np.zeros(img[:,:,k].shape)
        matrix = getMatrix(k,qFactor)
        imgChannel = img[:,:,k]
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                bocadinho = imgChannel[i:(i + 8), j:(j + 8)]
                
                bocadinho = np.round(bocadinho*matrix)
                unQuantizedImg[i:(i + 8), j:(j + 8)] = bocadinho
                
        img_out[:,:,k] = unQuantizedImg
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np.log(np.abs(unQuantizedImg)+0.0001), cmBW)
        stringTitle = "Channel:" + str(k) + " UnQuantized with Quality of: " + str(qFactor)
        plt.title(stringTitle)
    return img_out
        
    
    
def quantization(img, qFactor):
    w,h,c = img.shape
    img_out = np.zeros((w,h,3))
    cmBW = clr.LinearSegmentedColormap.from_list(
        'greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
    for k in range(3):
        quantizedImg = np.zeros(img[:,:,k].shape)
        matrix = getMatrix(k,qFactor)
        imgChannel = img[:,:,k]
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                bocadinho = imgChannel[i:(i + 8), j:(j + 8)]
                
                bocadinho = np.round(bocadinho/(matrix))
                quantizedImg[i:(i + 8), j:(j + 8)] = bocadinho
                
        img_out[:,:,k] = quantizedImg
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np.log(np.abs(quantizedImg)+0.0001), cmBW)
        stringTitle = "Channel:" + str(k) + " Quantized with Quality of: " + str(qFactor)
        plt.title(stringTitle)
    return img_out

def DCdpcpm(img):
    w,h,c = img.shape
    img_out = np.zeros((w,h,3))
    cmBW = clr.LinearSegmentedColormap.from_list(
        'greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
    for k in range(3):
        quantizedImg = np.zeros(img[:,:,k].shape)
        matrix = getMatrix(k,qFactor)
        imgChannel = img[:,:,k]
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                bocadinho = imgChannel[i:(i + 8), j:(j + 8)]

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
    cmgreen = clr.LinearSegmentedColormap.from_list(
        'mygreen', colorlistGreen, N=256)
    cmblue = clr.LinearSegmentedColormap.from_list(
        'myblue', colorlistBlue, N=256)

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
    cbcr[:, :, 1] = 128 + tc[1][0] * floatR + \
        tc[1][1] * floatG + tc[1][2] * floatB
    # Cr
    cbcr[:, :, 2] = 128 + tc[2][0] * floatR + \
        tc[2][1] * floatG + tc[2][2] * floatB

    transcol = cbcr

    colorlistgray = ["black", (0.5, 0.5, 0.5)]

    cmgray = clr.LinearSegmentedColormap.from_list(
        'myclrmap', colorlistgray, N=256)

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

    return cbcr


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

    R = Y * tc_invertida[0][0] + (Cb - 128) * \
        tc_invertida[0][1] + (Cr - 128) * tc_invertida[0][2]
    G = Y * tc_invertida[1][0] + (Cb - 128) * \
        tc_invertida[1][1] + (Cr - 128) * tc_invertida[1][2]
    B = Y * tc_invertida[2][0] + (Cb - 128) * \
        tc_invertida[2][1] + (Cr - 128) * tc_invertida[2][2]

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
    cmred_rev = clr.LinearSegmentedColormap.from_list(
        'myred', colorlistRed[::-1], N=256)
    cmgreen_rev = clr.LinearSegmentedColormap.from_list(
        'mygreen', colorlistGreen[::-1], N=256)
    cmblue_rev = clr.LinearSegmentedColormap.from_list(
        'myblue', colorlistBlue[::-1], N=256)

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


def encoder(img, tc, ratio):
    print('Encoding image')
    # 2
    img = visualizacao(img)
    # 4
    img = padding(img)
    # 5
    cbcr = RGB2YCbCr(img, tc)

    cbcr_inter = downsample_inter(cbcr, ratio)

    cbcr = downsample_nointer(cbcr, ratio)

    imgDct = DCT(cbcr)
    

    dct8x8, dct64x64 = DCT_block(cbcr)
    quantizedImg = quantization(dct8x8,95)
    reverse = reverseQuantization(quantizedImg,95)
    idctBlocks(reverse,8)
    
    return img, cbcr, cbcr_inter, dct64x64, dct8x8, imgDct


def decoder(img, cbcr, cbcr_inter, dct64x64, dct8x8, h, w, tc, ratio):
    print('Decoding image')
    
    idctBlocks(dct64x64, 64)
    idctBlocks1 = idctBlocks(dct8x8,8)
    
    comparison = cbcr == idctBlocks1
    
    cbcr_inter = upsample_inter(cbcr_inter, ratio)

    cbcr = upsample_rep(cbcr, ratio)
    
    
    
    print(comparison)
    
    idctBlocks3 = upsample_inter(idctBlocks1,ratio)
    
    # 5
    img = YCbCr2RGb(idctBlocks3, tc)
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

    dsusRatio = 420
    img_enc, cbcr, cbcr_inter,dct64x64, dct8x8, imgDct = encoder(img_in, tc, dsusRatio)
    img_dec = decoder(img_enc, cbcr, cbcr_inter,dct64x64, dct8x8, h, w, tc, dsusRatio)
    # comparison = img[1] == img_dec
    # print(comparison.all())
    # print(img[2], " \n A \n", img_dec)


if __name__ == '__main__':
    plt.close('all')
    main()
