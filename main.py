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
    
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    if num == 420:
        img_out = np.zeros((w//2, h//2, 2))
    elif num == 422:
        img_out = np.zeros((w, h//2, 2))
    for i in range(2):
        img = img_in[:, :, i]
        stepX = 2
        stepY = 2
        if num == 420:  # coisa bonita
            dsImgInterp = cv2.resize(img, None, fx=1 / stepX, fy=1 / stepY, interpolation=cv2.INTER_LINEAR)

        elif num == 422:  # coisa bonita
            dsImgInterp = cv2.resize(img, None, fx=1 / stepX, fy=1, interpolation=cv2.INTER_LINEAR)

        # print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

        plt.imshow(dsImgInterp,cm_grey)
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
    
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    if num == 420:
        img_out = np.zeros((w * 2, h * 2, 2))
    elif num == 422:
        img_out = np.zeros((w, h*2, 2))
    for i in range(2):
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
        plt.imshow(dsImg,cm_grey)
        plt.title('downsampled')
        plt.axis('image')

        fig.add_subplot(1, 2, 2)
        plt.imshow(usImg,cm_grey)
        plt.title('upsampled with repetitions')
        plt.axis('image')
        plt.show()

        # print("dsImg size = ", dsImg.shape)
        # print("usImg size = ", usImg.shape)
        img_out[:, :, i] = usImg

    return img_out


def upsample_inter(img_in, num):
    w, h, c = img_in.shape
    
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    if num == 420:
        img_out = np.zeros((w * 2, h * 2, 2))
    elif num == 422:
        img_out = np.zeros((w , h* 2, 2))
    for i in range(2):
        dsImg = img_in[:, :, i]
        stepX = 2
        stepY = 2

        # print("Upsampling with interpolation")


        if num == 420:
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=stepY,
                               interpolation=cv2.INTER_LINEAR)
        elif num == 422:  # coisa bonita
            usImg = cv2.resize(dsImg, None, fx=stepX, fy=1,
                               interpolation=cv2.INTER_LINEAR)

        plt.imshow(usImg,cm_grey)
        plt.title('upsampled with interpolation')
        plt.axis('image')
        plt.show()

        # print()
        # print("dsImg size = ", dsImg.shape)
        # print("usImg size = ", usImg.shape)
        img_out[:, :, i] = usImg

    return img_out


def downsample_nointer(img_in, num):
    w, h, c = img_in.shape
    
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    if num == 420:
        img_out = np.zeros((w//2, h//2, 2))
    elif num == 422:
        img_out = np.zeros((w, h//2, 2))
    for i in range(2):
        img = img_in[:, :, i]
        stepX = 2
        stepY = 2

        # print("\nDownsampling 4:2:0 using openCv with interpolation filter\n")

        if num == 420:  # coisa bonita
            dsImg = img[::stepX, ::stepY]

        elif num == 422:  # coisa bonita
            dsImg = img[:, ::stepX]

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img,cm_grey)
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
        plt.imshow(dsImg,cm_grey)
        if num == 420:  # coisa bonita
            plt.title('downsampled 4:2:0 sx = 0.5, sy = 0.5  no interp')

        elif num == 422:  # coisa bonita
            plt.title('downsampled 4:2:2 sx = 0.5 no interp')
        plt.axis('image')
        img_out[:, :, i] = dsImg

    return img_out


def DCT(yin, cbcrin):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for i in range(3):
        if(i==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
        else:
            img_in = cbcrin
            img = img_in[:, :, i]
            w, h, c = img_in.shape
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
    
        dctImg = dct2(img)
        # print(dctImg)
        if(i==0):
            yout = dctImg 
        else:
            cbcrout[:,:,i-1] = dctImg
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

        
    
    return yout, cbcrout


def iDCT(yin, cbcrin):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for i in range(3):
        if(i==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
        else:
            img_in = cbcrin
            img = img_in[:, :, i]
            w, h, c = img_in.shape
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
        # print(img)
        invDctImg = idct2(img)
        
        if(i==0):
            yout = invDctImg 
        else:
            cbcrout[:,:,i-1] = invDctImg
        plt.figure(figsize=(20, 20))
        
        plt.imshow(invDctImg, cm_grey)
        plt.title('IDCT' + str(i))
        plt.axis('image')
        
    return yout,cbcrout


def idctBlocks(yin,cbcrin, d):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for k in range(3):
        if(k==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
            idct = np.zeros((w,h))
            
        else:
            img_in = cbcrin
            img = img_in[:, :, k-1]
            w, h, c = img_in.shape
            idct1 = np.zeros((w,h))
            
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
        
        
        dctImg1 = img
        
        
        
        if(k==0):
            for i in r_[:w:d]:
                for j in r_[:h:d]:
                    idct[i:(i + d), j:(j + d)] = idct2(dctImg1[i:(i + d), j:(j + d)])
            yout = idct 
        else:
            for i in r_[:w:d]:
                for j in r_[:h:d]:
                    idct1[i:(i + d), j:(j + d)] = idct2(dctImg1[i:(i + d), j:(j + d)])
            cbcrout[:,:,k-1] = idct1
        
        
        print("IDCT ---------------------------------------------------- \n" + str(idct))
        plt.figure(figsize=(20, 20))
        plt.imshow(idct,cm_grey)
        stringTitle = "IDCT of " + str(d) + "blocks"
        plt.title(stringTitle)
    
    return yout,cbcrout


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
    idctImg = fft.idct(fft.idct(dctImg, norm="ortho").T, norm="ortho").T
    return idctImg


def DCT_block(yin, cbcrin):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    cbcrout64 = np.zeros((w,h,2))
    for k in range(3):
        if(k==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
            yout = np.zeros((w,h))
            yout64 = np.zeros((w,h))
            
        else:
            img_in = cbcrin
            img = img_in[:, :, k-1]
            w, h, c = img_in.shape
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)

        dct8x8, log8x8 = dctBasisImg(img, 8)
        
        fig = plt.figure(figsize=(20, 20))
        fig.add_subplot(1,2,1)
        plt.imshow(log8x8, cm_grey)
        plt.title("Log of 8x8 DCT")
        
        fig.add_subplot(1,2,2)
        plt.imshow(dct8x8,  cm_grey , vmax=np.max(dct8x8) * 0.01, vmin=0)
        plt.title("8x8 DCTs of the image")
        
        dct64x64,log64x64 = dctBasisImg(img, 64)
        
        fig = plt.figure(figsize=(20, 20))
        fig.add_subplot(1,2,1)
        plt.imshow(log64x64,  cm_grey )
        plt.title("Log of 64x64 DCT")
        
        fig.add_subplot(1,2,2)
        plt.imshow(dct64x64,  cm_grey , vmax=np.max(dct8x8) * 0.01, vmin=0)
        plt.title("64x64 DCTs of the image")
       
        if(k==0):
            yout = dct8x8
            yout64 = dct64x64
        else:
            cbcrout[:,:,k-1] = dct8x8
            cbcrout64[:,:,k-1] = dct64x64
        
    return yout, yout64, cbcrout, cbcrout64

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
        matrix1 = np.ones(LumMatrix.shape)
        return matrix1
    
    if(i == 0):
        matrix1 = LumMatrix
    else:
        matrix1 = CromMatrix
        
    if (qFactor < 50):
        
        S = 50/qFactor
    else:
        S = (100 - qFactor)/50
        

    matrix = np.round(S*matrix1)
    matrix[matrix > 255] = 255
    matrix[matrix < 1] = 1
    
    
   
    print(matrix)
    return matrix
    
    
def reverseQuantization(yin,cbcrin,qFactor):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for k in range(3):
        if(k==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
            unQuantizedImg = np.zeros(img.shape)
        else:
            img_in = cbcrin
            img = img_in[:, :, k-1]
            w, h, c = img_in.shape
            unQuantizedImg = np.zeros((w,h))
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
       
        matrix = getMatrix(k,qFactor)
        imgChannel = img[:,:]
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                bocadinho = imgChannel[i:(i + 8), j:(j + 8)]
                
                bocadinho = np.round(bocadinho*matrix)
                unQuantizedImg[i:(i + 8), j:(j + 8)] = bocadinho
                
        if(k==0):
            yout = unQuantizedImg

        else:
            cbcrout[:,:,k-1] = unQuantizedImg
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np.log(np.abs(unQuantizedImg)+0.0001),cm_grey)
        stringTitle = "Channel:" + str(k) + " UnQuantized with Quality of: " + str(qFactor)
        plt.title(stringTitle)
    return  yout,cbcrout
        
    
    
def quantization(yin, cbcrin, qFactor):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for k in range(3):
        if(k==0):
            img = yin
            w, h = img.shape
            yout = np.zeros((w,h))
            QuantizedImg = np.zeros(img.shape)
            imgChannel = img
            
        else:
            img_in = cbcrin
            img = img_in[:, :, k-1]
            w, h, c = img_in.shape
            QuantizedImg = np.zeros((w,h))
            imgChannel = img[:,:]
        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
        
        matrix = getMatrix(k,qFactor)
        
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                bocadinho = imgChannel[i:(i + 8), j:(j + 8)]
                
                bocadinho = np.round(bocadinho/matrix)
                QuantizedImg[i:(i + 8), j:(j + 8)] = bocadinho
                
        if(k==0):
            yout = QuantizedImg

        else:
            cbcrout[:,:,k-1] = QuantizedImg
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np.log(np.abs(QuantizedImg)+0.0001),cm_grey)
        stringTitle = "Channel:" + str(k) + " Quantized with Quality of: " + str(qFactor)
        plt.title(stringTitle)
        
    return  yout,cbcrout

def DCdpcm(yin, cbcrin, flag):
    w, h, c = cbcrin.shape
    cbcrout = np.zeros((w,h,2))
    for k in range(3):
        if(k==0):
            imgChannel = yin
            w, h = imgChannel.shape
            yout = np.zeros((w,h))

            
        else:
            img_in = cbcrin
            w, h, c = cbcrin.shape
            imgChannel = img_in[:, :, k-1]
            

        cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)
        
        for i in r_[:w:8]:
            for j in r_[:h:8]:
                if(i ==0 and j==0):
                    dc0 = imgChannel[0,0]
                    
                    print(dc0)
                    continue
                dci = imgChannel[i,j]
                if(flag):
                    di = dci - dc0
                else:
                    di= dci+dc0
                #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" + str(di))
                
                imgChannel[i,j] = di
                if(flag):
                    dc0 = dci
                else:
                    dc0 = di
        
        if(k==0):
            yout = imgChannel

        else:
            cbcrout[:,:,k-1] = imgChannel
        
        plt.figure(figsize=(20, 20))
        plt.imshow(np.log(np.abs(imgChannel)+0.0001),cm_grey)
        if(flag):
            stringTitle = "Channel:" + str(k) + " DPCM'd "
        else:
            stringTitle = "Channel:" + str(k) + " un- DPCM'd "
        
        plt.title(stringTitle)
        
    return yout,cbcrout


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
    # print(transcol.shape)
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
    # print(tc_invertida)

    R = Y * tc_invertida[0][0] + (Cb - 128) * \
        tc_invertida[0][1] + (Cr - 128) * tc_invertida[0][2]
    G = Y * tc_invertida[1][0] + (Cb - 128) * \
        tc_invertida[1][1] + (Cr - 128) * tc_invertida[1][2]
    B = Y * tc_invertida[2][0] + (Cb - 128) * \
        tc_invertida[2][1] + (Cr - 128) * tc_invertida[2][2]

    rgb = img.astype(float)
    # print(R[:8, :8])
    # print("-----------------------")

    rgb = invRGB(R, G, B, img.shape)
    rgb = rgb.round()

    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0
    rgb = np.uint8(rgb)

    # print(tc_invertida.T)

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
    # print(img.shape)
    # print("padding")
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
    # print(img.shape)

    return img


def getImageOriginal(img, height, width):
    img = img[0:height, 0:width]

    plt.figure()
    plt.imshow(img)
    print(img.shape)
    plt.axis('off')
    plt.show()

    return img

def encoder(img, tc, ratio, qFactor):
    print('Encoding image')
    # 2
    img = visualizacao(img)
    # 4
    img = padding(img)
    # 5
    cbcr = RGB2YCbCr(img, tc)
    
    w,h,c = cbcr.shape
    imgY = np.zeros((w,h))
    imgNew = np.zeros((w,h,2))
    imgY = cbcr[:,:,0]
    imgNew[:,:,0] = cbcr[:,:,1]
    imgNew[:,:,1] = cbcr[:,:,2]
    cbcrNew = downsample_nointer(imgNew, ratio)
    

    yout, yout64, cbcrout, cbcrout64 = DCT_block(imgY,cbcrNew)

    qyout, qcbcrout = quantization(yout,cbcrout, qFactor)

    dpcmy, dpcmcbcr = DCdpcm(qyout,qcbcrout, True)

    return  yout64,cbcrout64, dpcmy, dpcmcbcr


def decoder( yout64,cbcrout64, dpcmy, dpcmcbcr, h, w, tc, ratio, qFactor):
    print('Decoding image')

    dpcmIy, dpcmIcbcr = DCdpcm(dpcmy,dpcmcbcr,False)
    rDpcmy, rDpcmcbcr = reverseQuantization(dpcmIy, dpcmIcbcr, qFactor)

    blockedOuty, blockedOutcbcr = idctBlocks(rDpcmy, rDpcmcbcr, 8)


    cbcrInter = upsample_rep(blockedOutcbcr, ratio)
    w1,h1,c = cbcrInter.shape
    idctBlocks3 = np.zeros((w1,h1,3))
    
    idctBlocks3[:,:,0] = blockedOuty
    idctBlocks3[:,:,1] = cbcrInter[:,:,0]
    idctBlocks3[:,:,2] = cbcrInter[:,:,1]

    # 5
    img = YCbCr2RGb(idctBlocks3, tc)
    # 4
    img = getImageOriginal(img, h, w)
    # 3
    img = getImage_inv(img)

    return img

def metrics(img,img_rec):    
    # Calculation of Mean Squared Error (MSE)
    mse = np.square(np.subtract(img,img_rec)).mean()
    rmse = np.sqrt(mse)
    
    #snr = signaltonoise(img_rec, axis=0,ddof=0)
    P = np.sum(np.square(img))/np.prod(img.shape)
    snr = 10*np.log10(P/mse)
    
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / rmse)
    
    print("mse:",mse)
    print("rmse:",rmse)
    print("snr:",snr)
    print("psnr:",psnr)
    
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
    qFactor = 100
    dsusRatio = 420
    cbcr_inter, dct64x64, imgDct, dpcmImg = encoder(img_in, tc, dsusRatio, qFactor)
    img_dec = decoder(cbcr_inter, dct64x64, imgDct, dpcmImg, h, w, tc, dsusRatio, qFactor)
    
    metrics(img[k],img_dec)
    # imgFinal = img_in - img_dec
    # cmBW = clr.LinearSegmentedColormap.from_list(
    #     'greyMap', [(0, 0, 0), (1., 1., 1.)], 256)
    # plt.figure()
    # plt.imshow(imgFinal, cmBW)
    # plt.axis('off')
    # plt.show()
    # comparison = img[1] == img_dec
    # print(comparison.all())
    # print(img[2], " \n A \n", img_dec)

if __name__ == '__main__':
    plt.close('all')
    main()
