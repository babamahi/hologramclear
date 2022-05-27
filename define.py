import sys
from PIL import Image
import numpy as np
import custom_layers as cl
import cv2
#テキストファイル読み込み
def textfileload(filepath):
    mat = []
    with open(filepath,'r',encoding='utf-8') as fin:
        for line in fin.readlines():
            row = []
            toks = line.split(' ')
            for tok in toks:
                try:
                    num =float(tok)
                except ValueError as e:
                    print(e,file=sys.stderr)
                    continue
                row.append(num)
            mat.append(row)
    return mat

#光強度調整，intensity:光強度，std_dev：標準偏差倍率　　正規分布以外は0に
##for 文　時間かかる
def hist(intensity,height,width,std_dev):
    i_ave = np.mean(intensity)
    print(i_ave)
    i_stdv = np.std(intensity)
    print(i_stdv)
    for l in range(height):
        for m in range(width):
            if i_ave+(-std_dev)*i_stdv <= intensity[l][m] and intensity[l][m]<=  i_ave+(std_dev)*i_stdv:
                intensity[l][m] = intensity[l][m]
            else:
                intensity[l][m] = 0
    return intensity

##for文の代わりにnp.whereを使用
def hist2(intensity,std_dev):
    i_ave = np.mean(intensity)
    i_stdv = np.std(intensity)
    intensity = np.where((intensity>=i_ave+(std_dev)*i_stdv),i_ave+(std_dev)*i_stdv,intensity)
    print(i_ave+(-std_dev)*i_stdv)
    print(i_ave+(std_dev)*i_stdv)
    return intensity

##最小の条件も追加
def hist3(intensity,std_dev):
    i_ave = np.mean(intensity)
    i_stdv = np.std(intensity)
    intensity = np.where((intensity>=i_ave+(std_dev)*i_stdv),i_ave+(std_dev)*i_stdv,intensity)
    intensity = np.where((intensity<=i_ave+(-std_dev)*i_stdv),i_ave+(-std_dev)*i_stdv,intensity)
    print(i_ave+(-std_dev)*i_stdv)
    print(i_ave+(std_dev)*i_stdv)
    return intensity

#256階調化，PILLOWで画像保存ただし，for文で遅い
def gradation(Intensity,height,width):
    max = 0
    min = Intensity[0][0]
    for i in range(height):
        for j in range(width):
            if (Intensity[i][j]>max):
                max = Intensity[i][j]
            if (Intensity[i][j]<min):
                min = Intensity[i][j]
    for i in range(height):
        for j in range (width):
            Intensity[i][j] = 255*(Intensity[i][j]-min)/(max-min)
            if Intensity[i][j]>255:
                Intensity[i][j] = 255
    print(Intensity.max())
    print(Intensity.min())
    return Intensity

##for文使わない
def gradation2(Intensity):
    Intensity = 255*(Intensity-np.min(Intensity))/(np.max(Intensity)-np.min(Intensity))
    return Intensity

def Imagesave(array,savepath):
    im = Image.fromarray(array.astype(np.uint8))
    return im.save(savepath)