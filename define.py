import sys
from PIL import Image
import numpy as np
import custom_layers as cl
import cv2
import os

##テキストファイル読み込み##
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

### バイナリファイル読み込み(x,y,z,ampl)##
def readPClbin(fname, readHeader = True):
    file = open(fname, "rb");
    if(readHeader):
        npts = np.fromfile(file, dtype=np.uint64, count=1)[0]
    else:
        npts = np.int64(os.path.getsize(fname)/(4*4))

    print("Reading {} points.".format(npts))
    
    pcl = np.zeros((npts, 4), dtype=np.single)
    for ptid in range(npts):
        pcl[ptid, :] = np.fromfile(file, dtype=np.single, count=4)
        
    return pcl, npts


####光強度調整，intensity:光強度，std_dev：標準偏差倍率　　正規分布以外は0に###
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

##### スレッショルド処理 #####
##for文の代わりにnp.whereを使用
def hist2(intensity,std_dev):
    i_ave = np.mean(intensity)
    i_stdv = np.std(intensity)
    intensity = np.where((intensity>=i_ave+(std_dev)*i_stdv),i_ave+(std_dev)*i_stdv,intensity)
    #print(i_ave+(-std_dev)*i_stdv)
    #print(i_ave+(std_dev)*i_stdv)
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

##for文使わないですべてnumpyで計算
def gradation2(Intensity):
    Intensity = 255*(Intensity-np.min(Intensity))/(np.max(Intensity)-np.min(Intensity))
    return Intensity

def Imagesave(array,savepath):
    im = Image.fromarray(array.astype(np.uint8))   
    return im.save(savepath)

### 視域の定義関数
def visual_range(pp,wl, z):
    t = np.arcsin(wl/(2*pp))
    dx = z * np.tan(t)
    return dx

### ホログラムを分割して保存する関数
def holosplit(data, original_path,path,original_height,height):
    
    # data：データ数(訓練用、検証用データの数)
    # original_path ：　元ホログラムのディレクトリパス
    # path ：　分割ホログラムのディレクトリパス
    # original_height ：　元のホログラムの縦サイズ
    # height ：　分割したホログラムのサイズ
    
    j = 0
    
    n = int(original_height / height)    ##ホログラムの縦横の分割数
    
    for i in range(data):
        hologram = np.load(original_path + 'holo' + str(i) +'.npy')
        for h1 in range(n):
            for w1 in range(n):
                w2 = w1 * height
                h2 = h1 * height
                holo = hologram[h2 : height + h2, w2 : height + w2]
                holo = np.array(holo)
                split_path = path + 'holo' + str(i) +'/'+ 'split' + str(j) + '.npy'
                np.save(split_path, holo)
                j += 1
                if j >= n**2:
                    j = j - n**2      


## 分割したホログラムを結合する関数                    
def concatenate_matrix(height, width, original_height, original_width, k,path):
    
    # height,width ：　分割ホログラムのサイズ
    # original_height, original_width：元ホログラムのサイズ
    # k ： k番目のホログラム
    # path ：　分割ホログラムのパス
    
    n = int(original_width/width)
    ### 0の配列を作成 →　形状は(original_height, original_width, 2)
    new_holo = np.zeros((original_height,original_width,2), dtype = np.float32)
    h = 0
    w = 0
    for i in range(int(n*n)):
        split_path = path + 'holo' + str(k) + '/' + 'split' + str(i) + '.npy'
        array_split = np.load(split_path)
        new_holo[h * height : height + h * height, w * width : width + w * width] = array_split
        w += 1
        if w % int(n) == 0:
            w = w - int(n)
            h += 1
    
    return new_holo


### PSNR ###
def psnr(img_1, img_2, data_range=255):
    mse = np.mean((img_1.astype(float)- img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


### フォルダにあるファイルを消去する ####
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#files = sorted(glob.glob('data/*.jpg'), key=natural_keys)
def remove_glob(path, recursive=True):
    for p in glob.glob(path, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)