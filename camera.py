import cv2
import rawpy
import imageio
import PIL
import numpy as np 
import pandas as pd

def CANON_KA (XK,YK): #Lum: 13900 - 1.2 cd
    p00 =       37.06
    p10 =    0.005177
    p01 =    0.005452
    p20 =   2.071e-06
    p11 =   3.156e-06
    p02 =  -1.424e-06
    p30 =  -9.131e-10
    p21 =  -6.061e-10
    p12 =  -7.508e-10
    p40 =    7.02e-14
    p31 =   1.615e-14
    p22 =   1.214e-13

    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K

def CANON_KB (XK,YK):   #Lum:  784 - 0.049 cd
    p00 =       45.88
    p10 =    0.004336
    p01 =    0.005549
    p20 =   3.319e-06
    p11 =   4.947e-06
    p02 =  -1.482e-06
    p30 =  -1.246e-09
    p21 =  -1.028e-09
    p12 =  -1.109e-09
    p40 =    9.41e-14
    p31 =   3.963e-14
    p22 =   1.785e-13
    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K

def NIKON_info (path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
        for key, value in tags.items():
            if key is not 'JPEGThumbnail':  # do not print (uninteresting) binary thumbnail data
                if key== 'Image Make': make=value 
                if key== 'Image Model': model=value 
                if key=='EXIF ExposureTime': exp_time=value
                if key=='EXIF ISOSpeedRatings': ISO=value
                if key=='EXIF FocalLength': FL=value
                if key=='EXIF FNumber': Fs=value
    return make, model, exp_time, ISO,FL,Fs

def dark_D (x,y):
    p00 =       1.554
    p10 =  -1.221e-06
    p01 =   9.852e-05
    p11 =   2.944e-10
    p02 =  -6.546e-08
    p12 =  -3.689e-13
    p03 =   1.214e-11
    D=p00 + p10*x + p01*y + p11*x*y + p02*y**2 + p12*x*y**2 + p03*y**3
    return D

def NIKON_KA (XK,YK): #Lum: 13900 - 1.2 cd
    p00 =       34.01
    p10 =    0.002985
    p01 =    0.002825
    p20 =  -5.746e-07
    p11 =  -3.217e-07
    p02 =  -5.839e-07
    p30 =   1.696e-11
    p21 =   1.143e-10
    p12 =   2.427e-11
    p40 =  -9.058e-16
    p31 =  -8.273e-15
    p22 =   -7.57e-15
    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K

def NIKON_KB (XK,YK):   #Lum:  784 - 0.049 cd
    p00 =       46.74
    p10 =     0.00333
    p01 =     0.00284
    p20 =  -3.784e-07
    p11 =    1.05e-07
    p02 =  -6.079e-07
    p30 =  -8.285e-11
    p21 =    5.85e-11
    p12 =   -7.32e-11
    p40 =   8.756e-15
    p31 =  -1.099e-14
    p22 =   9.125e-15
    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K


def camera_info (path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f)
        for key, value in tags.items():
            if key is not 'JPEGThumbnail':  # do not print (uninteresting) binary thumbnail data
                if key== 'Image Make': make=value 
                if key== 'Image Model': model=value 
                if key=='EXIF ExposureTime': exp_time=value
                if key=='EXIF ISOSpeedRatings': ISO=value
                if key=='EXIF FocalLength': FL=value
                if key=='EXIF FNumber': Fs=value
    return make, model, exp_time, ISO,FL,Fs

def SONY_D (x,y):
    p00 =       1.554
    p10 =  -1.221e-06
    p01 =   9.852e-05
    p11 =   2.944e-10
    p02 =  -6.546e-08
    p12 =  -3.689e-13
    p03 =   1.214e-11
    D=p00 + p10*x + p01*y + p11*x*y + p02*y**2 + p12*x*y**2 + p03*y**3
    return D

def SONY_KA (XK,YK): #Lum: 13900 - 1.2 cd
    p00 =       9.965
    p10 =     0.01936
    p01 =     0.01214
    p20 =  -2.835e-06 
    p11 =   1.738e-06
    p02 =  -2.392e-06
    p30 =   8.097e-11
    p21 =  -2.989e-10
    p12 =  -3.265e-10
    p40 =  -5.408e-15
    p31 =   7.585e-15
    p22 =    4.56e-14
    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K

def SONY_KB (XK,YK):   #Lum:  784 - 0.049 cd
    p00 =       13.64
    p10 =     0.02162
    p01 =     0.01344
    p20 =  -2.931e-06
    p11 =   2.231e-06
    p02 =  -2.653e-06
    p30 =  -8.083e-12
    p21 =  -3.021e-10
    p12 =  -4.884e-10
    p40 =   3.322e-15
    p31 =   -1.34e-15
    p22 =   6.777e-14
    K = p00 + p10*XK + p01*YK + p20*XK**2 + p11*XK*YK + p02*YK**2 + p30*XK**3 + p21*XK**2*YK + p12*XK*YK**2 + p40*XK**4 + p31*XK**3*YK + p22*XK**2*YK**2
    return K

def norm_image (path1):   #Lum:  784 - 0.049 cd
    K_Gauss = cv2.getGaussianKernel(20,5)
    df= pd.read_excel(path1, index_col=0)
    path="/content/cameracalibration/"+df['File RGB'][1]
    print(path)
    raw = rawpy.imread(path)
    rgb16 = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=True, output_bps=16)
    height, width, channels = rgb16.shape
    full_img2_1=np.zeros((height, width),np.float64)
    for n in range(13): #Total de imagnes 13
        path="/content/cameracalibration/"+df['File RGB'][n+1]
        Xm=df['Xm'][n+1]        # 
        Ym=df['Ym'][n+1]           # 
        raw = rawpy.imread(path)
        rgb16 = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=True, output_bps=16)
        R=rgb16[:,:,0]
        G=rgb16[:,:,1]
        B=rgb16[:,:,2]
        Yimg=(0.2162*R)+(0.7152*G)+(0.0722*B)
        #Convolve function
        dst2 = cv2.sepFilter2D(Yimg,-1,K_Gauss,K_Gauss)
        mask=np.zeros((height, width),np.uint8)
        cv2.circle(mask,(Ym,Xm), 120, (1), -1)
        full_img2_1=full_img2_1+(dst2*mask)
    return full_img2_1

def calc_luminance (path1,fs,ISO,TE,full_img,cam,reference):
    df= pd.read_excel(path1, index_col=0)
    path="/content/cameracalibration/"+df['File RGB'][1]
    print(path)
    raw = rawpy.imread(path)
    rgb16 = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=True, output_bps=16)
    height, width, channels = rgb16.shape
    YK = np.linspace(0, height, height)
    XK = np.linspace(0, width, width)
    XK, YK = np.meshgrid(XK, YK)
    if cam=='C1':
        K1=CANON_KB (XK,YK)
    if cam=='C2':
        K1=NIKON_KB (XK,YK)
    if cam=='C3':
        K1=SONY_KB (XK,YK)    
    aux=np.ones((height, width),np.float64)
    aux1=aux*((fs**2)/(TE*ISO*K1))
    L1=full_img*aux1
    Px=11  # Cantidad de pixeles para el superpixeles
    nn=8 #numero de megapixeles
    aux=0
    df1 = pd.DataFrame(columns=['Camera','Region','Luminance', 'Reference'])
    for l in range(13): #Total de imagnes 13
        Xm=df['Xm'][l+1]        
        Ym=df['Ym'][l+1]          
        LR_f=L1[Xm-(Px*6):Xm+(Px*2),Ym-(Px*2):Ym+(Px*6)]
        for n in range(nn):
            for m in range(nn):
                auxx_f=LR_f[n*Px:(n+1)*Px,m*Px:(m+1)*Px]
                Average_f=np.nanmean(np.where(auxx_f!=0,auxx_f,np.nan))      
                df1.loc[aux] = [cam,l+1,Average_f,reference]
                aux=aux+1
    return(df1)