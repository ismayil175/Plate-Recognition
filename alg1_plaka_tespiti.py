import os #verileri diskten okumaya saglar
import cv2 #görüntü işlememize sağlar
import matplotlib.pyplot as plt #resimleri görselleştirmek için detaylı incelemek için kullanılan kütbane
import numpy as np #list işlemleri daha hızlı ve kolaylıkla yapmamızı sağlamaktadır

foto_adress = os.listdir("dataset") #verisetin altındakı her şeyi getirecektir

fot = cv2.imread("dataset/"+foto_adress[0]) #0. adresteki fotoyu okuycaktır
fot = cv2.resize(fot,(500,500))  

plt.imshow(cv2.cvtColor(fot,cv2.COLOR_BGR2RGB)) #bgr dan rgb ye dönüşüm işlemini gerçekleştiricektir
plt.show()

fot_bgr = fot
fot_gray = cv2.cvtColor(fot,cv2.COLOR_BGR2GRAY) #gray kullanarak daha hızlı işlem yapmamızı ve kolay kullanım sağlamaktatır
plt.imshow(fot_bgr,cmap="gray")
plt.show()

ir_fot = cv2.medianBlur(fot_gray,5) #5x5 boyutta çalıştırılacak
ir_fot = cv2.medianBlur(ir_fot,5) #gürültüden dahada kurtulucaz
plt.imshow(fot_gray,cmap="gray")
plt.show()

mediann = np.median(ir_fot) #yoğunluk merkezini almaktadır
low = 0.67 *mediann #alt yoğunluk merkezi
high = 1.33*mediann #üst yoğunluk merkezi

kenarlik = cv2.Canny(ir_fot,low,high) #yogunluk merkezi alarak kenarlık tespiti yaptık
plt.imshow(kenarlik,cmap="gray")
plt.show()

kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1) #dikdörtgenleri daha iyi yakalayabilme olanağı sağlamaktır iteration ise kaç deefa genişletsin anlamında
plt.imshow(kenarlik,cmap="gray")
plt.show()

cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #Bu sayesinde dikdörgeni bulmuş olacaz
cnt = cnt[0]
cnt = sorted(cnt,key=cv2.contourArea,reverse=True) #en buyuk 20 alanı alıcak

H,W = 500,500
plate = None

for c in cnt:
    rect = cv2.minAreaRect(c) #dikdortgen yapıda al
    (x,y),(w,h),r = rect #merkez noktanın kordinatorlarını 
    if(w>h and w>h*2)or (h>w and h>w*2): #oranen az 2
        box = cv2.boxPoints(rect) #dikdorkenlerin yerini gostericektir
        box = np.int64(box) #tam sayiları döndürücektir
        
        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])

        
        m_plate = fot_gray[miny:maxy,minx:maxx].copy()
        m_mediann = np.median(m_plate)
        
        kontrol1 = m_mediann>84 and m_mediann<250 # yogunluk kontrolu (3)
        kontrol2 = h<50 and w<150 #sınır kontrolu (4)
        kontrol3 = w<50 and h<150 #sınır kontrolu (4)
        print(f"m_plate mediann:{m_mediann} width: {w} height:{h}")
        plt.figure()
        kon=False
        if(kontrol1 and (kontrol2 or kontrol3)):
            #bu kon lardan 1 tanesi dogru olursa platedir
            
            cv2.drawContours(fot,[box],0,(0,255,0),2) #kare çizdirme
            
            plate =[int(i) for i in [minx,miny,w,h]]#x,y,w,h sol üst köşe`yi göstercek
            
            plt.title("plate detected!")
            kontrol=True
        else:
            #plate değidir
            cv2.drawContours(fot,[box],0,(0,0,255),2)#kare çizdirme.
            plt.title("plate not detected !")
        
        plt.imshow(cv2.cvtColor(fot,cv2.COLOR_BGR2RGB)) #ekranda gösterme.
        plt.show()
        if(kontrol):
            break


def plate_konum_don(fot):
    fot_bgr = fot
    fot_gray = cv2.cvtColor(fot,cv2.COLOR_BGR2GRAY)

    ir_fot = cv2.medianBlur(fot_gray,5) #5x5
    ir_fot = cv2.medianBlur(ir_fot,5) #5x5

    mediann = np.median(ir_fot)

    low = 0.67*mediann
    high = 1.33*mediann

    kenarlik = cv2.Canny(ir_fot,low,high)


 
    kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)
    
    cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt,key=cv2.contourArea,reverse=True)

    H,W = 500,500
    plate = None

    for c in cnt:
        rect = cv2.minAreaRect(c) #dikdortgen yapıda al (1)
        (x,y),(w,h),r = rect
        if(w>h and w>h*2) or (h>w and h>w*2):#oran en az 2 (2)
            box = cv2.boxPoints(rect) #[[12,13],[25,13],[20,13],[13,45]]
            box = np.int64(box)

            minx = np.min(box[:,0])
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])


            m_plate = fot_gray[miny:maxy,minx:maxx].copy()
            m_mediann = np.median(m_plate)

            
            kon1 = m_mediann>84 and m_mediann<200 # yogunluk kontrolu (3)
            kon2 = h<50 and w<150 #sınır kontrolu (4)
            kon3 = w<50 and h<150 #sınır kontrolu (4)

            print(f"m_plate mediann:{m_mediann} genislik: {w} yukseklik:{h}")

            kon=False
            if(kon1 and (kon2 or kon3)):
                #plate'dır
                
                
                plate =[int(i) for i in [minx,miny,w,h]]#x,y,w,h
                kon=True
            else:
                #plate değidir
            
                pass
            if(kontrol):
                return plate
    return []   

           

    

