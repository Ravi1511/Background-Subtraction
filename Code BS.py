import cv2
#from PIL import Image
#import PIL
import numpy as np
import math



def gauss(m,v,x):
    pi = 3.1415926
    d = (2*pi*v)**.5
    n = math.exp(-((float(x)-float(m))**2)/(2*v))
    return n/d

capture = cv2.VideoCapture('umcp.mpg')
ret,frame = capture.read()
pix = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
w,h = pix.shape

video1 = cv2.VideoWriter('fg81.avi',-1, 10, (352,240))
video2 = cv2.VideoWriter('bg81.avi',-1, 10, (352,240))

fg=np.zeros((w,h,3),dtype=np.uint8,order='C')
bg=np.zeros((w,h,3),dtype=np.uint8,order='C')

mr=np.zeros((w,h,3),dtype=np.float64,order='C')
v=np.zeros((w,h,3),dtype=np.float64,order='C')
pr=np.zeros((w,h,3),dtype=np.float64,order='C')

mg=np.zeros((w,h,3),dtype=np.float64,order='C')
mb=np.zeros((w,h,3),dtype=np.float64,order='C')

pg=np.zeros((w,h,3),dtype=np.float64,order='C')
pb=np.zeros((w,h,3),dtype=np.float64,order='C')





count=0
while count<=997:
    ret,frame = capture.read()
    #pix = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for x in range(w):
         for y in range(h):
                if count==0:
                    mr[x][y][0] = 40;
                    mr[x][y][1] = 90;
                    mr[x][y][2] = 160;

                    mg[x][y][0] = 40;
                    mg[x][y][1] = 90;
                    mg[x][y][2] = 160;

                    mb[x][y][0] = 40;
                    mb[x][y][1] = 90;
                    mb[x][y][2] = 160;
           #--------------------------------------------            
                    v[x][y][0] = 20;
                    v[x][y][1] = 20;
                    v[x][y][2] = 20;
           #-------------------------------------------------            
                    pr[x][y][0] = 0.33;
                    pr[x][y][1] = 0.33;
                    pr[x][y][2] = 0.34;

                    pg[x][y][0] = 0.33;
                    pg[x][y][1] = 0.33;
                    pg[x][y][2] = 0.34;

                    pb[x][y][0] = 0.33;
                    pb[x][y][1] = 0.33;
                    pb[x][y][2] = 0.34;
             #----------------------------------------------           

                sb=[0,0,0]
                sg=[0,0,0]
                sr=[0,0,0]
                    
                ab=[0,0,0]
                ag=[0,0,0]
                ar=[0,0,0]

                match=-1
                none1=0;none2=0;none3=0
                rho = 0.1
                alpha = 0.025

                sumb=0
                sumg=0
                sumr=0
            #=-----------------------------------------------
                blue=frame[x,y,0]
                green=frame[x,y,1]
                red=frame[x,y,2]
        #---------------------------------------------
                #print(x,y)
                for i in range(0,3):
                    if(abs(blue-mb[x][y][i])<=2.5* np.sqrt(v[x][y][i])):
                        match=i
                        #rho = alpha * gauss(m[x][y][match], v[x][y][match], value)
                        mb[x][y][match] = (1 - rho) * mb[x][y][match] + rho * blue
                        v[x][y][match] = (1 - rho) * v[x][y][match] + rho * (blue - mb[x][y][match]) * (
                                             blue - mb[x][y][match])
                        pb[x][y][match] = (1 - alpha) * pb[x][y][match] + alpha * 1
                        sb[match] = 1
                        none1=none1+1
                        sumb+=pb[x][y][i]
                    else:
                        pb[x][y][i] = (1 - alpha) * pb[x][y][i]
                        sumb+=pb[x][y][i]
         #-----------------------------------------------------------------------
                                 
                    if(abs(green-mg[x][y][i])<=2.5* np.sqrt(v[x][y][i])):
                            match=i
                                 #rho = alpha * gauss(m[x][y][match], v[x][y][match], value)
                            mg[x][y][match] = (1 - rho) * mg[x][y][match] + rho * green
                            v[x][y][match] = (1 - rho) * v[x][y][match] + rho * (green - mg[x][y][match]) * (
                                             green - mg[x][y][match])
                            pg[x][y][match] = (1 - alpha) * pg[x][y][match] + alpha * 1
                            sg[match] = 1
                            none2=none2+1
                            sumg+=pg[x][y][i]
                    else:
                            pg[x][y][i] = (1 - alpha) * pg[x][y][i]
                            sumg+=pg[x][y][i]
          #---------------------------------------------------------------------

                    if(abs(red-mr[x][y][i])<=2.5* np.sqrt(v[x][y][i])):
                            match=i
                                 #rho = alpha * gauss(m[x][y][match], v[x][y][match], value)
                            mr[x][y][match] = (1 - rho) * mr[x][y][match] + rho * red
                            v[x][y][match] = (1 - rho) * v[x][y][match] + rho * (red - mr[x][y][match]) * (
                                             red - mr[x][y][match])
                            pr[x][y][match] = (1 - alpha) * pr[x][y][match] + alpha * 1
                            sr[match] = 1
                            none3=none3+1
                            sumr+=pr[x][y][i]
                    else:
                            pr[x][y][i] = (1 - alpha) * pr[x][y][i]
                            sumr+=pr[x][y][i]
        #-------------------------------------------------------------------------
                                 
                    ##normalise pi
      
                for i in range(0,3):
                        pb[x][y][i]=pb[x][y][i]/sumb
                        pg[x][y][i]=pg[x][y][i]/sumg
                        pr[x][y][i]=pr[x][y][i]/sumr
                    ##arrange in decreasing order
                ar[0] = pr[x][y][0] / np.sqrt(v[x][y][0])
                ar[1] = pr[x][y][1] / np.sqrt(v[x][y][1])
                ar[2] = pr[x][y][2] / np.sqrt(v[x][y][2])

                ab[0] = pb[x][y][0] / np.sqrt(v[x][y][0])
                ab[1] = pb[x][y][1] / np.sqrt(v[x][y][1])
                ab[2] = pb[x][y][2] / np.sqrt(v[x][y][2])
                    
                ag[0] = pg[x][y][0] / np.sqrt(v[x][y][0])
                ag[1] = pg[x][y][1] / np.sqrt(v[x][y][1])
                ag[2] = pg[x][y][2] / np.sqrt(v[x][y][2])
                    #a[3] = p[x][y][3] / np.sqrt(v[x][y][3])
                    
                for i in range(0,3):
                    max=i
                    for j in range(i+1,3):
                        if (ab[j] > ab[max]):
                            max = j
                        temp = ab[i]
                        ab[i] = ab[max]
                        ab[max] = temp
                        temp = mb[x][y][i]
                        mb[x][y][i] = mb[x][y][max]
                        mb[x][y][max] = temp
                        temp = v[x][y][i]
                        v[x][y][i] = v[x][y][max]
                        v[x][y][max] = temp
                        temp = pb[x][y][i]
                        pb[x][y][i] = pb[x][y][max]
                        pb[x][y][max] = temp
                        temp=sb[i]
                        sb[i]=sb[max]
                        sb[max]=temp
                 #-------------------------------------------
                for i in range(0,3):
                        max=i
                        for j in range(i+1,3):    
                            if (ag[j] > ag[max]):
                                max = j
                        temp = ag[i]
                        ag[i] = ag[max]
                        ag[max] = temp
                        temp = mg[x][y][i]
                        mg[x][y][i] = mg[x][y][max]
                        mg[x][y][max] = temp
                        temp = v[x][y][i]
                        v[x][y][i] = v[x][y][max]
                        v[x][y][max] = temp
                        temp = pg[x][y][i]
                        pg[x][y][i] = pg[x][y][max]
                        pg[x][y][max] = temp
                        temp=sg[i]
                        sg[i]=sg[max]
                        sb[max]=temp
                 #------------------------------------------
                for i in range(0,3):
                          max=i
                          for j in range(i+1,3):
                            if (ar[j] > ar[max]):
                                max = j
                            temp = ar[i]
                            ar[i] = ar[max]
                            ar[max] = temp
                            temp = mr[x][y][i]
                            mr[x][y][i] = mr[x][y][max]
                            mr[x][y][max] = temp
                            temp = v[x][y][i]
                            v[x][y][i] = v[x][y][max]
                            v[x][y][max] = temp
                            temp = pr[x][y][i]
                            pr[x][y][i] = pr[x][y][max]
                            pr[x][y][max] = temp
                            temp=sr[i]
                            sr[i]=sr[max]
                            sr[max]=temp
                         
                        
                         
                 #----------------------------------------
                    ##code to add new least probable distribution

                if none3==0:
                    mr[x][y][2] = red
                    v[x][y][2] = 100000

                if none2==0:
                    mg[x][y][2] = green
                    v[x][y][2] = 100000

                if none1==0:
                    mb[x][y][2] = blue
                    v[x][y][2] = 100000


                ##check if b or fg

                Bb=0;i=0;pi_sumb=0
                while(i<=2):
                    if(pi_sumb>0.8):
                        break
                    pi_sumb+=pb[x][y][i]
                    Bb=Bb+1
                    i=i+1
        #---------------------------------------

                Bg=0;j=0;pi_sumg=0
                while(j<=2):
                    if(pi_sumg>0.8):
                        break
                    pi_sumg+=pg[x][y][j]
                    Bg=Bg+1
                    j=j+1
        #-------------------------------------
                Br=0;k=0;pi_sumr=0
                while(k<=2):
                    if(pi_sumr>0.8):
                        break
                    pi_sumr+=pr[x][y][k]
                    Br=Br+1
                    k=k+1
        #--------------------------------------

            
                if(none1==0 or abs(blue-mb[x][y][0])>2.5*np.sqrt(v[x][y][0])):
                    fg[x,y,0]=blue
                    bg[x,y,0]=mb[x][y][0]
                elif Bb==1:
                    fg[x,y,0]=fg[x,y,0]
                    bg[x,y,0]=blue
                else:
                    fg[x,y,0]=255
                    bg[x,y,0]=mb[x][y][0]
        #-----------------------------------------------------
                if(none2==0 or abs(green-mg[x][y][0])>2.5*np.sqrt(v[x][y][0])):
                    fg[x,y,1]=green
                    bg[x,y,1]=mg[x][y][0]
                elif Bg==1:
                    fg[x,y,1]=fg[x,y,1]
                    bg[x,y,1]=green
                else:
                    fg[x,y,1]=255
                    bg[x,y,1]=mg[x][y][0]
        #------------------------------------------------------
                if(none3==0 or abs(red-mr[x][y][0])>2.5*np.sqrt(v[x][y][0])):
                    fg[x,y,2]=red
                    bg[x,y,2]=mr[x][y][0]
                elif Br==1:
                    fg[x,y,2]=fg[x,y,2]
                    bg[x,y,2]=red
                else:
                    fg[x,y,2]=255
                    bg[x,y,2]=mr[x][y][0]
        #--------------------------------------------------------

    
    #cv2.imshow('fg',fg)
    cv2.imwrite( r"C:\Users\Ravi\Desktop\bg8.2\frame%d.jpg" %count, bg );
    med = cv2.medianBlur(fg,3)
    cv2.imwrite( r"C:\Users\Ravi\Desktop\fg8.2\frame%d.jpg" %count, med );
    count += 1
    if cv2.waitKey(1000) == 27 & 0xFF:
        break


count=0
while count<=997:
    img1=cv2.imread( r"C:\Users\Ravi\Desktop\fg8.2\frame%d.jpg" %count, 0)
    video1.write(img1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    count+=1

count=0
while count<=997:
    img1=cv2.imread( r"C:\Users\Ravi\Desktop\bg8.2\frame%d.jpg" %count, 0)
    video2.write(img1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    count+=1

    
capture.release()
cv2.destroyAllWindows()
video1.release()
video2.release()


