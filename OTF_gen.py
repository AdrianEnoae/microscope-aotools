import numpy as np
from numpy.fft import *
from skimage import feature

def Generate_OTF(x,decide_=0,stack_size = 11,Cropped_image_size = 32,expand_size = 3,input_image_size = 8,defocus_top = [3,5,6],defocus_bottom = [1,2,4],overlap = 1,maxattempt=10):

    def Crop_feature(x,stack_size,Cropped_image_size,defocus_top,defocus_bottom,overlap):
#         x -= np.amin(x)+0.001
        x /= np.mean(x)
#         x /= (np.amax(x)-np.amin(x))
        
        X0_top = np.zeros((len(defocus_top),x.shape[0],x.shape[1]))
        X0_bottom = np.zeros((len(defocus_bottom),x.shape[0],x.shape[1]))
        print('x', x.shape)

        for i in range(len(defocus_top)):
            currentx = np.squeeze(x[:,:,defocus_top[i]].copy())
            currentx -=np.amin(currentx)
            X0_top[i,:,:] = currentx.copy()
            currentx = np.squeeze(x[:,:,defocus_bottom[i]].copy())
            currentx -=np.amin(currentx)
            X0_bottom[i,:,:] = currentx.copy()
#         plt.subplot(1,2,1)
#         plt.imshow(X0[0,:,:])
        convolvedx = feature.canny(X0_top[0,:,:], sigma=2)
#         plt.subplot(1,2,2)
#         plt.imshow(convolvedx)
        totalsteps = round(x.shape[0]/Cropped_image_size)
        
        for i in range(overlap):
            totalsteps = totalsteps*2-1
        
        overlap = 2**overlap
        
        todecide = np.array(range(totalsteps**2))
        p = np.zeros(totalsteps**2)
        x0crop_top = np.zeros((len(defocus_top),Cropped_image_size*expand_size,Cropped_image_size*expand_size,totalsteps**2))
        x0crop_bottom = np.zeros((len(defocus_bottom),Cropped_image_size*expand_size,Cropped_image_size*expand_size,totalsteps**2))

#         x1crop = np.zeros((Cropped_image_size*3,Cropped_image_size*3,totalsteps**2))
#         x2crop = np.zeros((Cropped_image_size*3,Cropped_image_size*3,totalsteps**2))

        paddingleft = round((x0crop_top.shape[1]-Cropped_image_size)/2)
        paddingright = x0crop_top.shape[1]-paddingleft-Cropped_image_size
        paddingtop = round((x0crop_top.shape[2]-Cropped_image_size)/2)
        paddingbottom = x0crop_top.shape[2]-paddingtop-Cropped_image_size
#         print('left:',paddingleft,'right:',paddingright,'top',paddingtop,'bottom',paddingbottom)
#         print('padding',paddingleft,paddingright)
        for i in range(totalsteps):
            for ii in range(totalsteps):
                for iii in range(len(defocus_top)):
                    x0crop_top[iii,:,:,i*totalsteps+ii] = np.pad(X0_top[iii,round(i/overlap*Cropped_image_size):round((i/overlap+1)*Cropped_image_size),round(ii/overlap*Cropped_image_size):round((ii/overlap+1)*Cropped_image_size)].copy(), ((paddingleft, paddingright), (paddingtop, paddingbottom)), 'constant', constant_values=((0, 0),(0,0)))
                    x0crop_bottom[iii,:,:,i*totalsteps+ii] = np.pad(X0_bottom[iii,round(i/overlap*Cropped_image_size):round((i/overlap+1)*Cropped_image_size),round(ii/overlap*Cropped_image_size):round((ii/overlap+1)*Cropped_image_size)].copy(), ((paddingleft, paddingright), (paddingtop, paddingbottom)), 'constant', constant_values=((0, 0),(0,0)))
                    
#                 x1crop[:,:,i*totalsteps+ii] = np.pad(X1.copy()[round(i/overlap*Cropped_image_size):round((i/overlap+1)*Cropped_image_size),round(ii/overlap*Cropped_image_size):round((ii/overlap+1)*Cropped_image_size)].copy(), ((paddingleft, paddingright), (paddingtop, paddingbottom)), 'constant', constant_values=(0, 0))
#                 x2crop[:,:,i*totalsteps+ii] = np.pad(X2.copy()[round(i/overlap*Cropped_image_size):round((i/overlap+1)*Cropped_image_size),round(ii/overlap*Cropped_image_size):round((ii/overlap+1)*Cropped_image_size)].copy(), ((paddingleft, paddingright), (paddingtop, paddingbottom)), 'constant', constant_values=(0, 0))
                p[i*totalsteps+ii] = np.sum(convolvedx.copy()[round(i/overlap*Cropped_image_size):round((i/overlap+1)*Cropped_image_size),round(ii/overlap*Cropped_image_size):round((ii/overlap+1)*Cropped_image_size)].copy())
#         for i in range(totalsteps):
#             for ii in range(totalsteps):
#                 plt.subplot(7,7,i*totalsteps+ii+1)
#                 plt.imshow(x0crop[iii,:,:,i*totalsteps+ii].copy())
        p /=(np.sum(p)+0.001)
        p += 0.1/totalsteps**2
        p /=np.sum(p)
#         print(p)
#         print(p)
#         decisionidx = np.random.choice(todecide, size=stack_size, replace=True, p=p)
        
        return x0crop_top,x0crop_bottom,todecide,p
    
    def outputindex(todecide,size,p):
        decisionidx = np.random.choice(todecide, size=size, replace=True, p=p)
        return decisionidx
    
    def generateOPT(Im0_top,Im0_bottom,ii,defocus_top,Cropped_image_size):
        Im = np.zeros((Cropped_image_size,Cropped_image_size,Im0_top.shape[-1]))
        for i in range(Im.shape[-1]):
            im0 = np.squeeze(Im0_bottom[(ii % (len(defocus_top))),:,:,i]).copy()
#             print(defocus,i % (len(defocus)-1)+1)
            im1 = np.squeeze(Im0_top[(ii % (len(defocus_top))),:,:,i]).copy()
            
            
            im = ifftshift(ifft2(np.divide(fft2(im1),fft2(im0))))
            
            leftpixel = round((im.shape[0]-Cropped_image_size)/2)
            rightpixel = leftpixel+Cropped_image_size
            
            Im[:,:,i] = im[leftpixel:rightpixel,leftpixel:rightpixel]
        return Im
    
    X = np.zeros((x.shape[0],x.shape[1],x.shape[2],stack_size))
    xsmall = np.zeros((x.shape[0],input_image_size,input_image_size,stack_size))
    Xaveraged = np.zeros((x.shape[0],x.shape[1],x.shape[2],len(defocus_top)))
    Xsmallaveraged = np.zeros((x.shape[0],input_image_size,input_image_size,len(defocus_top)))
    for i in range(x.shape[0]):
        
        cropped0_top,cropped0_bottom,todecide,p = Crop_feature(x[i],stack_size,Cropped_image_size,defocus_top,defocus_bottom,overlap)
#         print(cropped0.shape,todecide,p)
        Xsmall = np.zeros((Cropped_image_size,Cropped_image_size,stack_size))
        
        ii=0
        
        while ii <stack_size:
            iii =0
            if stack_size > len(defocus_top):
                while iii <= maxattempt:
                    dix = outputindex(todecide,9,p)
                    Xsmall1D = generateOPT(cropped0_top[:,:,:,dix].copy(),cropped0_bottom[:,:,:,dix].copy(),ii,defocus_top,Cropped_image_size)
                    XsmalliDmiddle = np.argsort(np.var(np.var(Xsmall1D,axis=0),axis=0))
        #             print(i,",",ii,"middle index:",XsmalliDmiddle)
                    if np.isnan(np.var(Xsmall1D[:,:,XsmalliDmiddle[0]])):
                        if np.isnan(np.var(Xsmall1D[:,:,XsmalliDmiddle[3]])):
                            if np.isnan(np.var(Xsmall1D[:,:,XsmalliDmiddle[5]])):
                                if iii== maxattempt:
                                    print('Problem!',i,'-',ii)
                                    Xsmall[:,:,ii] = np.zeros((Cropped_image_size,Cropped_image_size))
                                    ii +=1
                                    iii += 1
                                    decide_+=1

                                else:
                                    iii +=1
                            else:
                                if np.var(Xsmall1D[:,:,XsmalliDmiddle[5]])<1:
                                    Xsmall[:,:,ii] = Xsmall1D[:,:,XsmalliDmiddle[5]].copy()
                                    ii +=1
                                    iii = maxattempt+1

                        else:
                            if np.var(Xsmall1D[:,:,XsmalliDmiddle[3]])<1:
                                Xsmall[:,:,ii] = Xsmall1D[:,:,XsmalliDmiddle[3]].copy()
                                ii +=1
                                iii = maxattempt+1

                    else:
                        if np.var(Xsmall1D[:,:,XsmalliDmiddle[4]])<1:
                            Xsmall[:,:,ii] = Xsmall1D[:,:,XsmalliDmiddle[4]].copy()
                            ii +=1
                            iii = maxattempt+1
            else:
                dix = outputindex(todecide,1,p)
#                 print(dix)
                Xsmall1D = generateOPT(cropped0_top[:,:,:,dix].copy(),cropped0_bottom[:,:,:,dix].copy(),ii,defocus_top,Cropped_image_size)
                if np.isnan(np.var(Xsmall1D)):
                    print('Problem!',i,'-',ii)
                    Xsmall[:,:,ii] = np.zeros((Cropped_image_size,Cropped_image_size))
                    decide_+=1
                else:
                    Xsmall[:,:,ii] = Xsmall1D[:,:,0].copy()
                ii +=1
                        
        
        leftidx = round((x.shape[1]-Cropped_image_size)/2)
        rightidx = leftidx+Cropped_image_size
        topidx = round((x.shape[2]-Cropped_image_size)/2)
        bottomidx = topidx+Cropped_image_size
        
#         print(Xsmall.shape)
        
        X[i,leftidx:rightidx,topidx:bottomidx,:] = np.moveaxis(Xsmall.copy().reshape((Xsmall.shape[0],Xsmall.shape[1],Xsmall.shape[2],1)),-1,0)
        
        
        lefti = round(Xsmall.shape[0]/2)-round(input_image_size/2)
        righti = lefti+input_image_size
        topi = round(Xsmall.shape[1]/2)-round(input_image_size/2)
        bottomi = topi+input_image_size
        
        xsmall[i,:,:,:] = np.moveaxis(Xsmall[lefti:righti,topi:bottomi,:].copy().reshape((input_image_size,input_image_size,Xsmall.shape[2],1)),-1,0)
        
        count = np.zeros(len(defocus_top))
        for iv in range(X.shape[-1]):
            Xaveraged[i,:,:,iv%(len(defocus_top))] += X[i,:,:,iv]
            Xsmallaveraged[i,:,:,iv%(len(defocus_top))] += xsmall[i,:,:,iv]
            count[iv%(len(defocus_top))] +=1
            
        for iv in range(len(defocus_top)):
            Xaveraged[i,:,:,iv] /=count[iv]
            Xsmallaveraged[i,:,:,iv] /= count[iv]

    return Xaveraged.astype("float32"),Xsmallaveraged.astype("float32"),decide_