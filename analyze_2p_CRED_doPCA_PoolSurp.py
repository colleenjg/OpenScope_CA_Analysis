import h5py
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA, RandomizedPCA

concatfile = 'concat_31Hz_0.h5'
pandas_pkl = '712483302_389778_20180621_df.pkl'

fsize = 500 #imaging frame size, in pixels. Images are 512x512, but I trim to 500 square
framerate = 31 #Hz...

#open and read the files
twop = h5py.File(concatfile,'r+')
file = open(pandas_pkl, 'rb')
stim_panda = pickle.load(file)
file.close()

################## pull out some dataframes and do PCA
ncomps = 20
if(0):
    totframes = len(twop['data']);
    nframes = 5000 #for the PCA
    flist = np.ceil(np.linspace(1000,totframes-1000,nframes))
    flist.tolist()
    f2 = [int(i) for i in flist]
    datframes = np.array(twop['data'][f2,0:fsize,0:fsize]) #this cuts off the edges of frames, which were wonky
    df2 = datframes.reshape([nframes,fsize*fsize])
    baselineF = np.mean(df2,axis=0)
    princomps = RandomizedPCA(ncomps).fit(df2)

#with open(r"PCA_eval.pkl","wb") as output_file:
#    pickle.dump(princomps,output_file)
#with open(r"df2.pkl","wb") as output_file:
    #pickle.dump(df2,output_file)
    #princomps.explained_variance_/sum(np.var(df2,axis=0))
#the components are in princomps.components_ as a n_components*fsize^2 array
#later, get PCA components as nfc = np.dot(princomps.components_,np.transpose(newframe-baselineF))
#PC1 = np.reshape(princomps.components_[0,:],[fsize,fsize])
#plt.imshow(PC1)

#with open(r"baselinef.pkl","wb") as output_file:
    #pickle.dump(baselineF,output_file)

if(1):
    ff = open('PCA_eval.pkl','rb')
    princomps = pickle.load(ff)
    ff.close()
    fff = open('baselinef.pkl','rb')
    baselineF = pickle.load(fff)
    fff.close()

is_brick = 1
is_gabor = 0
##########################################
#pull out the some Surprises. Get df/f relative to the 1 second prior...
mystimframesb = np.where((stim_panda['surp'] == 1) & (stim_panda['stimType'] == 0) ) #& (stim_panda['stimPar2'] == 'left') & (stim_panda['stimPar1'] == 128.0))

#find the first one in each surprise block
nframesb = np.size(mystimframesb)
framearrayb = np.array(mystimframesb) 
boolsb = np.zeros(nframesb)
boolsb[0] = 1
twop_startframesb = np.zeros(nframesb)
twop_startframesb[0] = stim_panda['start_frame'][framearrayb.item(0)]
for i in range(1,nframesb):
    twop_startframesb[i] = stim_panda['start_frame'][framearrayb.item(i)]
    if (framearrayb.item(i) - framearrayb.item(i-1)) > (1*is_brick + 4*is_gabor):
        boolsb[i] = 1

is_brick = 0
is_gabor = 1
mystimframesg = np.where((stim_panda['surp'] == 1) & (stim_panda['stimType'] == 1) & (stim_panda['GABORFRAME'] == 3))
#find the first one in each surprise block
nframesg = np.size(mystimframesg)
framearrayg = np.array(mystimframesg) 
boolsg = np.zeros(nframesg)
boolsg[0] = 1
twop_startframesg = np.zeros(nframesg)
twop_startframesg[0] = stim_panda['start_frame'][framearrayg.item(0)]
for i in range(1,nframesg):
    twop_startframesg[i] = stim_panda['start_frame'][framearrayg.item(i)]
    if (framearrayg.item(i) - framearrayg.item(i-1)) > (1*is_brick + 4*is_gabor):
        boolsg[i] = 1

bools = np.concatenate((boolsb,boolsg),axis=0)
framearray = np.concatenate((framearrayb,framearrayg),axis=1)
twop_startframes = np.concatenate((twop_startframesb,twop_startframesg),axis=0)

#go back and forward from framestart by secs_f and secs_b. Get df/f relative to the baseline computed below
sframes = twop_startframes[bools==1]
print(sframes)
secs_f = 4
secs_b = 4
df_trace = np.zeros([len(sframes),(secs_f + secs_b)*framerate,ncomps])
baselinearray = np.tile(baselineF,[(secs_f + secs_b)*framerate,1])
framegetter = np.zeros([len(sframes),(secs_f + secs_b)*framerate,fsize,fsize])
for i in range(0,len(sframes)):
    print(i)
    thedata = np.reshape(twop['data'][int(sframes.item(i)) - secs_b*framerate:int(sframes.item(i)) + secs_f*framerate,0:fsize,0:fsize],[(secs_f + secs_b)*framerate,fsize*fsize])
    df_trace[i,:,:] = np.transpose(np.dot(princomps.components_,np.transpose(np.divide((thedata-baselinearray),baselinearray))))
    framegetter[i,:,:,:] = twop['data'][int(sframes.item(i)) - secs_b*framerate:int(sframes.item(i)) + secs_f*framerate,0:fsize,0:fsize]


meanpre = np.mean(framegetter[:,0:secs_b*framerate,:,:],axis=(0,1))
meanpost = np.mean(framegetter[:,secs_b*framerate:,:,:],axis=(0,1))

dfpre = np.mean(df_trace[:,0:framerate-1,:],axis=1)
dfpost = np.mean(df_trace[:,5*framerate+1:,:],axis=1)

component_to_show = 7
print(stats.ttest_ind(dfpre[:,component_to_show],dfpost[:,component_to_show]))

#avgtrace =np.nanmean(df_trace,axis=0)
#avpre = np.mean(avgtrace[0:secs_b*framerate-1,:],axis=0)
#avpost = np.mean(avgtrace[secs_b*framerate:(secs_b+secs_f)*framerate,:],axis=0)
#selectivity = np.divide((avpost-avpre),abs(avpost + avpre))

#print(selectivity)
#print(np.mean(selectivity))
#print(stats.ttest_ind(avpre.flatten(),avpost.flatten()))

al = np.linspace(-secs_b*framerate,secs_f*framerate,(secs_f + secs_b)*framerate)/framerate
meancurve = np.mean(df_trace[:,:,component_to_show],axis=0)
stdcurve = np.sqrt(np.var(df_trace[:,:,component_to_show],axis=0))/np.sqrt(len(sframes))
plt.plot(al,meancurve)
plt.fill_between(al,meancurve-stdcurve,meancurve+stdcurve)
plt.show()


al = np.linspace(-secs_b*framerate,secs_f*framerate,(secs_f + secs_b)*framerate)/framerate
plt.plot(al,np.transpose(df_trace[:,:,component_to_show]))
plt.show()
#################################
