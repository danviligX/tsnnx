import os
import sys

sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from libx.dataio import Dset

path = './data/set/01_tracks.pth'
dset = Dset(path=path)

be_f = dset.frame(1)
car_num = len(be_f)
car_track = []
for idx in range(car_num):
    car_id = int(be_f[idx,0].item())
    track = dset.search_track(target_id=car_id,begin_frame=1,end_frame=124)
    car_track.append((car_id,track))

plt.figure(figsize=(20,3))
plt.title('Tracks')
for track in car_track:
    car_idx = track[0]
    track_info = track[1]

    x = track_info[:,0]
    y = track_info[:,1]
    plt.plot(x,y)
    plt.arrow(x[-1],y[-1],0.01*(x[-1]-x[-2]),0.01*(y[-1]-y[-2]),head_width=0.2,color='limegreen')
plt.show()

def show_log():
    f = open('./logs/FLinearNet_1024.log','r')

    el = []
    for line in f.readlines(0):
        if line[0]=='t':
            idx = line.find('error')
            el.append(float(line[idx+6:-1]))

    import matplotlib.pylab as plt
    plt.plot(range(len(el)),el)
    plt.show()
            