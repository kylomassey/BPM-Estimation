import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy

def smooth(onset, framel):
    smooth = []
    for f in range(len(onset)):
        l = max(0,f-int((framel-1)/2))
        r = min(len(onset)-1, f+int((framel/2)))
        smooth.append(numpy.average(onset[l:r+1]))
    return smooth
    
def autoCorr(onset, laglow, laghigh):
    eps=1e-12
    onset = numpy.asarray(onset, dtype=float)
    onset = onset -  numpy.mean(onset)
    corr = []
    for k in range(laglow, laghigh + 1, 1):
        corr.append(
            (numpy.sum(numpy.multiply(onset[:-k], onset[k:]))) /
            ((numpy.sqrt(numpy.sum(onset[:-k] ** 2)))*(numpy.sqrt(numpy.sum(onset[k:] ** 2))) + eps)
        )
    return corr
    
def tempogram(onset, hop, laglow, laghigh):
    tempogram = []
    score = []
    win_size = int(4/hop)
    win_hop = int(win_size/4)
    print(win_size, " ", win_hop, " ", len(onset))
    for k in range(0, len(onset)-win_size+1, win_hop):
        win =  onset[k : k + win_size]
        corr = autoCorr(win, laglow, laghigh)
        if len(score) == 0:
            score = numpy.zeros(len(corr))    
        score[harmScor(corr,laglow,laghigh)] += 1
        tempogram.append(corr)
    print(score)
    highscore = numpy.argmax(score)
    return tempogram, highscore

def harmScor(corr, laglow, laghigh):
    scores = []
    for k in range(laglow, laghigh + 1, 1):
        total = corr[k - laglow]
        if k // 2 > laglow:
            total += corr[int(k / 2 - laglow)] * .5
        if k * 2 <= laghigh:
            total += corr[k * 2 - laglow] * .5
        if k * 3 <= laghigh:
            total += corr[k * 3 - laglow] * .33
        if k * 4 <= laghigh:
            total += corr[k * 4 - laglow] * .25
        scores.append(total)
    #print (scores[16], " ", scores[28])
    return numpy.argmax(scores)

path = "C:/Users/PC/Desktop/bpm-estimator/music/queen.mp3"
y, sr = librosa.load(path, sr=None)
print(sr)
framel =  sr * .05
frame = librosa.util.frame(x = y, frame_length = int(framel), hop_length = int(framel * .5) )
window = numpy.hanning(frame.shape[0])
hop = ((framel*.5)/sr)

spectrum = []
for f in range(len(frame[0])):
    windowed = frame[:,f] * window
    fft = numpy.fft.rfft(windowed)
    spectrum.append(numpy.abs(fft) **2)
spectrum = numpy.array(spectrum).T

master, low, lowmid, highmid, high = [],[],[],[],[]
#print(spectrum[int((20*framel)/sr):int((200*framel)/sr),f])
for f in range(len(spectrum[0])):
    master.append(numpy.average(spectrum[:,f]))
    low.append(numpy.average(spectrum[int((20*framel)/sr):int((200*framel)/sr),f]))
    lowmid.append(numpy.average(spectrum[int((200*framel)/sr):int((800*framel)/sr),f]))
    highmid.append(numpy.average(spectrum[int((800*framel)/sr):int((3000*framel)/sr),f]))
    high.append(numpy.average(spectrum[int((3000*framel)/sr):int((8000*framel)/sr),f]))
    
time = numpy.arange(len(spectrum[0])) * hop

master_onset = numpy.maximum(numpy.diff(master) / hop, 0)
low_onset = numpy.maximum(numpy.diff(low) / hop, 0)
lowmid_onset = numpy.maximum(numpy.diff(lowmid) / hop, 0)
highmid_onset = numpy.maximum(numpy.diff(highmid) / hop, 0)
high_onset = numpy.maximum(numpy.diff(high) / hop, 0)

master_smooth = smooth(master_onset, int(.1/(hop)))
low_smooth = smooth(low_onset, int(.1/ hop))
lowmid_smooth = smooth(lowmid_onset, int(.1/(hop)))
highmid_smooth = smooth(highmid_onset, int(.1/hop))
high_smooth = smooth(high_onset, int(.1/hop))

laglow = int(60 / hop / 180)
laghigh = int(60 / hop / 60)
print(laglow, " ", laghigh)
bpm_graph =  numpy.divide(60, numpy.multiply(numpy.arange(laglow,laghigh+1),hop))
master_corr = autoCorr(master_smooth, laglow, laghigh)
bpm = bpm_graph[numpy.argmax(master_corr)]
bpm2 = bpm_graph[harmScor(master_corr, laglow, laghigh)]
print("The BPM of this audio is: ", bpm, " or ", bpm2)
#print (bpm_graph[16], " ", bpm_graph[28])

tempo, highscore = tempogram(master_smooth, hop, laglow, laghigh)
bpm = bpm_graph[highscore]
print("or it could be", bpm)
tempo =  numpy.array(tempo).T

time_diff = time[1:]
plt.plot(time_diff, master_onset)
plt.show()
plt.plot(time_diff, master_smooth)
plt.show()
plt.plot(bpm_graph,master_corr)
plt.show()
plt.imshow(
    tempo,
    aspect = 'auto',
    origin = 'upper',
)
plt.show()