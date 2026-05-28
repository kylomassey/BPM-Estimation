import numpy

def auto_correlation(onset, laglow, laghigh):
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

def harmonic_scoring(corr, laglow, laghigh):
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

def tempogram(onset, hop, laglow, laghigh):
    tempogram = []
    score = []
    win_size = int(4/hop)
    win_hop = int(win_size/4)
    print(win_size, " ", win_hop, " ", len(onset))
    for k in range(0, len(onset)-win_size+1, win_hop):
        win =  onset[k : k + win_size]
        corr = auto_correlation(win, laglow, laghigh)
        if len(score) == 0:
            score = numpy.zeros(len(corr))    
        score[harmonic_scoring(corr,laglow,laghigh)] += 1
        tempogram.append(corr)
    print(score)
    highscore = numpy.argmax(score)
    return tempogram, highscore