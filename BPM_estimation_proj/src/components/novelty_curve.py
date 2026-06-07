import numpy

def onset_curve(spectrum, hop_time):
    spectrum =  numpy.sum(spectrum, axis=0)
    return numpy.maximum(numpy.diff(spectrum)/hop_time, 0)

def smooth_curve(onset, framel):
    kernel = numpy.ones(framel, dtype=float)
    smooth = numpy.convolve(onset, kernel, mode="same")
    return smooth

#def smooth_curve(onset, framel):
    #smooth = []
    #for f in range(len(onset)):
        #l = max(0,f-int((framel-1)/2))
        #r = min(len(onset)-1, f+int((framel/2)))
        #smooth.append(numpy.average(onset[l:r+1]))
    #return smooth

def process_band(spec, hop):
    return smooth_curve(onset= onset_curve(spectrum=spec, hop_time=hop), framel=max(1,int(.1/hop)))