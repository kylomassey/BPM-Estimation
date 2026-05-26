#Spectrogram.py
import numpy

def spectrogram(audio):
    spectrum = []
    window = numpy.hanning(audio.shape[0])
    
    for f in range(len(audio[0])):
        windowed_audio_frame = audio[:,f] * window
        fft_audio_frame =  numpy.fft.rfft(windowed_audio_frame)
        spectrum.append(numpy.abs(fft_audio_frame) ** 2)
    spectrum = numpy.array(spectrum).T
    return(spectrum)