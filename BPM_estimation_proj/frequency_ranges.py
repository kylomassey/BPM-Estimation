class freq_range:
    def __init__(self, spectrum, frame_len, sample_rate):
        self.full_range = spectrum
        self.sub_bass_range = spectrum[int(20*frame_len/sample_rate):int(60*frame_len/sample_rate)]
        self.bass_range = spectrum[int(60*frame_len/sample_rate):int(250*frame_len/sample_rate)]
        self.low_mid_range = spectrum[int(250*frame_len/sample_rate):int(500*frame_len/sample_rate)]
        self.mid_range = spectrum[int(500*frame_len/sample_rate):int(2000*frame_len/sample_rate)]
        self.upper_mid_range = spectrum[int(2000*frame_len/sample_rate):int(4000*frame_len/sample_rate)]
        self.presence = spectrum[int(4000*frame_len/sample_rate):int(6000*frame_len/sample_rate)]
        self.brilliance = spectrum[int(6000*frame_len/sample_rate):int(20000*frame_len/sample_rate)]