import numpy

from src.components.frequency_ranges import freq_range


def test_freq_range_full_range_is_untouched():
    spectrum = numpy.arange(100 * 5).reshape(100, 5).astype(float)
    fr = freq_range(spectrum, frame_len=1024, sample_rate=22050, hop_len=256)
    assert numpy.array_equal(fr.full_range, spectrum)


def test_freq_range_bands_match_expected_bin_slices():
    sample_rate = 22050
    frame_len = 1024
    spectrum = numpy.arange(1000 * 4).reshape(1000, 4).astype(float)
    fr = freq_range(spectrum, frame_len=frame_len, sample_rate=sample_rate, hop_len=256)

    def bin_of(hz):
        return int(hz * frame_len / sample_rate)

    assert numpy.array_equal(
        fr.sub_bass_range, spectrum[bin_of(20):bin_of(60)]
    )
    assert numpy.array_equal(
        fr.bass_range, spectrum[bin_of(60):bin_of(250)]
    )
    assert numpy.array_equal(
        fr.brilliance, spectrum[bin_of(6000):bin_of(20000)]
    )


def test_freq_range_bands_are_contiguous_and_ordered():
    sample_rate = 22050
    frame_len = 1024
    spectrum = numpy.zeros((1000, 1))
    fr = freq_range(spectrum, frame_len=frame_len, sample_rate=sample_rate, hop_len=256)

    def bin_of(hz):
        return int(hz * frame_len / sample_rate)

    # Each band should start exactly where the previous one ends.
    assert bin_of(60) >= bin_of(20)
    assert bin_of(250) >= bin_of(60)
    assert fr.sub_bass_range.shape[0] == bin_of(60) - bin_of(20)
    assert fr.bass_range.shape[0] == bin_of(250) - bin_of(60)


def test_freq_range_derived_scalars():
    sample_rate = 22050
    frame_len = 1024
    hop_len = 256
    spectrum = numpy.zeros((10, 2))
    fr = freq_range(spectrum, frame_len=frame_len, sample_rate=sample_rate, hop_len=hop_len)

    assert fr.hop_len == hop_len
    assert fr.hop_time == hop_len / sample_rate
    assert fr.bin_size == sample_rate / frame_len
