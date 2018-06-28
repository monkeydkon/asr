import numpy as np
import scipy.io.wavfile
import scipy as scipy

from scipy.fftpack import dct

class SignalProcessing:



    def __init__(self):
        return None



    def readwav(path):
        sample_rate, signal = scipy.io.wavfile.read(path)  # File assumed to be in the same directory    return sample_rate, signal
        return sample_rate, signal

    def pre_emphasis(signal, prefactor = 0.95):

        return np.append(signal[0], signal[1:] - prefactor * signal[:-1])



    def frame_data(emphasized_signal, sample_rate, frame_size = 0.025, frame_swift = 0.010):

        frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_swift * sample_rate))
        signal_length = len(emphasized_signal)
        number_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = number_frames * frame_step + frame_length # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        z = np.zeros((pad_signal_length - signal_length))
        padsignal = np.append(emphasized_signal,
                                  z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        indices = np.tile(np.arange(0, frame_length), (number_frames, 1)) + np.tile(np.arange(0, number_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = padsignal[indices.astype(np.int32, copy=False)]
        return frames,frame_length

    def window_frame(frames, frame_length):

        frames *= np.hamming(frame_length)
        return frames

    def fft(frames, nfft = 512):
        magnitude_frames = np.absolute(np.fft.rfft(frames,nfft)) # Magnitude of the FFT
        return magnitude_frames

    def power(frames, nfft=512):
        power_frames = ((1.0 / nfft) * ((frames) ** 2)) # Power Spectrum
        return power_frames

    def filter_banks(sample_rate,power_frames, nfft = 512, nfilt= 40):

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((nfft + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(power_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8) # Mean Normalization

        return filter_banks

    def mfcc(filterbanks, num_ceps = 12, cep_lifter = 22):
        mfcc = dct(filterbanks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8) # Mean Normalization

        return mfcc

    def GetFinal(self):
        sample_rate, signal = SignalProcessing.readwav("/home/monkeydkon/Desktop/tests2/chunk8.wav")
        signal = SignalProcessing.pre_emphasis(signal)
        frames, frame_length = SignalProcessing.frame_data(signal, sample_rate)
        frames = SignalProcessing.window_frame(frames, frame_length)
        frames = SignalProcessing.fft(frames)
        frames = SignalProcessing.power(frames)
        filterbanks = SignalProcessing.filter_banks(sample_rate, frames)
        mf1cc = SignalProcessing.mfcc(filterbanks)
        final = np.array(mf1cc)
        return final





    # sample_rate, signal = readwav("/home/monkeydkon/Desktop/tests2/chunk8.wav")
    # signal = pre_emphasis(signal)
    # frames, frame_length = frame_data(signal,sample_rate)
    # frames = window_frame(frames,frame_length)
    # frames = fft(frames)
    # frames = power(frames)
    # filterbanks = filter_banks(sample_rate,frames)
    # mf1cc = mfcc(filterbanks)
    # final = np.array(mf1cc)
    # print(final)