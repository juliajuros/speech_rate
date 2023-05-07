import os
import pickle
import numpy as np
import scipy
from attr import dataclass
from matplotlib import pyplot as plt
from numpy import sqrt, mean, floor, dot, zeros, array, ndarray, argmin, insert
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, sosfilt
from librosa import pyin, load, resample
from numpy_ringbuffer import RingBuffer


@dataclass()
class SpeechRateEstimator:
    subbands: ndarray = [
        240,
        360,
        480,
        600,
        720,
        840,
        1000,
        1150,
        1300,
        1450,
        1600,
        1800,
        2000,
        2200,
        2400,
        2700,
        3000,
        3300,
        3750,
    ]  # sub-bands ranges [Hz] #optional - first subband
    M: int = 12  # number of selected sub-bands
    sigma: float = 1.2  # weighting Gaussian window variance
    smooth_len: int = 15  # smoothing window length
    smooth_var: float = 1.3  # smoothing gaussian window variance
    thresh_time: int = 13  # neighboring peak distance threshold
    thresh_comp: float = 1.7876397958306827  # left-compare-only threshold in logarithmic scale
    fs: int = 9000  # sample rate - init
    resample_fs: int = 900  # sample rate after filtering and resampling, 450*2=900,
    K: int = 11  # temporal correlation window len
    pyin_frame_len: int = 360  # pyin frame length
    buffer_sr = RingBuffer(3)  # buffer for speech rate
    mean_s = 3.514118788387982  # average speech rate for slow speech
    mean_n = 4.013732909607244  # average speech rate for normal speech
    mean_f = 4.508238429262722  # average speech rate for fast speech
    std_s = 0.5703347069007503  # std of speech rate for slow speech
    std_n = 0.725115344532664  # std of speech rate for normal speech
    std_f = 0.820951705950185  # std of speech rate for fast speech

    def __attrs_post_init__(self):
        self.N = (
                self.M * (self.M - 1) / 2
        )  # number of pairs between sub-bands for subbands correlation
        self.A = int(self.fs / self.resample_fs)  # resampling parameter
        self.energy_win_len = int(self.resample_fs / 50)  # 18 samples, 20 ms (fs = 900)
        self.energy_hop_len = int(self.resample_fs / 100)  # 9 samples, ~10 ms
        self.energy_fs = (
                self.resample_fs / self.energy_hop_len
        )  # sample rate after extracting energy in windows - 100 Hz
        self.pyin_win_len = self.pyin_frame_len / 4  # pyin window length

    def filter(self, x: ndarray, sb: int = 240) -> list:
        """
        filtration with second-order butterworth band-pass filter
        :param x: audio time series
        :param sb: optional - lower range  of the first sub-band
        :return: filtered: 18 (optional 19) subbands after filtration
        """
        filt_resamp = []
        if sb != 240:
            self.subbands = insert(self.subbands, 0, sb)
        for n in range(len(self.subbands) - 1):
            sos = butter(
                2,
                [self.subbands[n] * 2 / self.fs, self.subbands[n + 1] * 2 / self.fs],
                "bandpass",
                output="sos",
            )
            filtered = sosfilt(sos, x)
            resampled = filtered[:: self.A]
            filt_resamp.append(resampled)
        return filt_resamp

    def extract_energy(self, x: ndarray, filtered: list) -> ndarray:
        """
        energy envelope for all sub-bands in frames with length fs/50 and overlap 1/2
        :param x: audio time series
        :param filtered: 18 subbands after filtration
        :return: energy envelope for all subbands
        """

        energy_envelopes = zeros(
            (len(self.subbands) - 1, int(len(filtered[0]) / self.energy_hop_len) + 1)
        )

        for k in range(len(self.subbands) - 1):
            j = 0  # samples in energy env
            for i in range(0, len(filtered[k]), self.energy_hop_len):
                if len(x) - i >= self.energy_hop_len:
                    energy_envelopes[k][j] = (
                            dot(
                                filtered[k][i: i + self.energy_win_len],
                                filtered[k][i: i + self.energy_win_len],
                            )
                            / self.energy_win_len
                    )
                    # average energy in windows
                    j += 1
                else:
                    energy_envelopes[k][j] = (
                            dot(filtered[k][i: len(x)], filtered[k][i: len(x)])
                            / self.energy_win_len
                    )
                    # average energy in window for the last sample
        return energy_envelopes

    def get_top_subbands(self, energy_envelopes: ndarray) -> ndarray:
        """
        get top M energy envelopes
        :param energy_envelopes: energy envelope for all sub-bands in frames with length fs/50 and overlap 1/2
        :return: energy envelope for top M sub-bands in frames with length fs/50 and overlap 1/2
        """
        avg = zeros(len(self.subbands) - 1)
        for k in range(len(self.subbands) - 1):
            avg[k] = mean(energy_envelopes[k][:])

        #  sort and get top M envelopes
        index_sorted = sorted(range(len(avg)), key=avg.__getitem__)
        index_sorted.reverse()
        index_sorted = index_sorted[:12]

        energy_envelopes = energy_envelopes[index_sorted][:]
        return energy_envelopes

    def apply_temporal_correlation(self, energy_envelopes: ndarray) -> ndarray:
        """
        temporal correlation of energy envelope for each subband
        :param energy_envelopes: energy envelope for top M subbands
        :return: energy envelope after temporal correlation for top M subbands
        """

        w = gaussian(self.K, self.sigma)  # window coefficients
        y_temp_correlated = zeros([len(energy_envelopes[0]), self.M])
        for l in range(len(energy_envelopes[0]) - 2):
            if len(energy_envelopes[0]) - l < self.K:  # for a few final samples
                self.K = self.K - 1
                w = gaussian(self.K, self.sigma)
            part_energy = zeros([self.M, self.K])
            for m in range(self.M):
                part_energy[m] = energy_envelopes[m][l: l + self.K] * w
                for j in range(self.K - 2):
                    for p in range(j + 1, self.K - 1):
                        y_temp_correlated[l, m] += part_energy[m][j] * part_energy[m][p]
                y_temp_correlated[l, m] = sqrt(
                    1 / (2 * self.K * (self.K - 1)) * y_temp_correlated[l, m]
                )
        y_temp_correlated = (
                8 * y_temp_correlated.T
        )  # checked how this function works for constant signal and the
        # same (approximately) values before and after I get when I multiplied by 8
        return y_temp_correlated

    def apply_subband_correlation(self, y_temp_correlated: ndarray) -> ndarray:
        """
        subbands cross correlation to obtain 1D envelope
        :param y_temp_correlated: energy envelopes after temporal correlation for top M subbands
        :return: y_cross_corr: 1D envelope obtained by subband energy vectors correlation
        """
        y_cross_corr = []
        for l in range(len(y_temp_correlated[0])):
            y_cross_corr.append(0)
            for i in range(self.M - 1):
                for j in range(i + 1, self.M):
                    y_cross_corr[l] += (
                            1 / self.N * y_temp_correlated[i][l] * y_temp_correlated[j][l]
                    )
        y_cross_corr = array(y_cross_corr)
        return y_cross_corr

    def smooth(self, y_cross_corr: ndarray) -> ndarray:
        """
        envelope smoothing by gaussian filter
        :param y_cross_corr: 1D envelope obtained by subband energy vectors correlation
        :return: y_smoothed: smoothed 1D envelope
        """
        y_smoothed = zeros([len(y_cross_corr)])
        for l in range(
                int((self.smooth_len - 1) / 2),
                int(len(y_cross_corr) - (self.smooth_len - 1) / 2 - 1),
        ):
            idx_bottom = int(
                l - (self.smooth_len - 1) / 2
            )  # gaussian filter is centered in the middle of segment
            idx_top = int(l + (self.smooth_len - 1) / 2)
            y_smoothed[idx_bottom:idx_top] += gaussian_filter(
                y_cross_corr[idx_bottom:idx_top], np.sqrt(self.smooth_var)
            )
        y_smoothed = y_smoothed / self.smooth_len
        return y_smoothed

    def apply_thresholding(self, y_smoothed: ndarray) -> list:
        """
        temporal threshold (13 samples as in article) and amplitude threshold (set after optimization based on our base)
        :param y_smoothed: smoothed 1D envelope
        :return: peaks after thresholding
        """
        y_smoothed = array(y_smoothed)
        y_smoothed = np.log(y_smoothed)
        all_zeros = not np.any(y_smoothed)
        if all_zeros:
            peaks_final = []
            return peaks_final
        peaks, _ = find_peaks(y_smoothed[:])  # peaks in envelope
        peaks_final = []
        if len(peaks) == 0:
            return peaks_final
        mins, _ = find_peaks(y_smoothed[:] * -1)  # minimums in envelope
        first_min_idx = argmin(y_smoothed[: peaks[0]])
        mins = insert(mins, 0, first_min_idx)  # add first minimum

        for l in range(len(peaks)):
            if (
                    y_smoothed[peaks[l]] - y_smoothed[mins[l]] > self.thresh_comp
            ):  # amplitude threshold
                if not peaks_final:
                    peaks_final.append(peaks[l])
                elif (
                        peaks[l] - peaks_final[-1] > self.thresh_time
                ):  # temporal threshold
                    peaks_final.append(peaks[l])
        return peaks_final

    def get_peaks_without_thresholding(self, y_smoothed: ndarray) -> list:
        """
        peaks without thresholding - for test only
        :param y_smoothed: smoothed 1D envelope
        :return: peaks without thresholding
        """
        y_smoothed = array(y_smoothed)
        peaks, _ = find_peaks(y_smoothed[:])  # peaks in envelope
        peaks = list(peaks)
        return peaks

    def verify_pitch(self, peaks: list, x: ndarray) -> tuple[list, list]:
        """
        pitch verification, using pyin, to delete unvoiced peaks
        :param peaks: peaks after thresholding
        :param x: audio time series
        :return: peaks_with_voicing: peaks after pitch verification
        """
        f0, voiced_flag, voiced_probs = pyin(
            x, fmin=65, fmax=2093, sr=self.fs, frame_length=self.pyin_frame_len
        )
        # in documentation frame length was ~10 x less than sr
        pyin_ref = []
        for i in voiced_flag:
            if i:
                pyin_ref.extend([1] * int(self.pyin_win_len * self.energy_fs / self.fs))
            else:
                pyin_ref.extend([0] * int(self.pyin_win_len * self.energy_fs / self.fs))
        peaks_with_voicing = []
        for p in range(len(peaks)):
            index = int(floor((peaks[p]) / (self.pyin_win_len * self.energy_fs / self.fs)))
            if len(voiced_flag) < index + 1:
                if (
                        voiced_flag[index] or voiced_flag[index - 1] or voiced_flag[index + 1]
                ):  # voiced_flag[int(floor(peaks[p] / self.pyin_win_len))]:
                    peaks_with_voicing.append(peaks[p])
            else:
                if (
                        voiced_flag[index] or voiced_flag[
                    index - 1]):  # voiced_flag[int(floor(peaks[p] / self.pyin_win_len))]:
                    peaks_with_voicing.append(peaks[p])
        return peaks_with_voicing, pyin_ref

    def verify_pitch_vad(self, peaks: list, vad_is_speech: ndarray, y_cross_corr: ndarray) -> list:
        """
        voice activity detection, using vad, to delete unvoiced peaks
        :param peaks: peaks after thresholding
        :param vad_is_speech: vector containing True or False values if piece of audio is voiced or unvoiced
        :param y_cross_corr: 1D envelope obtained by subband energy vectors correlation
        :return: peaks_with_voicing: peaks after pitch verification
        """
        peaks_with_voicing = []
        for peak in peaks:
            i = int(peak * len(vad_is_speech) / len(y_cross_corr))
            if vad_is_speech[i]:
                peaks_with_voicing.append(peak)
        return peaks_with_voicing

    def return_peaks(
            self, peaks: list, y_smoothed: ndarray, win_len: int, overlap_len: int
    ) -> list:
        """
        return a vector with the number of peaks per frame - with specified frame size and overlap
        :param peaks: peaks after pitch verification
        :param y_smoothed: smoothed 1D envelope
        :param win_len: window length in samples
        :param overlap_len: overlap length in samples
        :return: speech_rate_vector: vector with the number of peaks per frame
        """
        speech_rate_vector = []
        len_y = len(y_smoothed)
        w = 0
        while w < len_y:
            peaks_n = 0
            for p in range(len(peaks)):
                if (peaks[p] > w) and (peaks[p] < w + win_len):
                    peaks_n += 1
            speech_rate_vector.append(peaks_n)
            w += win_len - overlap_len
        return speech_rate_vector

    def estimate_speech_rate(
            self,
            wav: str or ndarray,
            win_len: int,
            overlap_len: int,
            opts: list,
            sb: int = 240,
            fs: int = None,
            normalize: bool = True,
            vad_is_speech: ndarray = []
    ) -> tuple[list, array, list, list]:
        """
        Estimate speech rate from a wav file, with specified window and overlap length
        :param wav: filepath or signal array as ndarray
        :param fs: sampling rate of signal given in wav or None (default) when wav is a path
        :param normalize: if True (default) input signal is normalized
        :param win_len: window length for speech rate vector in samples
        :param overlap_len: overlap length for speech rate vector in samples
        :param opts: option for removing component from function
        :param sb: first subband (optional)
        :param vad_is_speech: vector containing True or False values if piece of audio is voiced or unvoiced
        :return: peaks_final: peaks in envelope (syllable nuclei), y_filt: smoothed envelope required to find peaks(syllable nuclei), sr: speech rate vector
        """
        if os.path.exists(wav):
            x, fs = load(wav, sr=9000)
        elif type(wav) == ndarray and type(fs) == int:
            x = resample(wav, orig_sr=fs, target_sr=self.fs)
        else:
            raise ValueError
        if normalize:
            x = x / sqrt(mean(x ** 2))  # * 2e15
        filtered = self.filter(x, sb)
        energy_envelopes = self.extract_energy(x, filtered)
        if "all_subbands" not in opts:
            energy_envelopes = self.get_top_subbands(energy_envelopes)
        if "no_temp_corr" not in opts:
            energy_envelopes = self.apply_temporal_correlation(energy_envelopes)
        y_cross_corr = self.apply_subband_correlation(energy_envelopes)
        if "no_smoothing" not in opts:
            y_cross_corr = self.smooth(y_cross_corr)
        if "no_thresholding" not in opts:
            peaks = self.apply_thresholding(y_cross_corr)
        else:
            peaks = self.get_peaks_without_thresholding(y_cross_corr)
        if "vad" in opts:
            peaks = self.verify_pitch_vad(peaks, vad_is_speech, y_cross_corr)
            ref_pyin = []
        elif "no_pyin" not in opts:
            peaks, ref_pyin = self.verify_pitch(peaks, x)
        else:
            ref_pyin = []
        sr = self.return_peaks(peaks, y_cross_corr, win_len, overlap_len)

        return peaks, y_cross_corr, sr, ref_pyin

    def speech_rate_with_buffer(self,
                                path: str or ndarray,
                                vad_is_speech: ndarray,
                                fs: int = 9000,
                                opts: list = ["vad"],
                                frame_len: int = 2,
                                win_len: int = 100,
                                overlap_len: int = 50
                                ):
        """

        :param path: filepath or signal array as ndarray
        :param vad_is_speech: vector containing True or False values if piece of audio is voiced or unvoiced
        :param fs: sampling rate of signal given in wav or None when wav is a path
        :param opts: option for removing component from function
        :param frame_len: frame length in seconds (default 2 seconds) for tests in real time
        :param win_len: window length for speech rate vector in samples
        :param overlap_len: overlap length for speech rate vector in samples
        :return: buffer of 3 previous speech rates
        """
        if os.path.exists(path):
            x, fs = load(path, sr=9000)
        elif type(path) == ndarray and type(fs) == int:
            x = resample(path, orig_sr=fs, target_sr=self.fs)
            fs = self.fs
        else:
            raise ValueError
        for i in range(0, len(x), frame_len * fs):
            peaks, y_cross_filtered, sr_vec, ref_pyin = self.estimate_speech_rate(
                x[i:i + fs * frame_len], win_len, overlap_len, opts, normalize=True, fs=fs, vad_is_speech=vad_is_speech
            )
            if peaks:
                self.buffer_sr.append(len(peaks) / (len(x[i:i + fs * 2]) / fs))
            else:
                self.buffer_sr.append(0)
            sr_mean = np.sum(self.buffer_sr) / len(self.buffer_sr)
            print(self.buffer_sr, sr_mean)
            pdf_s = scipy.stats.norm(self.mean_s, self.std_s).pdf(sr_mean)
            pdf_n = scipy.stats.norm(self.mean_n, self.std_n).pdf(sr_mean)
            pdf_f = scipy.stats.norm(self.mean_f, self.std_f).pdf(sr_mean)
            pdf = [pdf_s, pdf_n, pdf_f]  # probability density function
            p_sum = (pdf_s * 1 / 3 + pdf_n * 1 / 3 + pdf_f * 1 / 3)
            lh_s = pdf_s * 1 / 3 / p_sum
            lh_n = pdf_n * 1 / 3 / p_sum
            lh_f = pdf_f * 1 / 3 / p_sum

            lh = [lh_s, lh_n, lh_f]  # probability
            max_lh = lh.index(max(lh))

            log_lh = np.log(lh)
            log_lh = list(log_lh)  # log probability
            max_log_lh = log_lh.index(max(log_lh))

            print("likelihood: " + str(lh) + ", max: " + str(max_lh))
            # print("log likelihood: " + str(log_lf) + ", max: " + str(max_log_lh))
            max_pdf = pdf.index(max(pdf))
            if max_lh == 0:
                print("slow")
            if max_lh == 1:
                print("normal")
            if max_lh == 2:
                print("fast")
        return self.buffer_sr
