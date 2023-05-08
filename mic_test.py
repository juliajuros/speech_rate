import pyaudio
import numpy as np
from speech_rate import SpeechRateEstimator
import webrtcvad

# vad init
vad = webrtcvad.Vad()
vad.set_mode(2)
frame_duration = 20  # 10 ms, *2 (bytes)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # sr for vad
CHUNK = 1000
CHUNK_vad = int(RATE * frame_duration / 1000)  # number of bytes in 10 ms
WINDOW_LEN = 1  # window length in seconds

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

win_len = 100
overlap_len = 50
opts = ['vad']
sr_est = SpeechRateEstimator()
previous_frame_float = []
previous_is_speech = []
sr_estimator = SpeechRateEstimator()

try:
    while True:
        frame = []
        is_speech = []
        frame_bytes = []
        # read data from mic (1 sec)
        for i in range(0, int(RATE / CHUNK * WINDOW_LEN)):
            data = stream.read(CHUNK)
            frame_bytes.append(data)
            numpydata = np.frombuffer(data, dtype=np.int16)
            frame.append(numpydata)
            # check if there is voice activity
            for j in range(0, CHUNK, CHUNK_vad):
                is_speech.append(vad.is_speech(data[j:j + CHUNK_vad], RATE))
        true_num = is_speech.count(True)
        false_num = is_speech.count(False)
        if false_num < true_num:
            frame = np.hstack(frame)
            frame_float = np.array(frame).astype(np.float32)
            sr_est.speech_rate_with_buffer(np.concatenate((previous_frame_float, frame_float)), vad_is_speech=np.concatenate((previous_is_speech, is_speech)), fs=RATE)
            previous_frame_float = frame_float  # sr is counted from 2 sec data
            previous_is_speech = is_speech  # sr is counted from 2 sec data
        else:
            print("unvoiced segment\n")

except KeyboardInterrupt:
    pass
