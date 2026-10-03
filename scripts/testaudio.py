from src.utils.audio import AudioLoader

audio = AudioLoader.load("sample.wav")

print(audio.waveform.shape)

print(audio.sample_rate)