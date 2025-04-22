import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======== CONFIGURE PATHS =========
context_dir = r'C:\Users\lenovo\Downloads\audio_cont\audio_context'
utter_dir = r'C:\Users\lenovo\Downloads\audio_utter\audio_utterance'

# ======== UTILITY FUNCTION =========
def process_audio_folder(folder_path, label):
    lengths = []
    waveforms = []

    print(f"\Processing: {label}")

    for fname in os.listdir(folder_path):
        if not fname.endswith('.wav'):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            wave, sr = librosa.load(fpath, sr=16000)
            lengths.append(len(wave) / sr)
            waveforms.append(librosa.util.fix_length(wave, size=16000))  # pad/truncate to 1s
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

    return np.array(lengths), np.array(waveforms)

# ======== PROCESS CONTEXT =========
context_lengths, context_waveforms = process_audio_folder(context_dir, 'Context Audio')

# Plot: Context Duration
plt.figure(figsize=(10, 4))
sns.histplot(context_lengths, bins=30, kde=True, color='orange')
plt.title("Audio Duration Distribution - Context")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("context_audio_length_distribution.png")
plt.show()

# Plot: Mean Context Waveform
if len(context_waveforms) > 0:
    mean_context = np.mean(context_waveforms, axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(mean_context, color='darkorange')
    plt.title("Mean Waveform - Context Audio")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("context_mean_waveform.png")
    plt.show()
# Stats for Context Audio
mean_context_len = np.mean(context_lengths)
median_context_len = np.median(context_lengths)
print(f"\Context Audio — Mean Length: {mean_context_len:.2f}s, Median: {median_context_len:.2f}s")

# ======== PROCESS UTTERANCE =========
utter_lengths, utter_waveforms = process_audio_folder(utter_dir, 'Utterance Audio')

# Plot: Utterance Duration
plt.figure(figsize=(10, 4))
sns.histplot(utter_lengths, bins=30, kde=True, color='skyblue')
plt.title("Audio Duration Distribution - Utterance")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("utterance_audio_length_distribution.png")
plt.show()

# Plot: Mean Utterance Waveform
if len(utter_waveforms) > 0:
    mean_utter = np.mean(utter_waveforms, axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(mean_utter, color='slateblue')
    plt.title("Mean Waveform - Utterance Audio")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("utterance_mean_waveform.png")
    plt.show()
# Stats for Utterance Audio
mean_utter_len = np.mean(utter_lengths)
median_utter_len = np.median(utter_lengths)
print(f"\Utterance Audio — Mean Length: {mean_utter_len:.2f}s, Median: {median_utter_len:.2f}s")