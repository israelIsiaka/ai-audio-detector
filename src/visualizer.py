import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from feature_extractor import extract_features
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor":   "#12121a",
    "axes.edgecolor":   "#1e1e2e",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e1e2e",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})

NATURAL_COLOR = "#00ffc8"   # teal  = natural
AI_COLOR      = "#ff6b35"   # orange = AI


def plot_waveform_comparison(natural_path: str, ai_path: str, save_path: str = "results/waveform_comparison.png"):
    """Plot waveforms of both audio files side by side."""
    y_nat, sr_nat = librosa.load(natural_path, sr=22050, mono=True, duration=10)
    y_ai,  sr_ai  = librosa.load(ai_path,      sr=22050, mono=True, duration=10)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle("Waveform Comparison: Natural vs AI Audio", fontsize=14, fontweight="bold", y=1.01)

    axes[0].plot(np.linspace(0, len(y_nat)/sr_nat, len(y_nat)), y_nat, color=NATURAL_COLOR, linewidth=0.5, alpha=0.9)
    axes[0].set_title("Natural Audio", color=NATURAL_COLOR, fontsize=11)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(np.linspace(0, len(y_ai)/sr_ai, len(y_ai)), y_ai, color=AI_COLOR, linewidth=0.5, alpha=0.9)
    axes[1].set_title("AI Generated Audio", color=AI_COLOR, fontsize=11)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def plot_spectrogram_comparison(natural_path: str, ai_path: str, save_path: str = "results/spectrogram_comparison.png"):
    """Plot mel spectrograms side by side — visual fingerprint of each audio."""
    y_nat, sr_nat = librosa.load(natural_path, sr=22050, mono=True, duration=10)
    y_ai,  sr_ai  = librosa.load(ai_path,      sr=22050, mono=True, duration=10)

    S_nat = librosa.feature.melspectrogram(y=y_nat, sr=sr_nat, n_mels=128)
    S_ai  = librosa.feature.melspectrogram(y=y_ai,  sr=sr_ai,  n_mels=128)

    S_nat_db = librosa.power_to_db(S_nat, ref=np.max)
    S_ai_db  = librosa.power_to_db(S_ai,  ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Mel Spectrogram: Natural vs AI Audio", fontsize=14, fontweight="bold")

    img1 = librosa.display.specshow(S_nat_db, sr=sr_nat, x_axis="time", y_axis="mel", ax=axes[0], cmap="magma")
    axes[0].set_title("Natural Audio", color=NATURAL_COLOR, fontsize=11)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(S_ai_db, sr=sr_ai, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title("AI Generated Audio", color=AI_COLOR, fontsize=11)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def plot_feature_comparison(natural_path: str, ai_path: str, save_path: str = "results/feature_comparison.png"):
    """Bar chart comparing key acoustic features between natural and AI audio."""

    print("Extracting features for comparison chart...")
    nat = extract_features(natural_path)
    ai  = extract_features(ai_path)

    # Key features to compare — the ones most likely to differ
    features_to_plot = {
        "F0 Std Dev\n(pitch variation)":   ("f0_std",                nat, ai),
        "F0 Range\n(pitch range Hz)":       ("f0_range",              nat, ai),
        "Jitter\n(x100)":                   ("jitter_local",          nat, ai),
        "Shimmer":                           ("shimmer_local",         nat, ai),
        "HNR (dB)\n(/20)":                  ("hnr_mean",              nat, ai),
        "ZCR\n(x100)":                       ("zcr_mean",              nat, ai),
        "Spectral\nFlatness (x1000)":        ("spectral_flatness_mean",nat, ai),
        "MFCC Delta\n(rate of change)":      ("mfcc_delta_mean",       nat, ai),
    }

    # Scale some values so they're visible on the same chart
    scales = {
        "jitter_local": 100,
        "zcr_mean": 100,
        "spectral_flatness_mean": 1000,
        "hnr_mean": 1/20,
    }

    labels, nat_vals, ai_vals = [], [], []
    for label, (key, n, a) in features_to_plot.items():
        scale = scales.get(key, 1)
        labels.append(label)
        nat_vals.append((n.get(key) or 0) * scale)
        ai_vals.append((a.get(key) or 0) * scale)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("Acoustic Feature Comparison: Natural vs AI Audio\n(Your Research Data)", fontsize=14, fontweight="bold")

    bars1 = ax.bar(x - width/2, nat_vals, width, label="Natural",      color=NATURAL_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, ai_vals,  width, label="AI Generated", color=AI_COLOR,      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Feature Value (some scaled for visibility)")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=NATURAL_COLOR)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7, color=AI_COLOR)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def plot_pitch_contour(natural_path: str, ai_path: str, save_path: str = "results/pitch_contour.png"):
    """Plot F0 pitch contour over time — shows how pitch moves naturally vs AI."""
    y_nat, sr_nat = librosa.load(natural_path, sr=22050, mono=True, duration=10)
    y_ai,  sr_ai  = librosa.load(ai_path,      sr=22050, mono=True, duration=10)

    f0_nat, voiced_nat, _ = librosa.pyin(y_nat, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_ai,  voiced_ai,  _ = librosa.pyin(y_ai,  fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))

    times_nat = librosa.times_like(f0_nat, sr=sr_nat)
    times_ai  = librosa.times_like(f0_ai,  sr=sr_ai)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("Pitch (F0) Contour Over Time\nSmoother = more AI-like", fontsize=14, fontweight="bold")

    axes[0].plot(times_nat, f0_nat, color=NATURAL_COLOR, linewidth=1.2, label="F0")
    axes[0].set_title(f"Natural Audio  |  F0 std: {np.nanstd(f0_nat):.2f} Hz  |  Range: {np.nanmax(f0_nat) - np.nanmin(f0_nat):.2f} Hz", color=NATURAL_COLOR)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(0, 600)
    axes[0].grid(True)

    axes[1].plot(times_ai, f0_ai, color=AI_COLOR, linewidth=1.2, label="F0")
    axes[1].set_title(f"AI Audio  |  F0 std: {np.nanstd(f0_ai):.2f} Hz  |  Range: {np.nanmax(f0_ai) - np.nanmin(f0_ai):.2f} Hz", color=AI_COLOR)
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylim(0, 600)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✅ Saved: {save_path}")
    plt.close()


# ── Run All Plots ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/visualizer.py <natural_audio> <ai_audio>")
        print("Example: python src/visualizer.py data/natural/natural_sample.wav data/ai_generated/ai_sample.mp3")
        sys.exit(1)

    natural_path = sys.argv[1]
    ai_path      = sys.argv[2]

    print("\n Generating all comparison charts...\n")
    plot_waveform_comparison(natural_path, ai_path)
    plot_spectrogram_comparison(natural_path, ai_path)
    plot_feature_comparison(natural_path, ai_path)
    plot_pitch_contour(natural_path, ai_path)

    print("\n✅ All charts saved to results/")
    print("Open the results/ folder to view your research charts!")