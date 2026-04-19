import logging
import os
from typing import Dict, Any, List, Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Qaivid Audio Processor
    """

    DEFAULT_SR = 22050
    DEFAULT_BEATS_PER_BAR = 4
    DEFAULT_INTENSITY_POINTS = 64

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr if isinstance(sr, int) and sr > 0 else self.DEFAULT_SR

    def extract_features(
        self,
        file_path: str,
        beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
        target_intensity_points: int = DEFAULT_INTENSITY_POINTS,
    ) -> Dict[str, Any]:
        self._validate_file_path(file_path)
        beats_per_bar = self._validate_beats_per_bar(beats_per_bar)
        target_intensity_points = self._validate_target_points(target_intensity_points)

        logger.info("Analyzing audio file: %s", file_path)

        y, sr = librosa.load(file_path, sr=self.sr, mono=True)

        if y is None or len(y) == 0:
            raise ValueError("Audio file could not be loaded or is empty.")

        duration_seconds = float(librosa.get_duration(y=y, sr=sr))
        silence_ratio = self._estimate_silence_ratio(y)

        # Compute the onset envelope ONCE and reuse it for tempo + beat tracking
        # + onset detection. This is both faster and more accurate than letting
        # librosa re-derive it internally with default settings.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        bpm = self._estimate_tempo_robust(onset_env=onset_env, sr=sr)

        # Now seed beat tracking with the real estimate so it doesn't fall
        # back to the 120 BPM prior on ambiguous material.
        try:
            _bt_tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                start_bpm=bpm,
                tightness=100,
            )
        except Exception as exc:
            logger.warning("beat_track failed (%s); falling back to default", exc)
            _bt_tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        rms = librosa.feature.rms(y=y)[0]
        normalized_rms = self._normalize_array(rms)
        compressed_intensity_curve = self._compress_curve(
            normalized_rms,
            target_points=target_intensity_points,
        )

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        vibe_score = float(np.mean(spectral_centroids)) if len(spectral_centroids) else 0.0
        brightness_profile = self._classify_brightness(vibe_score)

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units="frames",
            backtrack=False,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        avg_energy = float(np.mean(normalized_rms)) if len(normalized_rms) else 0.0
        peak_energy = float(np.max(normalized_rms)) if len(normalized_rms) else 0.0
        dynamic_range = float(peak_energy - avg_energy)

        energy_profile = self._classify_energy(
            avg_energy=avg_energy,
            peak_energy=peak_energy,
            dynamic_range=dynamic_range,
        )

        vocal_gender, vocal_f0_hz, vocal_confidence = self._estimate_vocal_gender(y, sr)

        return {
            "bpm": bpm,
            "beats_per_bar": beats_per_bar,
            "duration_seconds": round(duration_seconds, 3),
            "intensity_curve": [round(x, 4) for x in compressed_intensity_curve],
            "intensity_curve_raw_points": len(normalized_rms),
            "intensity_curve_compressed_points": len(compressed_intensity_curve),
            "beat_times": [round(x, 3) for x in beat_times],
            "beat_count": len(beat_times),
            "onset_times": [round(x, 3) for x in onset_times],
            "onset_count": len(onset_times),
            "vibe_score": round(vibe_score, 3),
            "brightness_profile": brightness_profile,
            "avg_energy": round(avg_energy, 4),
            "peak_energy": round(peak_energy, 4),
            "dynamic_range": round(dynamic_range, 4),
            "energy_profile": energy_profile,
            "silence_ratio": round(silence_ratio, 4),
            "vocal_gender": vocal_gender,
            "vocal_f0_hz": vocal_f0_hz,
            "vocal_gender_confidence": vocal_confidence,
            "audio_hints": self._build_audio_hints(
                bpm=bpm,
                avg_energy=avg_energy,
                dynamic_range=dynamic_range,
                brightness_profile=brightness_profile,
                energy_profile=energy_profile,
            ),
        }

    def _estimate_vocal_gender(self, y: np.ndarray, sr: int) -> tuple:
        """Estimate the singer's gender from vocal fundamental frequency (F0).

        Uses librosa.pyin (probabilistic YIN) to track pitch over voiced frames,
        computes the median F0 of voiced regions, and classifies:
          * F0 <  165 Hz  -> "male"
          * 165 <= F0 < 265 Hz -> "female"
          * F0 >= 265 Hz  -> "female" (or child; treated as female for casting)
          * No reliable voiced frames -> "unknown" (likely instrumental)

        Returns (gender_label, median_f0_hz, confidence_0_to_1).
        Confidence reflects the fraction of frames that were voiced.
        """
        try:
            if y is None or len(y) == 0:
                return ("unknown", 0.0, 0.0)

            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=float(librosa.note_to_hz("C2")),   # ~65 Hz
                fmax=float(librosa.note_to_hz("C6")),   # ~1046 Hz
                sr=sr,
            )

            if f0 is None or len(f0) == 0:
                return ("unknown", 0.0, 0.0)

            voiced_f0 = f0[~np.isnan(f0)]
            voiced_ratio = float(len(voiced_f0)) / float(len(f0))

            # Need at least 5% voiced frames to call it a vocal track
            if len(voiced_f0) < 30 or voiced_ratio < 0.05:
                return ("unknown", 0.0, round(voiced_ratio, 3))

            median_f0 = float(np.median(voiced_f0))

            if median_f0 < 165.0:
                gender = "male"
            else:
                gender = "female"

            return (gender, round(median_f0, 1), round(voiced_ratio, 3))

        except Exception as exc:
            logger.warning("Vocal gender estimation failed: %s", exc)
            return ("unknown", 0.0, 0.0)

    def build_context_pre_analysis(
        self,
        audio_features: Dict[str, Any],
        base_pre_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = dict(base_pre_analysis or {})

        if not isinstance(audio_features, dict):
            return base

        bpm = self._safe_float(audio_features.get("bpm"), 120.0)
        brightness_profile = str(audio_features.get("brightness_profile", "")).strip()
        energy_profile = str(audio_features.get("energy_profile", "")).strip()

        audio_hints = {
            "audio_bpm": bpm,
            "audio_brightness_profile": brightness_profile,
            "audio_energy_profile": energy_profile,
            "audio_duration_seconds": self._safe_float(audio_features.get("duration_seconds"), 0.0),
            "vocal_gender": str(audio_features.get("vocal_gender") or "unknown"),
            "vocal_f0_hz": self._safe_float(audio_features.get("vocal_f0_hz"), 0.0),
            "vocal_gender_confidence": self._safe_float(audio_features.get("vocal_gender_confidence"), 0.0),
        }

        base["audio_hints"] = audio_hints
        return base

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        if arr is None or len(arr) == 0:
            return np.array([], dtype=float)

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if max_val - min_val <= 1e-9:
            return np.zeros_like(arr, dtype=float)

        normalized = (arr - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)

    def _compress_curve(self, curve: np.ndarray, target_points: int) -> List[float]:
        if curve is None or len(curve) == 0:
            return [0.0] * target_points

        if len(curve) <= target_points:
            return curve.astype(float).tolist()

        indices = np.linspace(0, len(curve) - 1, target_points).astype(int)
        sampled = curve[indices]
        return sampled.astype(float).tolist()

    def _estimate_silence_ratio(self, y: np.ndarray, threshold: float = 0.01) -> float:
        if y is None or len(y) == 0:
            return 1.0

        abs_y = np.abs(y)
        silent = np.sum(abs_y < threshold)
        return float(silent / len(abs_y))

    def _estimate_tempo_robust(self, onset_env: np.ndarray, sr: int) -> float:
        """Estimate BPM using multiple strategies and resolve octave errors.

        librosa's default beat_track silently falls back to its 120 BPM prior
        when the onset signal is ambiguous. This estimator:
          1. Runs librosa's tempo estimator with NO prior (start_bpm=None
             would error in some versions, so we use multiple priors and pick
             the strongest candidate from per-frame tempogram aggregation).
          2. Takes the median of per-frame tempos for stability.
          3. Resolves the classic half/double octave error by mapping the
             result into a musically reasonable range (60-180 BPM).
        """
        if onset_env is None or len(onset_env) == 0:
            return 120.0

        candidates: List[float] = []

        # Per-frame tempogram-based estimate, aggregated by median.
        try:
            tempo_fn = None
            for path in ("feature.rhythm", "feature", "beat"):
                try:
                    mod = librosa
                    for part in path.split("."):
                        mod = getattr(mod, part)
                    fn = getattr(mod, "tempo", None)
                    if callable(fn):
                        tempo_fn = fn
                        break
                except AttributeError:
                    continue
            if tempo_fn is not None:
                # Per-frame tempos -> median (robust to outliers).
                per_frame = tempo_fn(
                    onset_envelope=onset_env, sr=sr, aggregate=None
                )
                per_frame = np.asarray(per_frame).flatten()
                per_frame = per_frame[(per_frame >= 30) & (per_frame <= 300)]
                if per_frame.size > 0:
                    candidates.append(float(np.median(per_frame)))

                # Also try with a few different priors and collect aggregated
                # results — this catches songs where the tempogram is bimodal.
                for prior in (60.0, 90.0, 120.0, 150.0):
                    try:
                        t = tempo_fn(
                            onset_envelope=onset_env,
                            sr=sr,
                            start_bpm=prior,
                        )
                        t = float(np.asarray(t).flatten()[0])
                        if 30 <= t <= 300:
                            candidates.append(t)
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("tempo estimator failed: %s", exc)

        if not candidates:
            return 120.0

        # Resolve octave errors: fold every candidate into the 60-180 musical
        # window, then take the median of the folded set.
        folded = [self._fold_to_musical_range(c) for c in candidates]
        bpm = float(np.median(folded))

        # Final sanity clamp.
        if bpm < 40:
            bpm = 40.0
        elif bpm > 240:
            bpm = 240.0
        return round(bpm, 2)

    @staticmethod
    def _fold_to_musical_range(bpm: float, lo: float = 60.0, hi: float = 180.0) -> float:
        """Map a BPM into the [lo, hi) musical range by halving/doubling."""
        if bpm <= 0:
            return 120.0
        while bpm < lo:
            bpm *= 2.0
        while bpm >= hi:
            bpm /= 2.0
        return bpm

    def _repair_bpm(self, tempo: Any) -> float:
        bpm = self._safe_float(tempo, 120.0)

        if bpm <= 0:
            return 120.0

        if bpm < 40:
            return 40.0
        if bpm > 240:
            return 240.0

        return round(bpm, 3)

    def _classify_brightness(self, vibe_score: float) -> str:
        if vibe_score < 1500:
            return "warm_or_dark"
        if vibe_score < 3000:
            return "balanced"
        return "bright_or_airy"

    def _classify_energy(self, avg_energy: float, peak_energy: float, dynamic_range: float) -> str:
        if avg_energy < 0.2 and dynamic_range < 0.25:
            return "low_and_flat"
        if avg_energy < 0.35:
            return "restrained"
        if avg_energy < 0.6:
            return "moderate"
        if dynamic_range > 0.45:
            return "dynamic_high"
        return "consistently_high"

    def _build_audio_hints(
        self,
        bpm: float,
        avg_energy: float,
        dynamic_range: float,
        brightness_profile: str,
        energy_profile: str,
    ) -> Dict[str, Any]:
        pacing_hint = self._classify_pacing(bpm)
        motion_hint = self._classify_motion_hint(bpm, avg_energy)
        mood_hint = self._classify_mood_hint(brightness_profile, energy_profile)

        return {
            "pacing_hint": pacing_hint,
            "motion_hint": motion_hint,
            "mood_hint": mood_hint,
        }

    def _classify_pacing(self, bpm: float) -> str:
        if bpm < 75:
            return "slow"
        if bpm < 110:
            return "moderate"
        if bpm < 145:
            return "driving"
        return "fast"

    def _classify_motion_hint(self, bpm: float, avg_energy: float) -> str:
        if bpm < 80 and avg_energy < 0.3:
            return "linger_more"
        if bpm > 135 or avg_energy > 0.65:
            return "move_more"
        return "balanced_motion"

    def _classify_mood_hint(self, brightness_profile: str, energy_profile: str) -> str:
        if brightness_profile == "warm_or_dark" and energy_profile in {"restrained", "low_and_flat"}:
            return "intimate_or_melancholic"
        if brightness_profile == "bright_or_airy" and energy_profile in {"moderate", "dynamic_high"}:
            return "open_or_lifted"
        if energy_profile == "consistently_high":
            return "driven_or_intense"
        return "balanced"

    def _validate_file_path(self, file_path: str) -> None:
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

    def _validate_beats_per_bar(self, beats_per_bar: Any) -> int:
        try:
            value = int(beats_per_bar)
        except Exception:
            return self.DEFAULT_BEATS_PER_BAR

        if value <= 0:
            return self.DEFAULT_BEATS_PER_BAR
        return value

    def _validate_target_points(self, target_points: Any) -> int:
        try:
            value = int(target_points)
        except Exception:
            return self.DEFAULT_INTENSITY_POINTS

        if value < 8:
            return 8
        if value > 512:
            return 512
        return value

    def _safe_float(self, value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback
