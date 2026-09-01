import argparse
import os
import re
import tempfile
import threading
import time
import wave
import shutil
import subprocess
from collections import deque

import cv2
import numpy as np
import torch

from checkpoint_utils import load_checkpoint
from clip_to_LLM_embedding import VisualProjector
from dataset_builder import EMOTION_LABELS
from model_heads import emotion_head, forward_pass, model, tokenizer
from sequence_builder import build_input_embeds, emo_token
from single_video_processing import process_frame_sequence


HUD_STATE = {
    "status": "starting",
    "user_text": "",
    "emotion": "",
    "response": "",
    "frame_info": "",
    "selected_frames": [],
    "clip_crops": [],
}


class CameraBuffer:
    def __init__(self, camera_index=0, max_seconds=20.0):
        self.camera_index = camera_index
        self.max_seconds = max_seconds
        self.frames = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.capture is not None:
            self.capture.release()

    def _capture_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()

            if not ok:
                time.sleep(0.05)
                continue

            now = time.time()

            with self.lock:
                self.frames.append((now, frame.copy()))

                cutoff = now - self.max_seconds
                while self.frames and self.frames[0][0] < cutoff:
                    self.frames.popleft()

    def get_frames(self, start_time, end_time, fallback_seconds=3.0):
        frames, _ = self.get_frame_window(
            start_time,
            end_time,
            fallback_seconds=fallback_seconds,
        )

        return frames

    def get_frame_window(self, start_time, end_time, fallback_seconds=3.0):
        with self.lock:
            selected = [
                (timestamp, frame.copy())
                for timestamp, frame in self.frames
                if start_time <= timestamp <= end_time
            ]
            used_fallback = False

            if not selected:
                fallback_start = end_time - fallback_seconds
                selected = [
                    (timestamp, frame.copy())
                    for timestamp, frame in self.frames
                    if fallback_start <= timestamp <= end_time
                ]
                used_fallback = True

        frames = [frame for _, frame in selected]
        now = time.time()

        if selected:
            oldest_timestamp = selected[0][0]
            newest_timestamp = selected[-1][0]
            info = (
                f"frames={len(selected)} "
                f"span={newest_timestamp - oldest_timestamp:.2f}s "
                f"newest_age={now - newest_timestamp:.2f}s"
            )
        else:
            info = "frames=0"

        if used_fallback:
            info += " fallback"

        return frames, info

    def get_latest_frame(self):
        with self.lock:
            if not self.frames:
                return None

            return self.frames[-1][1].copy()


class VisualHud:
    def __init__(self, camera, state, enabled=True, width=960):
        self.camera = camera
        self.state = state
        self.enabled = enabled
        self.width = width
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if not self.enabled:
            return

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def _snapshot(self):
        with self.lock:
            return dict(self.state)

    def _loop(self):
        while not self.stop_event.is_set():
            frame = self.camera.get_latest_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            hud = self._build_canvas(frame, self._snapshot())

            try:
                cv2.imshow("Realtime Emotion Robot", hud)
            except cv2.error as exc:
                print(f"HUD disabled: {exc}")
                self.stop_event.set()
                break

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                self.stop_event.set()

        try:
            cv2.destroyWindow("Realtime Emotion Robot")
        except cv2.error:
            pass

    def _build_canvas(self, frame, state):
        live = self._resize_width(frame, self.width)
        panel_height = 220
        panel = np.zeros((panel_height, live.shape[1], 3), dtype=np.uint8)

        self._draw_text(panel, f"Status: {state['status']}", 18, 32)
        self._draw_text(panel, f"User: {state['user_text']}", 18, 68)
        self._draw_text(panel, f"Emotion: {state['emotion']}", 18, 104)
        self._draw_text(panel, f"Robot: {state['response']}", 18, 140)
        self._draw_text(panel, "Press q in this window to close HUD", 18, 196)

        crops = self._build_crop_strip(state.get("clip_crops", []), live.shape[1])
        return np.vstack([live, crops, panel])

    def _build_crop_strip(self, crops, width):
        height = 170
        strip = np.zeros((height, width, 3), dtype=np.uint8)

        if not crops:
            self._draw_text(strip, "CLIP input crops will appear here", 18, 92)
            return strip

        tile_width = max(1, width // len(crops))

        for i, crop_rgb in enumerate(crops):
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            tile = self._letterbox(crop_bgr, tile_width, height - 24)
            x0 = i * tile_width
            strip[: tile.shape[0], x0 : x0 + tile.shape[1]] = tile
            self._draw_text(strip, f"CLIP frame {i + 1}", x0 + 8, height - 8)

        return strip

    @staticmethod
    def _resize_width(image, width):
        h, w = image.shape[:2]
        scale = width / max(w, 1)
        return cv2.resize(image, (width, max(1, int(h * scale))))

    @staticmethod
    def _letterbox(image, width, height):
        h, w = image.shape[:2]
        scale = min(width / max(w, 1), height / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y0 = (height - resized.shape[0]) // 2
        x0 = (width - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    @staticmethod
    def _draw_text(image, text, x, y):
        text = str(text).replace("\n", " ")

        if len(text) > 130:
            text = text[:127] + "..."

        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


class EnergyVadRecorder:
    def __init__(
        self,
        audio_backend="auto",
        sample_rate=16000,
        block_duration=0.1,
        threshold=None,
        threshold_multiplier=3.0,
        min_threshold=0.01,
        silence_duration=0.8,
        min_utterance_duration=0.4,
        max_utterance_duration=12.0,
        pre_roll_duration=0.4,
        device=None,
    ):
        self.audio_backend = audio_backend
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_size = int(sample_rate * block_duration)
        self.threshold = threshold
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.silence_duration = silence_duration
        self.min_utterance_duration = min_utterance_duration
        self.max_utterance_duration = max_utterance_duration
        self.pre_roll_blocks = max(1, int(pre_roll_duration / block_duration))
        self.device = device
        self.sd = None

    def _resolve_audio_backend(self):
        if self.audio_backend in ("auto", "sounddevice"):
            try:
                import sounddevice as sd

                self.sd = sd
                self.audio_backend = "sounddevice"
                return
            except OSError as exc:
                if self.audio_backend == "sounddevice":
                    raise RuntimeError(
                        "sounddevice is installed, but the PortAudio system "
                        "library is missing. On Ubuntu/Debian install it with: "
                        "sudo apt install libportaudio2 portaudio19-dev"
                    ) from exc
            except ImportError as exc:
                if self.audio_backend == "sounddevice":
                    raise RuntimeError(
                        "sounddevice is not installed. Install it with: "
                        "pip install sounddevice"
                    ) from exc

        if self.audio_backend in ("auto", "arecord"):
            if shutil.which("arecord") is not None:
                self.audio_backend = "arecord"
                return

            if self.audio_backend == "arecord":
                raise RuntimeError(
                    "arecord is not available. On Ubuntu/Debian install it with: "
                    "sudo apt install alsa-utils"
                )

        raise RuntimeError(
            "No microphone backend is available. Install PortAudio with "
            "'sudo apt install libportaudio2 portaudio19-dev' for sounddevice, "
            "or install ALSA tools with 'sudo apt install alsa-utils' for arecord."
        )

    def _audio_blocks(self):
        self._resolve_audio_backend()

        if self.audio_backend == "sounddevice":
            with self.sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                device=self.device,
            ) as stream:
                while True:
                    yield stream.read(self.block_size)

        if self.audio_backend == "arecord":
            command = [
                "arecord",
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                "1",
                "-r",
                str(self.sample_rate),
            ]

            if self.device is not None:
                command.extend(["-D", self.device])

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                bytes_per_block = self.block_size * 2

                while True:
                    raw_audio = process.stdout.read(bytes_per_block)

                    if len(raw_audio) < bytes_per_block:
                        stderr = process.stderr.read().decode(
                            "utf-8",
                            errors="replace",
                        )
                        raise RuntimeError(
                            "arecord stopped while reading microphone audio. "
                            f"{stderr.strip()}"
                        )

                    audio = np.frombuffer(raw_audio, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                    yield audio.reshape(-1, 1), False
            finally:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()

    def calibrate(self, seconds=1.5):
        if self.threshold is not None:
            return self.threshold

        rms_values = []
        blocks = max(1, int(seconds / self.block_duration))

        print("Calibrating microphone noise floor...")

        audio_blocks = self._audio_blocks()

        try:
            for _ in range(blocks):
                audio, _ = next(audio_blocks)
                rms_values.append(self._rms(audio))
        finally:
            audio_blocks.close()

        ambient_rms = float(np.mean(rms_values)) if rms_values else 0.0
        self.threshold = max(
            self.min_threshold,
            ambient_rms * self.threshold_multiplier,
        )

        print(
            f"Mic threshold: {self.threshold:.5f} "
            f"(ambient {ambient_rms:.5f})"
        )

        return self.threshold

    def listen_once(self):
        self.calibrate()

        pre_roll = deque(maxlen=self.pre_roll_blocks)
        chunks = []
        started = False
        utterance_start = None
        utterance_end = None
        silence_time = 0.0

        audio_blocks = self._audio_blocks()

        try:
            print("Listening...")

            while True:
                audio, overflowed = next(audio_blocks)

                if overflowed:
                    print("Audio input overflowed; continuing.")

                now = time.time()
                rms = self._rms(audio)

                if not started:
                    pre_roll.append(audio.copy())

                    if rms >= self.threshold:
                        started = True
                        utterance_start = now - len(pre_roll) * self.block_duration
                        chunks.extend(chunk.copy() for chunk in pre_roll)
                        silence_time = 0.0
                        print("Utterance started.")

                    continue

                chunks.append(audio.copy())

                if rms < self.threshold:
                    silence_time += self.block_duration
                else:
                    silence_time = 0.0

                duration = now - utterance_start
                ended_by_silence = (
                    duration >= self.min_utterance_duration
                    and silence_time >= self.silence_duration
                )
                ended_by_limit = duration >= self.max_utterance_duration

                if ended_by_silence or ended_by_limit:
                    utterance_end = now
                    print("Utterance ended.")
                    break
        finally:
            audio_blocks.close()

        audio = np.concatenate(chunks, axis=0).reshape(-1)
        return audio, utterance_start, utterance_end

    @staticmethod
    def _rms(audio):
        return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


class SpeechTranscriber:
    def __init__(self, backend="auto", model_size="base", language="en"):
        self.backend = backend
        self.model_size = model_size
        self.language = language
        self.impl = None
        self._load_backend()

    def _load_backend(self):
        backends = [self.backend]

        if self.backend == "auto":
            backends = ["faster_whisper", "whisper", "speech_recognition"]

        errors = []

        for backend in backends:
            try:
                if backend == "faster_whisper":
                    from faster_whisper import WhisperModel

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"
                    self.impl = (
                        backend,
                        WhisperModel(
                            self.model_size,
                            device=device,
                            compute_type=compute_type,
                        ),
                    )
                    print(f"Using ASR backend: {backend}")
                    return

                if backend == "whisper":
                    import whisper

                    self.impl = (
                        backend,
                        whisper.load_model(self.model_size),
                    )
                    print(f"Using ASR backend: {backend}")
                    return

                if backend == "speech_recognition":
                    import speech_recognition as sr

                    self.impl = (backend, sr.Recognizer())
                    print(f"Using ASR backend: {backend}")
                    return
            except Exception as exc:
                errors.append(f"{backend}: {exc}")

        raise RuntimeError(
            "No ASR backend is available. Install one of: "
            "faster-whisper, openai-whisper, or SpeechRecognition. "
            f"Tried: {'; '.join(errors)}"
        )

    def transcribe(self, audio, sample_rate):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name

        try:
            self._write_wav(wav_path, audio, sample_rate)
            backend, transcriber = self.impl

            if backend == "faster_whisper":
                segments, _ = transcriber.transcribe(
                    wav_path,
                    language=self.language,
                    vad_filter=False,
                )
                return " ".join(segment.text.strip() for segment in segments).strip()

            if backend == "whisper":
                result = transcriber.transcribe(
                    wav_path,
                    language=self.language,
                    fp16=torch.cuda.is_available(),
                )
                return result["text"].strip()

            if backend == "speech_recognition":
                import speech_recognition as sr

                with sr.AudioFile(wav_path) as source:
                    audio_data = transcriber.record(source)

                try:
                    return transcriber.recognize_google(
                        audio_data,
                        language=self.language,
                    ).strip()
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError as exc:
                    print(f"Speech recognition request failed: {exc}")
                    return ""

            raise RuntimeError(f"Unsupported ASR backend: {backend}")
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    @staticmethod
    def _write_wav(path, audio, sample_rate):
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_i16.tobytes())


class Speaker:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.engine = None

        if enabled:
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
            except Exception as exc:
                print(f"Text-to-speech unavailable: {exc}")
                self.enabled = False

    def say(self, text):
        if not self.enabled or self.engine is None:
            return

        self.engine.say(text)
        self.engine.runAndWait()


def build_context(history, user_text, include_robot_history=False):
    lines = []

    for user_turn, robot_turn in history:
        lines.append(f"User: {user_turn}")

        if include_robot_history:
            lines.append(f"Robot: {robot_turn}")

    lines.append(f"User: {user_text}")

    return "\n".join(lines)


def clean_response(response, one_sentence=True):
    response = response.strip()

    if not response:
        return response

    response = re.sub(r"\s+", " ", response)

    if one_sentence:
        match = re.search(r"(.+?[.!?])(?:\s|$)", response)

        if match:
            response = match.group(1)

    phrases = []

    for phrase in re.split(r"(?<=[.!?])\s+", response):
        normalized = phrase.lower().strip()

        if normalized and normalized not in phrases:
            phrases.append(normalized)
        elif normalized:
            break

    if phrases:
        kept = []
        seen = set()

        for phrase in re.split(r"(?<=[.!?])\s+", response):
            normalized = phrase.lower().strip()

            if normalized in seen:
                break

            seen.add(normalized)
            kept.append(phrase)

        response = " ".join(kept)

    return response.strip()


def generate_response(text, frame_bgr_list, projector, device, args):
    with torch.no_grad():
        visual_features, clip_crops = process_frame_sequence(
            frame_bgr_list,
            clip_frames=args.clip_frames,
            focus=not args.no_visual_focus,
            return_debug=True,
        )
        visual_features = visual_features.to(device).to(model.dtype)
        visual_tokens = projector(visual_features)

        emotion_logits, _ = forward_pass(text, visual_tokens, device)
        emotion_index = emotion_logits.argmax(dim=-1).item()
        predicted_emotion = EMOTION_LABELS[emotion_index]

        inputs_embeds, attention_mask, _ = build_input_embeds(
            text,
            visual_tokens,
            device,
        )
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        generated_ids = model.generate(**generation_kwargs)

    response = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()
    response = clean_response(
        response,
        one_sentence=not args.allow_multi_sentence,
    )

    return predicted_emotion, response, clip_crops


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous laptop camera + microphone inference loop."
    )
    parser.add_argument("--checkpoint", default="checkpoints/epoch_2")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--audio-backend",
        choices=["auto", "sounddevice", "arecord"],
        default="auto",
    )
    parser.add_argument("--mic-device", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--asr-backend", default="auto")
    parser.add_argument("--asr-model", default="base")
    parser.add_argument("--language", default="en")
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--clip-frames", type=int, default=3)
    parser.add_argument("--visual-window-seconds", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--silence-duration", type=float, default=0.8)
    parser.add_argument("--min-utterance-duration", type=float, default=0.4)
    parser.add_argument("--max-utterance-duration", type=float, default=12.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--hud-width", type=int, default=960)
    parser.add_argument("--include-robot-history", action="store_true")
    parser.add_argument("--allow-multi-sentence", action="store_true")
    parser.add_argument("--no-visual-focus", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    projector = VisualProjector().to(device).to(model.dtype)
    model.to(device)
    emotion_head.to(device).to(model.dtype)
    emo_token.to(device).to(model.dtype)
    load_checkpoint(args.checkpoint, projector, emotion_head, emo_token, device)

    model.eval()
    projector.eval()
    emotion_head.eval()
    emo_token.eval()

    camera = CameraBuffer(camera_index=args.camera_index)
    recorder = EnergyVadRecorder(
        audio_backend=args.audio_backend,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        threshold_multiplier=args.threshold_multiplier,
        min_threshold=args.min_threshold,
        silence_duration=args.silence_duration,
        min_utterance_duration=args.min_utterance_duration,
        max_utterance_duration=args.max_utterance_duration,
        device=args.mic_device,
    )
    transcriber = SpeechTranscriber(
        backend=args.asr_backend,
        model_size=args.asr_model,
        language=args.language,
    )
    speaker = Speaker(enabled=args.speak)
    history = deque(maxlen=args.context_window)
    hud = VisualHud(
        camera,
        HUD_STATE,
        enabled=not args.no_hud,
        width=args.hud_width,
    )

    camera.start()
    hud.start()
    print("Realtime inference is running. Press Ctrl+C to stop.")

    try:
        while not hud.stop_event.is_set():
            hud.update(status="listening")
            audio, utterance_start, utterance_end = recorder.listen_once()
            hud.update(status="transcribing")

            try:
                user_text = transcriber.transcribe(audio, args.sample_rate)
            except Exception as exc:
                print(f"Speech recognition failed: {exc}")
                hud.update(status="speech recognition failed")
                continue

            if not user_text:
                print("No speech recognized.")
                hud.update(status="no speech recognized")
                continue

            frame_start = utterance_start - args.visual_window_seconds
            frames = camera.get_frames(frame_start, utterance_end)

            if not frames:
                print("No camera frames available for this utterance.")
                hud.update(
                    status="no camera frames",
                    user_text=user_text,
                    emotion="",
                    response="",
                    clip_crops=[],
                )
                continue

            context_text = build_context(
                history,
                user_text,
                include_robot_history=args.include_robot_history,
            )

            print(f"User: {user_text}")
            hud.update(
                status="thinking",
                user_text=user_text,
                emotion="",
                response="",
            )
            emotion, response, clip_crops = generate_response(
                context_text,
                frames,
                projector,
                device,
                args,
            )
            print(f"Emotion: {emotion}")
            print(f"Robot: {response}")
            hud.update(
                status="responded",
                user_text=user_text,
                emotion=emotion,
                response=response,
                clip_crops=clip_crops,
            )

            speaker.say(response)
            history.append((user_text, response))

    except KeyboardInterrupt:
        print("\nStopping realtime inference.")
    finally:
        hud.stop()
        camera.stop()


if __name__ == "__main__":
    main()
