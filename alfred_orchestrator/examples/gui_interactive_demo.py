from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from app.config import Settings
from app.orchestrator.schemas import SkillCall, UserRequest
from app.pipeline import AlfredRuntime
from test_elevenlabs_api import load_env_value, transcribe_audio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TK_PREVIEW_FRAME_PATH = PROJECT_ROOT / "runs" / "interactive" / "tk_preview_frame.ppm"


class AlfredGuiApp:
    def __init__(self, args):
        load_env_into_process()
        if args.camera_id is not None:
            os.environ["ALFRED_CAMERA_ID"] = str(args.camera_id)

        self.args = args
        self.api_key = load_env_value("ELEVENLABS_API_KEY")
        self.stt_model = load_env_value("ELEVENLABS_STT_MODEL") or Settings.load().elevenlabs_stt_model
        self.camera_id = int(os.getenv("ALFRED_CAMERA_ID", "1"))
        self.recording_process: Optional[subprocess.Popen[bytes]] = None
        self.recording_started_at: Optional[float] = None
        self.preview_capture: Optional[cv2.VideoCapture] = None
        self.preview_photo: Optional[tk.PhotoImage] = None
        self.captured_photo: Optional[tk.PhotoImage] = None
        self.preview_frame_count = 0
        self.message_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Alfred Orchestrator")
        self.root.geometry("1180x760")
        self.root.configure(bg="#202124")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="Ready. Press Start Recording.")
        self.speak_var = tk.BooleanVar(value=not args.no_speak)

        self._build_layout()
        self._log_header()
        self._start_preview()
        self.root.after(50, self._process_messages)
        self.root.after(100, self._update_preview)

    def run(self) -> int:
        if not self.api_key:
            self._log("ELEVENLABS_API_KEY was not found in environment, .env, or .env.example.")
            self.status_var.set("Missing ELEVENLABS_API_KEY.")
        self.root.mainloop()
        return 0

    def _build_layout(self) -> None:
        controls = tk.Frame(self.root, bg="#202124", padx=10, pady=10)
        controls.pack(fill=tk.X)

        self.start_button = tk.Button(
            controls,
            text="Start Recording",
            command=self.start_recording,
            bg="#1a73e8",
            fg="white",
            activebackground="#185abc",
            activeforeground="white",
            padx=12,
            pady=6,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_button = tk.Button(
            controls,
            text="Stop Recording",
            command=self.stop_recording,
            state=tk.DISABLED,
            bg="#d93025",
            fg="white",
            activebackground="#a50e0e",
            activeforeground="white",
            padx=12,
            pady=6,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))

        tk.Checkbutton(
            controls,
            text="Speak response",
            variable=self.speak_var,
            bg="#202124",
            fg="white",
            selectcolor="#303134",
            activebackground="#202124",
            activeforeground="white",
        ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            controls,
            textvariable=self.status_var,
            bg="#202124",
            fg="white",
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        panels = tk.Frame(self.root, bg="#202124", padx=10, pady=0)
        panels.pack(fill=tk.BOTH, expand=True)
        panels.grid_columnconfigure(0, weight=1, uniform="panels")
        panels.grid_columnconfigure(1, weight=1, uniform="panels")
        panels.grid_columnconfigure(2, weight=1, uniform="panels")
        panels.grid_rowconfigure(0, weight=1)

        left = self._make_panel(panels, "Live Camera Preview")
        middle = self._make_panel(panels, "Captured Image Sent To VLM")
        right = self._make_panel(panels, "Alfred Output / Debug")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
        middle.grid(row=0, column=1, sticky="nsew", padx=6, pady=(0, 10))
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=(0, 10))

        self.preview_label = tk.Label(
            left,
            text="Starting camera preview...",
            bg="#111111",
            fg="white",
            bd=1,
            relief=tk.SUNKEN,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.captured_label = tk.Label(
            middle,
            text="No captured image yet.",
            bg="#111111",
            fg="white",
            bd=1,
            relief=tk.SUNKEN,
        )
        self.captured_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.output_text = tk.Text(
            right,
            wrap=tk.WORD,
            height=30,
            bg="#111111",
            fg="#e8eaed",
            insertbackground="white",
            bd=1,
            relief=tk.SUNKEN,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.root.update_idletasks()

    def _make_panel(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg="#303134", bd=2, relief=tk.GROOVE)
        tk.Label(
            frame,
            text=title,
            bg="#303134",
            fg="white",
            font=("Helvetica", 14, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=8)
        return frame

    def _log_header(self) -> None:
        self._log("Alfred GUI interactive demo started.")
        self._log(f"Audio device: {self.args.device}")
        self._log(f"Camera id: {self.camera_id}")
        self._log(f"Audio output: {self.args.audio_output}")
        self._log("Click Start Recording, speak your request, then click Stop Recording.")

    def _log(self, message: str) -> None:
        print(message, flush=True)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def _log_step(self, label: str, title: str) -> None:
        self._log("")
        self._log(f"[{label}] {title}")
        self._log("-" * (len(str(label)) + len(title) + 3))

    def start_recording(self) -> None:
        if not self.api_key:
            self._log("Cannot record: ELEVENLABS_API_KEY is missing.")
            return
        try:
            self._log_step("1", "Recording started")
            self.recording_process = start_recording_process(self.args.device, self.args.audio_output)
            self.recording_started_at = time.monotonic()
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.status_var.set("Recording... click Stop Recording when done.")
            self._log("Speak now.")
        except Exception as exc:
            self._log(f"Could not start recording: {exc}")
            self.status_var.set("Recording failed to start.")

    def stop_recording(self) -> None:
        if not self.recording_process:
            return

        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping recording...")
        try:
            stop_recording_process(self.recording_process)
            duration = time.monotonic() - (self.recording_started_at or time.monotonic())
            self._log(f"Recording stopped after {duration:.1f} seconds.")
            self._log(f"Saved recording: {self.args.audio_output}")
        except Exception as exc:
            self._log(f"Recording stop failed: {exc}")
            self.start_button.configure(state=tk.NORMAL)
            self.status_var.set("Recording failed.")
            return
        finally:
            self.recording_process = None
            self.recording_started_at = None

        self._stop_preview()
        self.status_var.set("Processing request...")
        threading.Thread(target=self._run_pipeline_worker, daemon=True).start()

    def _run_pipeline_worker(self) -> None:
        try:
            self._pipeline_log_step("2", "Sending audio to ElevenLabs speech-to-text")
            transcript_payload = transcribe_audio(
                api_key=self.api_key or "",
                model_id=self.stt_model,
                audio_path=self.args.audio_output,
                timeout=self.args.timeout,
            )
            transcript = str(transcript_payload.get("text") or "").strip()
            self._pipeline_log(f"Transcript: {transcript}")
            if not transcript:
                self._pipeline_log(f"Raw STT response: {transcript_payload}")
                self._pipeline_status("No transcript returned.")
                return

            self._pipeline_log_step("3", "Creating Alfred runtime")
            settings = Settings.load()
            runtime = AlfredRuntime(settings)
            self._pipeline_log(f"Run artifacts directory: {runtime.run_dir}")

            self._pipeline_log_step("4", "Classifying intent")
            request = UserRequest(
                request_id=runtime.request_id,
                input_type="voice",
                raw_text=transcript,
            )
            runtime.logger.write_text("transcript.txt", request.raw_text)
            runtime.logger.log(
                stage="user_request",
                status="success",
                output_data=request.model_dump(mode="json"),
            )
            intent_result = runtime.orchestrator.classify_intent(request.raw_text)
            self._pipeline_log(f"Intent: {intent_result.intent.value}")
            self._pipeline_log(f"Confidence: {intent_result.confidence}")
            self._pipeline_log(f"Reason: {intent_result.reason}")

            self._pipeline_log_step("5", "Creating skill plan")
            plan = runtime.orchestrator.create_plan(request, intent_result)
            self._pipeline_log(f"Goal: {plan.goal}")
            for index, call in enumerate(plan.skill_calls, start=1):
                self._pipeline_log(f"  {index}. {call.skill_name} {call.arguments}")

            self._pipeline_log_step("6", "Executing skills")
            skill_results = []
            outputs_by_skill: Dict[str, Dict[str, Any]] = {}
            for call in plan.skill_calls:
                resolved_call = SkillCall(
                    skill_name=call.skill_name,
                    arguments=runtime.executor._resolve_arguments(call.arguments, outputs_by_skill),
                )
                self._pipeline_log(f"Executing skill: {resolved_call.skill_name}")
                result = runtime.executor.execute(resolved_call)
                skill_results.append(result)
                outputs_by_skill[result.skill_name] = result.output
                self._pipeline_log(f"Skill: {result.skill_name}")
                self._pipeline_log(f"  Status: {result.status}")
                if result.error:
                    self._pipeline_log(f"  Error: {result.error}")
                if result.output:
                    self._pipeline_log(f"  Output: {result.output}")
                if result.skill_name == "capture_wrist_camera_image" and result.status == "success":
                    image_path = result.output.get("image_path")
                    if image_path:
                        self.message_queue.put(("captured_image", image_path))
                        self._pipeline_log(f"Displayed captured image: {image_path}")
                if result.status == "error":
                    break

            self._pipeline_log_step("7", "Checking completion")
            completion = runtime.orchestrator.evaluate_completion(request, skill_results)
            self._pipeline_log(f"Task complete: {completion.task_complete}")
            self._pipeline_log(f"Reason: {completion.reason}")
            self._pipeline_log(f"Next action: {completion.next_action}")

            self._pipeline_log_step("8", "Generating final response")
            response = runtime.orchestrator.generate_final_answer(request, skill_results)
            self._pipeline_log(f"Alfred answer: {response.answer_text}")
            self._pipeline_log(f"Response confidence: {response.confidence}")
            self.message_queue.put(("answer", response.answer_text))

            if self.speak_var.get():
                self._pipeline_log_step("9", "Speaking answer with ElevenLabs TTS")
                speak_result = runtime.executor.execute(
                    SkillCall(skill_name="speak", arguments={"text": response.answer_text})
                )
                self._pipeline_log(f"TTS status: {speak_result.status}")
                if speak_result.error:
                    self._pipeline_log(f"TTS error: {speak_result.error}")
                if speak_result.output:
                    self._pipeline_log(f"TTS output: {speak_result.output}")
            else:
                self._pipeline_log_step("9", "Skipping text-to-speech")

            self._pipeline_log_step("Done", "GUI interactive loop complete")
            self._pipeline_log(f"Run artifacts directory: {runtime.run_dir}")
            self._pipeline_status("Done. Press Start Recording to run again.")
        except Exception as exc:
            self._pipeline_log(f"Pipeline failed: {exc}")
            self._pipeline_status("Pipeline failed.")
        finally:
            self.message_queue.put(("processing_done", None))

    def _pipeline_log(self, message: str) -> None:
        print(message, flush=True)
        self.message_queue.put(("log", message))

    def _pipeline_log_step(self, label: str, title: str) -> None:
        print("", flush=True)
        print(f"[{label}] {title}", flush=True)
        print("-" * (len(str(label)) + len(title) + 3), flush=True)
        self.message_queue.put(("log", ""))
        self.message_queue.put(("log", f"[{label}] {title}"))
        self.message_queue.put(("log", "-" * (len(str(label)) + len(title) + 3)))

    def _pipeline_status(self, message: str) -> None:
        self.message_queue.put(("status", message))

    def _process_messages(self) -> None:
        while True:
            try:
                kind, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._log(str(payload))
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "captured_image":
                self._show_captured_image(Path(str(payload)))
            elif kind == "answer":
                self._show_answer(str(payload))
            elif kind == "processing_done":
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
                self._start_preview()

        self.root.after(50, self._process_messages)

    def _show_answer(self, answer: str) -> None:
        self.output_text.insert(tk.END, "\nFinal answer:\n")
        self.output_text.insert(tk.END, answer + "\n")
        self.output_text.see(tk.END)

    def _start_preview(self) -> None:
        if self.preview_capture is not None:
            return
        self.preview_capture = open_camera_for_mac(self.camera_id)
        if not self.preview_capture.isOpened():
            self._log(f"Could not open camera {self.camera_id} for live preview.")
            self.preview_capture.release()
            self.preview_capture = None
            return
        self.preview_frame_count = 0
        self._log(f"Live camera preview started on camera {self.camera_id}.")

    def _stop_preview(self) -> None:
        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None
            self._log("Live camera preview paused for still capture.")

    def _update_preview(self) -> None:
        try:
            if self.preview_capture is not None:
                ret, frame = self.preview_capture.read()
                if ret:
                    self.preview_frame_count += 1
                    if self.preview_frame_count == 1:
                        height, width = frame.shape[:2]
                        self._log(f"Live preview first frame received: {width}x{height}.")
                    self.preview_photo = frame_to_photo(frame, max_width=520, max_height=360)
                    self.preview_label.configure(image=self.preview_photo, text="")
                else:
                    self.preview_label.configure(
                        image="",
                        text=f"Could not read live camera frame from camera {self.camera_id}.",
                    )
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview error: {exc}")
            print(f"Preview error: {exc}", flush=True)
        self.root.after(100, self._update_preview)

    def _show_captured_image(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.captured_label.configure(text=f"Could not load captured image: {image_path}")
            return
        self.captured_photo = frame_to_photo(frame, max_width=520, max_height=360)
        self.captured_label.configure(image=self.captured_photo, text="")

    def close(self) -> None:
        if self.recording_process:
            try:
                stop_recording_process(self.recording_process)
            except Exception:
                pass
        self._stop_preview()
        self.root.destroy()


def run_gui_demo(args) -> int:
    app = AlfredGuiApp(args)
    return app.run()


def load_env_into_process() -> None:
    for env_path in (PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def start_recording_process(device: str, output_path: Path) -> subprocess.Popen[bytes]:
    ffmpeg = require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        device,
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def stop_recording_process(process: subprocess.Popen[bytes]) -> None:
    if process.stdin:
        try:
            process.stdin.write(b"q")
            process.stdin.flush()
        except BrokenPipeError:
            pass
    try:
        _, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        _, stderr = process.communicate(timeout=5)
    if process.returncode not in (0, 255):
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg recording failed: {message}")


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    return ffmpeg


def open_camera_for_mac(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def frame_to_photo(frame, max_width: int, max_height: int) -> tk.PhotoImage:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    TK_PREVIEW_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(TK_PREVIEW_FRAME_PATH), frame):
        raise RuntimeError(f"Could not write Tk preview frame to {TK_PREVIEW_FRAME_PATH}")
    return tk.PhotoImage(file=str(TK_PREVIEW_FRAME_PATH))
