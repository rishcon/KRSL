"""Tkinter application for testing isolated-sign recognition on video files."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from krsl_ai.inference.video import VideoPrediction, load_lstm_checkpoint, recognize_video
from krsl_ai.ui.presentation import present_prediction

VIDEO_TYPES = (
    ("Видео", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
    ("Все файлы", "*.*"),
)


def runtime_paths(project_root: Path) -> tuple[Path, Path]:
    """Return the required MediaPipe and recognizer model paths."""
    return (
        project_root / "models" / "holistic_landmarker.task",
        project_root / "models" / "lstm-handcentric-v3.pt",
    )


def check_runtime(project_root: Path) -> str:
    """Validate local assets without opening a window."""
    holistic_model, checkpoint = runtime_paths(project_root)
    missing = [str(path) for path in (holistic_model, checkpoint) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены необходимые файлы:\n" + "\n".join(missing))
    loaded = load_lstm_checkpoint(checkpoint)
    return (
        f"model={loaded.model_type} features={loaded.model.lstm.input_size} "
        f"threshold={loaded.unknown_threshold:.2f}"
    )


class KrslRecognizerApp:
    """Desktop controller and view for one-video recognition."""

    def __init__(self, root: tk.Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.holistic_model, self.checkpoint = runtime_paths(project_root)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.video_path = tk.StringVar()
        self.status = tk.StringVar(value="Выберите видео с одним жестом")
        self.result = tk.StringVar(value="Результат появится здесь")
        self.confidence = tk.StringVar(value="Уверенность —")
        self._configure_window()
        self._build_layout()
        self.root.after(100, self._poll_events)

    def _configure_window(self) -> None:
        self.root.title("KRSL · Распознавание жеста")
        self.root.geometry("820x600")
        self.root.minsize(680, 520)
        self.root.configure(background="#F3F5F7")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#F3F5F7")
        style.configure("Panel.TFrame", background="#FFFFFF")
        style.configure(
            "Title.TLabel",
            background="#F3F5F7",
            foreground="#14213D",
            font=("Segoe UI Semibold", 24),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#F3F5F7",
            foreground="#5C667A",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Result.TLabel",
            background="#FFFFFF",
            foreground="#14213D",
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "PanelText.TLabel",
            background="#FFFFFF",
            foreground="#5C667A",
            font=("Segoe UI", 10),
        )
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(18, 10))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(14, 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=28)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        ttk.Label(container, text="KRSL Recognizer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            container,
            text="Тестирование 20 изолированных жестов по готовому видео",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 22))

        file_panel = ttk.Frame(container, style="Panel.TFrame", padding=18)
        file_panel.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        file_panel.columnconfigure(0, weight=1)
        ttk.Label(file_panel, text="Видео", style="PanelText.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.path_label = ttk.Label(
            file_panel,
            textvariable=self.video_path,
            style="PanelText.TLabel",
            wraplength=570,
        )
        self.path_label.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.choose_button = ttk.Button(
            file_panel,
            text="Выбрать видео",
            command=self.choose_video,
            style="Secondary.TButton",
        )
        self.choose_button.grid(row=0, column=1, rowspan=2, padx=(16, 0))

        result_panel = ttk.Frame(container, style="Panel.TFrame", padding=22)
        result_panel.grid(row=3, column=0, sticky="nsew")
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(6, weight=1)
        ttk.Label(result_panel, textvariable=self.status, style="PanelText.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.result_label = ttk.Label(
            result_panel, textvariable=self.result, style="Result.TLabel", wraplength=700
        )
        self.result_label.grid(row=1, column=0, sticky="w", pady=(8, 4))
        ttk.Label(result_panel, textvariable=self.confidence, style="PanelText.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.confidence_bar = ttk.Progressbar(result_panel, maximum=100, value=0)
        self.confidence_bar.grid(row=3, column=0, sticky="ew", pady=(8, 20))
        ttk.Label(
            result_panel, text="Три наиболее вероятных варианта", style="PanelText.TLabel"
        ).grid(row=4, column=0, sticky="w", pady=(0, 7))
        self.candidates = ttk.Treeview(
            result_panel,
            columns=("label", "confidence"),
            show="headings",
            height=3,
            selectmode="none",
        )
        self.candidates.heading("label", text="Жест")
        self.candidates.heading("confidence", text="Уверенность")
        self.candidates.column("label", anchor="w", width=420)
        self.candidates.column("confidence", anchor="center", width=140, stretch=False)
        self.candidates.grid(row=5, column=0, sticky="nsew")

        action_row = ttk.Frame(result_panel, style="Panel.TFrame")
        action_row.grid(row=7, column=0, sticky="ew", pady=(18, 0))
        action_row.columnconfigure(0, weight=1)
        self.activity = ttk.Progressbar(action_row, mode="indeterminate", length=170)
        self.activity.grid(row=0, column=0, sticky="w")
        self.recognize_button = ttk.Button(
            action_row,
            text="Распознать жест",
            command=self.start_recognition,
            style="Primary.TButton",
        )
        self.recognize_button.grid(row=0, column=1, sticky="e")

        ttk.Label(
            container,
            text="Модель экспериментальная и может ошибаться даже при высокой уверенности.",
            style="Subtitle.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.root.bind("<Control-o>", lambda _event: self.choose_video())
        self.root.bind("<Return>", lambda _event: self.start_recognition())

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(title="Выберите видео", filetypes=VIDEO_TYPES)
        if path:
            self.video_path.set(path)
            self.status.set("Видео выбрано — нажмите «Распознать жест»")

    def start_recognition(self) -> None:
        path = Path(self.video_path.get().strip())
        if not self.video_path.get().strip() or not path.is_file():
            messagebox.showwarning("Видео не выбрано", "Выберите существующий видеофайл.")
            return
        self._set_busy(True)
        self.status.set("Извлекаю landmarks и распознаю жест…")
        self.result.set("Обработка видео")
        self.confidence.set("Это может занять несколько секунд")
        self.confidence_bar.configure(value=0)
        self._clear_candidates()
        threading.Thread(target=self._recognize, args=(path,), daemon=True).start()

    def _recognize(self, path: Path) -> None:
        try:
            result = recognize_video(path, self.holistic_model, self.checkpoint)
            self.events.put(("result", result))
        except Exception as error:  # noqa: BLE001 - surfaced safely in the desktop UI
            self.events.put(("error", error))

    def _poll_events(self) -> None:
        try:
            event, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_events)
            return
        self._set_busy(False)
        if event == "result" and isinstance(payload, VideoPrediction):
            self._show_prediction(payload)
        else:
            self.status.set("Не удалось обработать видео")
            self.result.set("Ошибка")
            self.confidence.set("Проверьте файл и попробуйте ещё раз")
            messagebox.showerror("Ошибка распознавания", str(payload))
        self.root.after(100, self._poll_events)

    def _show_prediction(self, prediction: VideoPrediction) -> None:
        view = present_prediction(prediction)
        self.status.set(view.status)
        self.result.set(view.title)
        self.confidence.set(view.confidence_text)
        self.confidence_bar.configure(value=view.confidence_percent)
        self._clear_candidates()
        for label, confidence in view.candidates:
            self.candidates.insert("", "end", values=(label, confidence))

    def _clear_candidates(self) -> None:
        for item in self.candidates.get_children():
            self.candidates.delete(item)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.choose_button.configure(state=state)
        self.recognize_button.configure(state=state)
        if busy:
            self.activity.start(12)
        else:
            self.activity.stop()


def main(project_root: Path | None = None) -> None:
    """Open the KRSL desktop application."""
    root_path = project_root or Path(__file__).resolve().parents[3]
    root = tk.Tk()
    try:
        check_runtime(root_path)
        KrslRecognizerApp(root, root_path)
    except Exception as error:  # noqa: BLE001 - startup errors belong in a dialog
        root.withdraw()
        messagebox.showerror("KRSL не запущен", str(error))
        root.destroy()
        return
    root.mainloop()
