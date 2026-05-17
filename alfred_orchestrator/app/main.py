from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from app.config import Settings
from app.pipeline import AlfredRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alfred Orchestrator CLI")
    parser.add_argument("--mode", choices=["typed", "voice"], default="typed")
    parser.add_argument("--text", help="Run one typed request and exit.")
    parser.add_argument("--speak", action="store_true", help="Speak responses with ElevenLabs TTS.")
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS in voice mode.")
    parser.add_argument("--seconds", type=int, help="Voice recording duration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    settings = Settings.load()

    with AlfredRuntime(settings) as runtime:
        console.print(Panel.fit("Alfred is ready.", title="Alfred"))

        if args.text:
            result = runtime.handle_text(args.text, speak=args.speak)
            _print_result(console, result)
            return

        while True:
            try:
                if args.mode == "voice":
                    input("Press Enter, then ask Alfred a question...")
                    result = runtime.handle_voice(
                        seconds=args.seconds,
                        speak=not args.no_speak,
                    )
                else:
                    text = input("Ask Alfred (or 'quit'): ").strip()
                    if text.lower() in {"q", "quit", "exit"}:
                        break
                    result = runtime.handle_text(text, speak=args.speak)

                _print_result(console, result)
            except KeyboardInterrupt:
                console.print("\nGoodbye.")
                break
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")


def _print_result(console: Console, result) -> None:
    console.print(f"Transcript: {result.request.raw_text}")
    console.print(f"Alfred: {result.response.answer_text}")
    console.print(f"Run artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()
