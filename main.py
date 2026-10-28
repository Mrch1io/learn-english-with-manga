"""命令行入口；launch.bat 打开 Tk 界面。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.editing import PageEditorSession
from core.runtime import Cancelled, Config, OUTPUT_IMAGE, ROOT


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Japanese-to-English manga translation")
    p.add_argument("input", nargs="*", help="Images or folders (subfolders included); required for translation")
    p.add_argument("-o", "--output", default=OUTPUT_IMAGE, help="Separate output directory")
    p.add_argument("--gui", action="store_true", help="Open the simple desktop interface")
    p.add_argument("--read", metavar="PAGE_OR_FOLDER", help="Open original-page reader and Japanese speech controls")
    p.add_argument("--text-only", action="store_true", help="Save OCR/translation JSON for reading; defer image export")
    p.add_argument("--analyze-characters", metavar="PAGE_OR_FOLDER", help="Run optional Magi v2 character suggestions")
    p.add_argument("--magi-model", default=str(ROOT / "runtime" / "magiv2"))
    p.add_argument("--bubbles-only", action="store_true", help="Disable Magi page-text detection; only translate YOLO bubbles")
    p.add_argument("--ocr-candidate-review", action="store_true",
                   help="Opt in to box-relation audit and manual local OCR candidate commands")
    p.add_argument("--hybrid-ocr", action="store_true",
                   help="Use local PaddleOCR-VL-1.6 for empty or low-confidence Manga-OCR output")
    p.add_argument("--ocr-prob-threshold", type=float, default=Config.ocr_prob_threshold,
                   help="Manga token likelihood threshold (0..1, default 0.1); not calibrated accuracy")
    p.add_argument("--paddle-ocr-model", default=Config.paddle_ocr_model_path,
                   help="Local PaddleOCR-VL-1.6 directory; never downloaded automatically")
    p.add_argument("--work", default="default", help="Manga title / terminology namespace")
    p.add_argument("--target", choices=["en"], default="en")
    p.add_argument("--device", default="cuda:0", help="CUDA device, default cuda:0; CPU inference is disabled")
    p.add_argument("--memory-policy", choices=["staged", "resident"], default="staged")
    p.add_argument("--translator", choices=["vntl", "qwen"], default="vntl",
                   help="VNTL GGUF with CUDA llama.cpp (default), or the previous Qwen/LoRA model")
    p.add_argument("--gguf", default=Config.gguf_path, help="VNTL Q4_K_M GGUF path")
    p.add_argument("--llama-server", default=Config.llama_server_path, help="CUDA llama-server.exe path")
    p.add_argument("--llama-context", type=int, default=Config.llama_context_size,
                   help="llama.cpp context tokens, default 2048")
    p.add_argument("--db", default=str(ROOT/"data"/"memory.sqlite3"))
    p.add_argument("--font", help="TTF/OTF/TTC font path; glyph coverage is checked")
    p.add_argument("--min-font-size", type=int, default=12)
    p.add_argument("--max-font-size", type=int, default=40)
    p.add_argument("--direction", choices=["rtl", "ltr"], default="rtl")
    p.add_argument("--mode", choices=["auto", "individual", "grouped"], default="auto",
                   help="auto: grouped VNTL or individual Qwen; explicit modes are available for comparison")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--context-chars", type=int, default=1200)
    p.add_argument("--no-fragment-links", action="store_true",
                   help="Disable automatic nearby fragment linking; retain explicit prohibition/permission rules")
    p.add_argument("--no-speaker-context", action="store_true",
                   help="Do not send confirmed character bindings to the translation model")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--retries", type=int, choices=[0, 1], default=1)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--no-extract-terms", action="store_true")
    p.add_argument("--debug-mask", action="store_true")
    p.add_argument("--skip-inpaint", action="store_true", help="Diagnostic only: leaves original text visible")
    p.add_argument("--offline", action="store_true", help="Use cached models only")
    p.add_argument("--force", action="store_true", help="Reprocess even valid cached results")
    p.add_argument("--list-terms", action="store_true")
    p.add_argument("--import-terms", metavar="JSON")
    p.add_argument("--export-terms", metavar="JSON")
    p.add_argument("--set-term", nargs=2, metavar=("SOURCE", "TARGET"))
    p.add_argument("--alias", action="append", default=[])
    p.add_argument("--lock", action="store_true")
    p.add_argument("--reading", help="Japanese kana pronunciation for --set-term")
    p.add_argument("--delete-term", metavar="SOURCE")
    p.add_argument("--edit-page", metavar="PNG_OR_JSON", help="Edit an existing translated page")
    p.add_argument("--bubble", metavar="ID", help="Stable bubble ID from --list-bubbles")
    edit = p.add_mutually_exclusive_group()
    edit.add_argument("--list-bubbles", action="store_true")
    edit.add_argument("--set-translation", metavar="TEXT", help="Save text and redraw from the clean base only")
    edit.add_argument("--set-translation-file", metavar="UTF8_FILE")
    edit.add_argument("--set-ocr", metavar="TEXT", help="Correct OCR and retranslate only --bubble")
    edit.add_argument("--set-ocr-file", metavar="UTF8_FILE")
    edit.add_argument("--export-image", action="store_true", help="Export saved texts without repeating OCR or translation")
    edit.add_argument("--reclean", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Add a small cleanup region inside --bubble")
    edit.add_argument("--preview-ocr", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                      help="Preview one exact original-image crop using cached Manga-OCR; texts stay unchanged")
    edit.add_argument("--apply-ocr-candidate", metavar="CANDIDATE_ID",
                      help="Explicitly adopt a saved OCR candidate and retranslate only --bubble")
    p.add_argument("--list-voices", action="store_true", help="List actual voices from the local VOICEVOX engine")
    p.add_argument("--tts-url", default="http://127.0.0.1:50021")
    p.add_argument("--speak-page", metavar="JSON", help="Synthesize revised Japanese from --bubble, never the English text")
    p.add_argument("--voice-style", type=int, help="Style ID returned by --list-voices")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--audio-cache", default=str(ROOT / "data" / "audio"))
    return p


def _validate_cli(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    editing_action = (args.list_bubbles or args.export_image or args.reclean is not None or any(value is not None for value in
        (args.set_translation, args.set_translation_file, args.set_ocr, args.set_ocr_file,
         args.preview_ocr, args.apply_ocr_candidate)))
    if (editing_action or (args.bubble is not None and not args.speak_page)) and not args.edit_page:
        cli.error("Editing options require --edit-page PNG_OR_JSON")
    if args.gui and args.edit_page:
        cli.error("Use --gui and the page editor, or use --edit-page for CLI editing")
    if (args.preview_ocr is not None or args.apply_ocr_candidate is not None) and not args.ocr_candidate_review:
        cli.error("Local OCR candidates require --ocr-candidate-review")


def _config(args: argparse.Namespace) -> Config:
    return Config(work=args.work, target_language=args.target, device=args.device,
                    translation_backend=args.translator, gguf_path=args.gguf,
                    llama_server_path=args.llama_server, llama_context_size=args.llama_context,
                    memory_policy=args.memory_policy, memory_db=args.db, font_path=args.font,
                    min_font_size=args.min_font_size, max_font_size=args.max_font_size,
                    reading_direction=args.direction, translation_mode=args.mode,
                    use_fragment_links=not args.no_fragment_links,
                    use_speaker_context=not args.no_speaker_context,
                    group_size=args.group_size, context_chars=args.context_chars,
                    max_new_tokens=args.max_new_tokens, retries=args.retries,
                    detection_conf=args.conf, extract_terms=not args.no_extract_terms,
                    detect_free_text=not args.bubbles_only, magi_model_path=args.magi_model,
                    ocr_candidate_review=args.ocr_candidate_review,
                    use_hybrid_ocr=args.hybrid_ocr, ocr_prob_threshold=args.ocr_prob_threshold,
                    paddle_ocr_model_path=args.paddle_ocr_model,
                    local_files_only=args.offline, debug_mask=args.debug_mask,
                    skip_inpaint=args.skip_inpaint, export_images=not args.text_only)


def _analysis_command(args: argparse.Namespace) -> int:
    from vision.characters import analyze_pages
    cancelled = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    def check_analysis_cancel() -> None:
        if cancelled.is_set():
            raise Cancelled("Magi analysis cancelled after the current safe step")
    try:
        report = analyze_pages(args.analyze_characters, model_path=args.magi_model,
            device=args.device, cancel=check_analysis_cancel,
            progress=lambda event: print(event["message"], flush=True))
    finally:
        signal.signal(signal.SIGINT, previous)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 130 if report.get("cancelled") else (1 if report.get("failed") else 0)


def _edit_request(session: PageEditorSession, args: argparse.Namespace,
                  edit_cancel: Callable[[], None]) -> int:
    if args.offline:
        session.config.local_files_only = True
        session.pool.local_files_only = True
    if args.list_bubbles:
        print(json.dumps(session.record["blocks"], ensure_ascii=False, indent=2))
        return 0
    if args.export_image:
        result = session.export_image(cancel=edit_cancel)
        print(f"Saved: {result['output_path']}")
        return 0
    if not args.bubble:
        raise ValueError("Editing requires --bubble ID; use --list-bubbles first")
    source_text, translated_text = args.set_ocr, args.set_translation
    if args.set_ocr_file:
        source_text = Path(args.set_ocr_file).read_text(encoding="utf-8-sig")
    if args.set_translation_file:
        translated_text = Path(args.set_translation_file).read_text(encoding="utf-8-sig")
    if args.preview_ocr is not None:
        candidate = session.preview_ocr(args.bubble, args.preview_ocr,
                                        enabled=True, cancel=edit_cancel)
        print(json.dumps(candidate, ensure_ascii=False, indent=2))
        return 0 if candidate["status"] == "ok" else 1
    if args.apply_ocr_candidate is not None:
        result = session.apply_ocr_candidate(args.bubble, args.apply_ocr_candidate,
                                             enabled=True, cancel=edit_cancel)
    elif args.reclean is not None:
        result = session.reclean(args.bubble, [args.reclean], cancel=edit_cancel)
    elif source_text is not None:
        result = session.retranslate(args.bubble, source_text, cancel=edit_cancel)
    elif translated_text is not None:
        result = session.rerender(args.bubble, translated_text, cancel=edit_cancel)
    else:
        raise ValueError("Choose --set-translation, --set-ocr, or a UTF-8 text file option")
    print(f"Saved: {result['output_path']}\n{result['message']}")
    return 0


def _edit_command(args: argparse.Namespace) -> int:
    from pipeline.editing import PageEditorSession
    session = PageEditorSession(args.edit_page)
    cancelled = threading.Event()
    def edit_cancel() -> None:
        if cancelled.is_set():
            raise Cancelled("Edit cancelled; previous saved result retained")
    def edit_interrupt(signum: int, frame: FrameType | None) -> None:
        cancelled.set()
        print("Cancellation requested; finishing the current safe step...", flush=True)
    previous = signal.signal(signal.SIGINT, edit_interrupt)
    try:
        return _edit_request(session, args, edit_cancel)
    finally:
        signal.signal(signal.SIGINT, previous)
        session.close()


def _terms_command(args: argparse.Namespace, config: Config) -> int:
    from core.memory import GlossaryMemory
    with GlossaryMemory(config.memory_db, config.work, config.target_language) as memory:
        legacy = ROOT / "glossary.json"
        if config.target_language == "en" and legacy.is_file():
            memory.import_legacy_once(legacy)
        if args.import_terms:
            print(f"Imported {memory.import_json(args.import_terms)} new terms")
        if args.set_term:
            source, target = args.set_term
            affected = set(memory.affected_pages(source))
            memory.upsert_term(source, target, aliases=args.alias, locked=args.lock, status="confirmed", user_edit=True,
                               reading=args.reading)
            affected.update(memory.affected_pages(source))
            print(f"Updated; rerun the input folder to retranslate {len(affected)} affected pages")
        if args.delete_term:
            memory.delete_term(args.delete_term, user_edit=True)
        if args.export_terms:
            memory.export_json(args.export_terms)
        if args.list_terms:
            print(json.dumps(memory.list_terms(), ensure_ascii=False, indent=2))
    return 0


def _batch_command(args: argparse.Namespace, config: Config) -> int:
    from pipeline.batch import run_batch
    cancelled = threading.Event()
    def interrupt(signum: int, frame: FrameType | None) -> None:
        cancelled.set()
        print("Cancellation requested; finishing the current safe step...", flush=True)
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        def progress(event: dict[str, object]) -> None:
            position = f"[{event['current']}/{event['total']}] " if "current" in event else "  "
            print(position+str(event.get("message", event["type"])), flush=True)
        inputs = args.input[0] if len(args.input) == 1 else args.input
        report = run_batch(inputs, args.output, config, progress=progress,
                           cancel_event=cancelled, force=args.force)
    finally:
        signal.signal(signal.SIGINT, previous)
    print(f"Completed: {report['completed']}; partial: {report['partial']}; failed: {report['failed']}; skipped: {report['skipped']}")
    print(f"Report: {report['report_path']}")
    return 130 if report["cancelled"] else (1 if report["failed"] or report["partial"] else 0)


def _dispatch(cli: argparse.ArgumentParser, args: argparse.Namespace, config: Config) -> int:
    if args.read:
        from reader import launch_reader
        launch_reader(args.read, config=config, output_dir=args.output)
        return 0
    if args.analyze_characters:
        return _analysis_command(args)
    if args.list_voices or args.speak_page:
        return speech_command(args)
    if args.edit_page:
        return _edit_command(args)
    if any([args.list_terms, args.import_terms, args.export_terms, args.set_term, args.delete_term]):
        return _terms_command(args, config)
    if not args.input:
        cli.error("Translation requires at least one image or folder; use --gui to choose inputs")
    return _batch_command(args, config)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    _validate_cli(cli, args)
    if args.gui:
        from ui.app import launch_gui
        launch_gui()
        return 0
    config = _config(args)
    try:
        config.validate()
        return _dispatch(cli, args, config)
    except Cancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def speech_command(args: argparse.Namespace) -> int:
    from core.memory import GlossaryMemory
    from audio.speech import VoicevoxClient, list_voices, synthesize_cached
    client = VoicevoxClient(args.tts_url)
    voices = list_voices(client.get_catalog())
    if args.list_voices:
        print(json.dumps(voices, ensure_ascii=False, indent=2))
        return 0
    if not args.bubble:
        raise ValueError("--speak-page requires --bubble ID")
    path = Path(args.speak_page)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    page = json.loads(path.read_text(encoding="utf-8"))
    block = next((block for block in page["blocks"] if block["block_id"] == args.bubble), None)
    if block is None:
        raise ValueError("Unknown bubble ID")
    with GlossaryMemory(page.get("config", {}).get("memory_db", args.db), page["work"], page["target_language"]) as memory:
        binding = memory.get_character_binding(page["input_path"], args.bubble)
        character_id = binding["character_id"] if binding else "unknown"
        characters = {row["id"]: row for row in memory.list_characters()}
        character = characters[character_id]
        voice = character.get("voice") or characters["unknown"].get("voice") or {}
        if args.voice_style is not None:
            selected = next((row for row in voices if row["voice"]["style_id"] == args.voice_style), None)
            if selected is None:
                raise ValueError("Voice style not available; use --list-voices")
            voice = selected["voice"]
        if not voice:
            raise ValueError("No voice assigned to this character. Select one in the reader or use --voice-style from --list-voices.")
        voice = {**voice, "params": {**voice.get("params", {}), "speedScale": args.speed}}
        result = synthesize_cached(client, block["source_text"], reading_terms=memory.match(block["source_text"]),
                                   voice=voice, cache_dir=args.audio_cache)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def entrypoint() -> int:
    # pythonw 没有控制台流，模型日志和进度条需要 UTF-8 输出。
    if sys.stdout is None or sys.stderr is None:
        log_dir = ROOT / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        desktop_log = (log_dir / "desktop.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or desktop_log
        sys.stderr = sys.stderr or desktop_log
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
