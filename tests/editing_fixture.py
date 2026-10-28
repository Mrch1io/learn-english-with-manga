from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.runtime import Config, TextBlock
from pipeline.editing import PageEditorSession
from vision import graphics
from pipeline.batch import file_hash


class EditingFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()
        self.input_path = self.root / "original.png"
        self.output_path = self.output_dir / "page.png"
        self.record_path = self.output_path.with_suffix(".json")
        self.clean_path = self.output_dir / "page.clean.png"
        self.font_path = graphics.resolve_font()
        self.config = Config(device="cuda:0", local_files_only=True, work="fixture-comic", translation_backend="qwen",
                             memory_db=str(self.root / "terms.sqlite3"),
                             model_path=str(self.root / "no-model.pt"),
                             adapter_path=str(self.root / "no-adapter"),
                             font_path=self.font_path, min_font_size=12, max_font_size=24)
        self.pool = SimpleNamespace(local_files_only=True, policy="staged", offload=Mock(), close=Mock(),
                                    acquire=Mock(side_effect=AssertionError("No model may load in editing tests")))
        self.original = Image.new("RGB", (640, 260), (185, 185, 185))
        drawing = ImageDraw.Draw(self.original)
        drawing.ellipse((15, 20, 305, 240), fill="white", outline="black", width=3)
        drawing.ellipse((335, 20, 625, 240), fill="white", outline="black", width=3)
        source_font = ImageFont.truetype(self.font_path, 22)
        drawing.text((70, 105), "SOURCE ONE", font=source_font, fill="black")
        drawing.text((390, 105), "SOURCE TWO", font=source_font, fill="black")
        self.original.save(self.input_path)

    def make_result(self, *, with_clean=True, left_failed=False, inferred_regions=False):
        blocks = [
            TextBlock((20, 25, 300, 235), "first OCR source", "Old words here",
                      "left", "failed" if left_failed else "translated",
                      "translation failed" if left_failed else "", safe_box=(60, 65, 260, 195)),
            TextBlock((340, 25, 620, 235), "neighbor OCR source", "Neighbor stays",
                      "right", "translated", safe_box=(380, 65, 580, 195)),
        ]
        clean = self.original.copy()
        draw = ImageDraw.Draw(clean)
        for block in blocks:
            if block.status == "translated":
                draw.rectangle(block.safe_box, fill="white")
        output = clean.copy()
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype(self.font_path, 20)
        line_height = sum(font.getmetrics()) + 2
        for block in blocks:
            safe_box = block.safe_box
            if inferred_regions:
                block.safe_box = None
            if block.status != "translated":
                continue
            text = block.translated_text
            bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            layout = graphics.TextLayout(
                font_size=20, lines=[text], line_height=line_height, line_gap=3,
                width=bounds[2] - bounds[0], height=line_height, stroke_width=1)
            block.metadata["layout"] = dict(
                asdict(layout), safe_box=list(safe_box), font_path=self.font_path,
                signature=[text, list(block.box), list(block.safe_box) if block.safe_box else None,
                           list(self.original.size), self.font_path, "en", 12, 24])
            x = round((safe_box[0] + safe_box[2] - bounds[0] - bounds[2]) / 2)
            y = round((safe_box[1] + safe_box[3] - bounds[1] - bounds[3]) / 2)
            draw.text((x, y), text, font=font, fill="black", stroke_width=1, stroke_fill="white")
        output.save(self.output_path)
        record = {
            "schema_version": 2 if with_clean else 1,
            "input_path": str(self.input_path), "input_sha256": file_hash(self.input_path),
            "output_path": str(self.output_path), "output_sha256": file_hash(self.output_path),
            "image_size": list(self.original.size), "status": "partial" if left_failed else "done",
            "work": self.config.work, "target_language": "en", "config": asdict(self.config),
            "config_fingerprint": "fixture-fingerprint", "terms_fingerprint": "fixture-terms",
            "blocks": [asdict(block) for block in blocks],
        }
        if with_clean:
            clean.save(self.clean_path)
            record["clean_image"] = {
                "file": self.clean_path.name, "sha256": file_hash(self.clean_path),
                "cleaned_block_ids": [block.block_id for block in blocks if block.status == "translated"],
            }
        self.record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        self.fixture_clean = clean
        self.fixture_output = output
        return record

    def open_session(self):
        session = PageEditorSession(self.record_path, pool=self.pool)
        self.addCleanup(session.close)
        return session

    def output_pixels(self):
        with Image.open(self.output_path) as image:
            return np.array(image.convert("RGB"))

    def disk_snapshot(self):
        return {path.name: path.read_bytes() for path in self.output_dir.iterdir() if path.is_file()}

    def assert_uncommitted(self, session, before_disk, before_record):
        self.assertEqual(self.disk_snapshot(), before_disk)
        self.assertEqual(session.record, before_record)
        self.pool.acquire.assert_not_called()

    def make_reading_result(self):
        record = self.make_result()
        record.update(exported=False)
        record["config"]["export_images"] = False
        record.pop("clean_image")
        record.pop("output_sha256")
        for block in record["blocks"]:
            block["metadata"].pop("layout", None)
        self.clean_path.unlink()
        self.output_path.unlink()
        self.record_path.write_text(json.dumps(record), encoding="utf-8")
        return json.loads(json.dumps(record))

