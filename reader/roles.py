from __future__ import annotations

import copy

from .messages import BindingRequest, CharacterRequest, Command, JsonObject, MagiRequest

_CHARACTER_LABELS = {
    ("unknown", "未知角色"): "Default voice",
    ("unknown", "Unknown character"): "Default voice",
    ("narrator", "旁白"): "Narrator",
}


def _character_name(character: JsonObject) -> str:
    """Translate built-in labels for display without changing saved names."""
    name = character["name"]
    return _CHARACTER_LABELS.get((character["id"], name), name)

def _character_labels(characters: list[JsonObject]) -> list[str]:
    """Keep dropdown values unique without exposing database identifiers."""
    names = [_character_name(character) for character in characters]
    reserved, used, labels, next_suffix = set(names), set(), [], {}
    for name in names:
        label = name
        if label in used:
            suffix = next_suffix.get(name, 2)
            while label in used or label in reserved:
                label = f"{name} ({suffix})"
                suffix += 1
            next_suffix[name] = suffix
        used.add(label)
        labels.append(label)
    return labels


class RolesMixin:
    def _role_changed(self, _event=None):
        self._invalidate()
        index = self.role_box.current()
        if index >= 0:
            self._apply_role(self.characters[index])

    def _save_role(self, new=False):
        if self.page_index < 0 or self.loading or not self.pages[self.page_index]["path"]:
            self.status.set("Translate a page first. Characters are saved with the current manga.")
            return
        name = self.role_name.get().strip()
        if not name:
            self.status.set("Enter a character name.")
            return
        index = self.role_box.current()
        if not new and index < 0:
            return
        self._invalidate()
        self.commands.put(Command("character", CharacterRequest(
            self.pages[self.page_index]["path"], None if new else self.characters[index]["id"],
            name, self.current_voice), self.page_token))

    def _bind_role(self):
        index = self.role_box.current()
        if self.page_index < 0 or self.block_id is None or index < 0 or self.loading:
            return
        self._invalidate()
        self.commands.put(Command("bind", BindingRequest(
            self.pages[self.page_index]["path"], self.block_id, self.characters[index]["id"]), self.page_token))

    def _set_magi_busy(self, value):
        self.magi_busy = value
        self.cancel_button.configure(state="normal" if value else "disabled")
        self._set_controls_busy(value)
        if not value:
            self._schedule_thumbnails()

    def _suggest_characters(self):
        if not self.pages or self.loading or self.magi_busy or self.translation_busy:
            return
        paths = [page["path"] for page in self.pages if page["path"]]
        if not paths:
            self.status.set("Translate the page before requesting Magi character suggestions.")
            return
        token = self._invalidate()
        self.operation_cancel.clear()
        self._set_magi_busy(True)
        self.status.set("Magi is analyzing characters. Confirmed bindings are preserved; use Stop to cancel safely.")
        self.commands.put(Command("magi", MagiRequest(paths, self.config.magi_model_path, self.config.device), token))

    def _set_roles(self, character_id="unknown"):
        self.role_box.configure(values=_character_labels(self.characters))
        index = next((i for i, character in enumerate(self.characters) if character["id"] == character_id), -1)
        if index >= 0:
            self.role_box.current(index)
            self._apply_role(self.characters[index])

    def _apply_role(self, character):
        self.role_name.set(_character_name(character))
        fallback = next((c.get("voice") for c in self.characters if c["id"] == "unknown"), None)
        self.current_voice = copy.deepcopy(character.get("voice") or fallback or None)
        self.temporary_voice = False
        self._choose_temporary_voice()
        self._show_voice()
        self._voice_status_ready()

    def _refresh_rows(self):
        for key in self.blocks:
            binding = self.bindings.get(key) or {}
            role_id = binding.get("character_id", "unknown")
            name = next((_character_name(c) for c in self.characters if c["id"] == role_id), "Default voice")
            if binding and not binding.get("confirmed"):
                name += " (suggested)"
            self.bubble_list.item(key, values=(key.rsplit(":", 1)[-1], name))
