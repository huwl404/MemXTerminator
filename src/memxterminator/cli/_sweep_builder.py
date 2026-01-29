from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from PyQt5 import QtCore, QtWidgets


def _sanitise_suffix(value: object) -> str:
    s = str(value)
    # Keep it short and filesystem-safe.
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")[:48] or "x"


@dataclass(frozen=True)
class SweepSpec:
    param: str
    values: list[Any]


class SweepBuilderDialog(QtWidgets.QDialog):
    """
    Expand a single base job into multiple jobs by sweeping one parameter.
    """

    def __init__(self, *, param_candidates: list[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sweep Builder")
        self.setModal(True)

        self._values: list[Any] = []

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.paramCombo = QtWidgets.QComboBox(self)
        self.paramCombo.addItems(param_candidates)
        self.paramCombo.setToolTip("Which parameter to sweep (creates one job per value).")
        form.addRow("Parameter", self.paramCombo)

        self.modeCombo = QtWidgets.QComboBox(self)
        self.modeCombo.addItems(["List", "Range"])
        self.modeCombo.setToolTip("List: comma-separated values. Range: start/end/step.")
        form.addRow("Mode", self.modeCombo)

        self.listLineEdit = QtWidgets.QLineEdit(self)
        self.listLineEdit.setPlaceholderText("e.g. 0.001,0.002,0.005")
        self.listLineEdit.setToolTip("Comma-separated values. Whitespace is ignored.")
        form.addRow("Values", self.listLineEdit)

        rangeRow = QtWidgets.QHBoxLayout()
        self.rangeStart = QtWidgets.QLineEdit(self)
        self.rangeStart.setPlaceholderText("start")
        self.rangeEnd = QtWidgets.QLineEdit(self)
        self.rangeEnd.setPlaceholderText("end (inclusive)")
        self.rangeStep = QtWidgets.QLineEdit(self)
        self.rangeStep.setPlaceholderText("step")
        for w in (self.rangeStart, self.rangeEnd, self.rangeStep):
            w.setMaximumWidth(120)
        rangeRow.addWidget(self.rangeStart)
        rangeRow.addWidget(QtWidgets.QLabel("→", self))
        rangeRow.addWidget(self.rangeEnd)
        rangeRow.addWidget(QtWidgets.QLabel("step", self))
        rangeRow.addWidget(self.rangeStep)
        rangeWrap = QtWidgets.QWidget(self)
        rangeWrap.setLayout(rangeRow)
        form.addRow("Range", rangeWrap)

        self.previewLabel = QtWidgets.QLabel("", self)
        self.previewLabel.setWordWrap(True)
        layout.addWidget(self.previewLabel)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        layout.addWidget(buttons)

        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.modeCombo.currentIndexChanged.connect(self._update_mode)
        self.listLineEdit.textChanged.connect(self._update_preview)
        self.rangeStart.textChanged.connect(self._update_preview)
        self.rangeEnd.textChanged.connect(self._update_preview)
        self.rangeStep.textChanged.connect(self._update_preview)

        self._update_mode()
        self._update_preview()
        self.resize(520, 220)

    def sweep_spec(self) -> SweepSpec:
        if not self._values:
            raise ValueError("Sweep has no values")
        return SweepSpec(param=str(self.paramCombo.currentText()), values=list(self._values))

    def _update_mode(self) -> None:
        is_list = self.modeCombo.currentText() == "List"
        self.listLineEdit.setEnabled(is_list)
        self.rangeStart.setEnabled(not is_list)
        self.rangeEnd.setEnabled(not is_list)
        self.rangeStep.setEnabled(not is_list)
        self._update_preview()

    def _parse_values(self) -> list[Any]:
        mode = self.modeCombo.currentText()
        if mode == "List":
            raw = self.listLineEdit.text().strip()
            if raw == "":
                return []
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            values: list[Any] = []
            for p in parts:
                # Try int, then float, else keep string.
                try:
                    values.append(int(p))
                    continue
                except Exception:
                    pass
                try:
                    values.append(float(p))
                    continue
                except Exception:
                    pass
                values.append(p)
            return values

        # Range mode (numeric only).
        start_s = self.rangeStart.text().strip()
        end_s = self.rangeEnd.text().strip()
        step_s = self.rangeStep.text().strip()
        if start_s == "" or end_s == "" or step_s == "":
            return []
        start = float(start_s)
        end = float(end_s)
        step = float(step_s)
        if step <= 0:
            return []
        values = []
        x = start
        # Inclusive end with epsilon.
        while x <= end + 1e-12:
            values.append(x)
            x += step
            if len(values) > 500:
                break
        return values

    def _update_preview(self) -> None:
        values = self._parse_values()
        if not values:
            self.previewLabel.setText("Preview: (no values)")
            return
        shown = ", ".join(_sanitise_suffix(v) for v in values[:8])
        more = "" if len(values) <= 8 else f", … (+{len(values) - 8})"
        self.previewLabel.setText(f"Preview: {len(values)} job(s): {shown}{more}")

    def _accept(self) -> None:
        values = self._parse_values()
        if not values:
            QtWidgets.QMessageBox.warning(self, "Invalid sweep", "Please provide at least one sweep value.")
            return
        self._values = values
        self.accept()

