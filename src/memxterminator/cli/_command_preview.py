from __future__ import annotations

import shlex
from typing import Sequence

from PyQt5 import QtCore, QtGui, QtWidgets


class CommandPreviewDialog(QtWidgets.QDialog):
    def __init__(
        self,
        argv: Sequence[object],
        parent: QtWidgets.QWidget | None = None,
        *,
        title: str = "Command Preview",
        hint: str | None = "Copy and edit this command in a terminal (e.g. add --adopt_existing_outputs).",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._argv = [str(a) for a in argv]
        self._cmd_text = shlex.join(self._argv)

        layout = QtWidgets.QVBoxLayout(self)

        if hint:
            hint_label = QtWidgets.QLabel(hint, self)
            hint_label.setWordWrap(True)
            layout.addWidget(hint_label)

        text = QtWidgets.QPlainTextEdit(self)
        text.setReadOnly(True)
        text.setPlainText(self._cmd_text)
        text.setMinimumHeight(120)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(text)

        buttons = QtWidgets.QDialogButtonBox(self)
        copy_button = buttons.addButton("Copy", QtWidgets.QDialogButtonBox.ActionRole)
        close_button = buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)

        copy_button.clicked.connect(lambda: self._copy_to_clipboard(copy_button))
        close_button.clicked.connect(self.close)

        self.resize(820, 220)

    def _copy_to_clipboard(self, button: QtWidgets.QAbstractButton) -> None:
        QtWidgets.QApplication.clipboard().setText(self._cmd_text)
        original = button.text()
        button.setText("Copied")
        QtCore.QTimer.singleShot(1500, lambda: button.setText(original))
