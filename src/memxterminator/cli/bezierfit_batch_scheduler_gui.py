from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt5 import QtCore, QtGui, QtWidgets

from memxterminator.bezierfit.scheduler.spec import (
    BezierfitJob,
    JobResources,
    JobSpecFile,
    SchedulerSpec,
    build_job_argv,
    load_spec_file,
    parse_gpu_list,
    parse_spec_dict,
)
from memxterminator.path_resolve import infer_input_base_dir, normalise_dir

from ._command_preview import CommandPreviewDialog
from ._deps import check_cupy_cuda_available
from ._process import popen_kwargs_for_new_session, python_executable_for_subprocess, terminate_pid
from ._sweep_builder import SweepBuilderDialog, _sanitise_suffix


_KIND_TO_LABEL = {
    "bezierfit_particle_pms": "Particle PMS (mem_subtract_main)",
    "bezierfit_micrograph_mms": "Micrograph MMS (micrograph_mem_subtract_main)",
    "bezierfit_mem_analyze": "Membrane Analyzer (mem_analyze_main)",
}
_LABEL_TO_KIND = {v: k for k, v in _KIND_TO_LABEL.items()}


def _safe_int(text: str, *, default: int | None = None) -> int | None:
    s = str(text).strip()
    if s == "":
        return default
    return int(s)


def _safe_float(text: str, *, default: float | None = None) -> float | None:
    s = str(text).strip()
    if s == "":
        return default
    return float(s)


def _open_folder(path: str) -> None:
    url = QtCore.QUrl.fromLocalFile(os.fspath(path))
    QtGui.QDesktopServices.openUrl(url)


@dataclass
class _GuiJob:
    enabled: bool = True
    job_id: str = "job_001"
    kind: str = "bezierfit_particle_pms"
    gpus: int = 1
    procs: int | None = None
    output_root: str = ""
    custom_output_root: bool = False
    input_base_dir: str = ""
    custom_input_base_dir: bool = False
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = "queued"
    assigned_gpus: list[int] = field(default_factory=list)
    pid: int | None = None
    returncode: int | None = None


class BezierfitBatchSchedulerDialog(QtWidgets.QDialog):
    """
    GUI for building and running Bezierfit batches without editing JSON manually.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bezierfit Batch Scheduler - MemXTerminator")
        self.setModal(False)

        self._jobs: list[_GuiJob] = []
        self._selected_row: int | None = None
        self._updating_table = False
        self._dependency_errors: list[str] = []

        self._run_root: str = ""
        self._spec_path: str = ""
        self._state_path: str = ""
        self._log_path: str = ""
        self._pid_path: str = ""

        self._batch_popen: subprocess.Popen | None = None
        self._log_last_pos = 0

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self._build_config_group())
        root.addWidget(self._build_main_splitter(), stretch=1)
        root.addWidget(self._build_run_controls())
        root.addWidget(self._build_log_group(), stretch=1)

        self._log_timer = QtCore.QTimer(self)
        self._log_timer.timeout.connect(self._update_log_view)
        self._log_timer.start(500)

        self._state_timer = QtCore.QTimer(self)
        self._state_timer.timeout.connect(self._update_state_view)
        self._state_timer.start(500)

        # Start with one default job for discoverability.
        self._add_default_job()
        self.resize(1100, 720)

    # ------------------------- UI builders -------------------------

    def _build_config_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Batch configuration", self)
        layout = QtWidgets.QGridLayout(box)

        self.runRootLineEdit = QtWidgets.QLineEdit(box)
        self.runRootLineEdit.setPlaceholderText("/path/to/new/batch_run_root")
        self.runRootLineEdit.setToolTip(
            "All jobs write into subfolders under this directory.\n"
            "Recommendation: use a new empty folder per batch run."
        )
        self.runRootBrowseButton = QtWidgets.QPushButton("Browse...", box)
        self.runRootBrowseButton.clicked.connect(self._browse_run_root)

        self.gpusLineEdit = QtWidgets.QLineEdit(box)
        self.gpusLineEdit.setPlaceholderText("0,1,2,3")
        self.gpusLineEdit.setToolTip(
            "Comma-separated list of GPU ids available to the batch.\n"
            "The scheduler assigns GPUs per job via CUDA_VISIBLE_DEVICES."
        )
        self.gpusAutoButton = QtWidgets.QPushButton("Auto-detect", box)
        self.gpusAutoButton.setToolTip("Detect visible CUDA devices via CuPy.")
        self.gpusAutoButton.clicked.connect(self._auto_detect_gpus)

        self.policyCombo = QtWidgets.QComboBox(box)
        self.policyCombo.addItems(["fill_first (Recommended)", "round_robin"])
        self.policyCombo.setToolTip(
            "fill_first: pack jobs onto the lowest-index free GPUs.\n"
            "round_robin: spread allocations across GPUs more evenly."
        )

        self.maxRunningSpin = QtWidgets.QSpinBox(box)
        self.maxRunningSpin.setMinimum(1)
        self.maxRunningSpin.setMaximum(999)
        self.maxRunningSpin.setValue(1)
        self.maxRunningSpin.setToolTip("Maximum number of jobs running concurrently.")

        self.failFastCheck = QtWidgets.QCheckBox("Fail-fast", box)
        self.failFastCheck.setChecked(True)
        self.failFastCheck.setEnabled(False)
        self.failFastCheck.setToolTip("Fail-fast is mandatory: stop the batch on first job failure.")

        layout.addWidget(QtWidgets.QLabel("Run root", box), 0, 0)
        layout.addWidget(self.runRootLineEdit, 0, 1)
        layout.addWidget(self.runRootBrowseButton, 0, 2)

        layout.addWidget(QtWidgets.QLabel("GPUs", box), 1, 0)
        layout.addWidget(self.gpusLineEdit, 1, 1)
        layout.addWidget(self.gpusAutoButton, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Policy", box), 2, 0)
        layout.addWidget(self.policyCombo, 2, 1)
        layout.addWidget(self.failFastCheck, 2, 2)

        layout.addWidget(QtWidgets.QLabel("Max running jobs", box), 3, 0)
        layout.addWidget(self.maxRunningSpin, 3, 1)

        self.runRootLineEdit.editingFinished.connect(self._on_run_root_changed)
        return box

    def _build_main_splitter(self) -> QtWidgets.QSplitter:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        left = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.jobsTable = QtWidgets.QTableWidget(left)
        self.jobsTable.setColumnCount(8)
        self.jobsTable.setHorizontalHeaderLabels(
            ["Enabled", "Job ID", "Kind", "GPUs", "Procs", "Output root", "Status", "Assigned GPUs"]
        )
        self.jobsTable.horizontalHeader().setStretchLastSection(True)
        self.jobsTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.jobsTable.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.jobsTable.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.jobsTable.itemChanged.connect(self._on_table_item_changed)
        left_layout.addWidget(self.jobsTable, stretch=1)

        buttons_row = QtWidgets.QHBoxLayout()
        self.addJobButton = QtWidgets.QPushButton("Add job", left)
        self.dupJobButton = QtWidgets.QPushButton("Duplicate", left)
        self.removeJobButton = QtWidgets.QPushButton("Remove", left)
        self.sweepButton = QtWidgets.QPushButton("Sweep…", left)
        self.importButton = QtWidgets.QPushButton("Import…", left)
        self.exportButton = QtWidgets.QPushButton("Export…", left)

        self.addJobButton.setToolTip("Create a new job and edit it in the panel on the right.")
        self.dupJobButton.setToolTip("Duplicate the selected job (useful for manual parameter sweeps).")
        self.removeJobButton.setToolTip("Remove the selected job from the batch.")
        self.sweepButton.setToolTip("Generate multiple jobs by sweeping one parameter of the selected job.")
        self.importButton.setToolTip("Load a batch spec JSON.")
        self.exportButton.setToolTip("Save a batch spec JSON (reproducibility).")

        self.addJobButton.clicked.connect(self._on_add_job)
        self.dupJobButton.clicked.connect(self._on_duplicate_job)
        self.removeJobButton.clicked.connect(self._on_remove_job)
        self.sweepButton.clicked.connect(self._on_sweep_job)
        self.importButton.clicked.connect(self._on_import_spec)
        self.exportButton.clicked.connect(self._on_export_spec)

        for b in [self.addJobButton, self.dupJobButton, self.removeJobButton, self.sweepButton, self.importButton, self.exportButton]:
            buttons_row.addWidget(b)
        buttons_row.addStretch(1)
        left_layout.addLayout(buttons_row)

        right = self._build_editor_panel()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([600, 500])
        return splitter

    def _build_editor_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.editorTitle = QtWidgets.QLabel("Job editor", panel)
        font = self.editorTitle.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.editorTitle.setFont(font)
        layout.addWidget(self.editorTitle)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.jobIdEdit = QtWidgets.QLineEdit(panel)
        self.jobIdEdit.setToolTip("Unique job id (used for output folder name).")
        self.kindCombo = QtWidgets.QComboBox(panel)
        self.kindCombo.addItems([_KIND_TO_LABEL[k] for k in _KIND_TO_LABEL.keys()])
        self.kindCombo.setToolTip("Select which Bezierfit task to run.")

        self.gpusSpin = QtWidgets.QSpinBox(panel)
        self.gpusSpin.setMinimum(1)
        self.gpusSpin.setMaximum(64)
        self.gpusSpin.setToolTip("GPUs requested for this job.")

        self.procsEdit = QtWidgets.QLineEdit(panel)
        self.procsEdit.setPlaceholderText("(auto)")
        self.procsEdit.setToolTip("Worker processes per job. Blank = auto (defaults to GPUs/job).")

        self.customOutCheck = QtWidgets.QCheckBox("Custom output root", panel)
        self.customOutCheck.setToolTip(
            "When disabled, output_root is auto-derived as <run_root>/<job_id>.\n"
            "Enable only if you need a custom per-job folder."
        )
        self.outputRootEdit = QtWidgets.QLineEdit(panel)
        self.outputRootEdit.setToolTip("Per-job output root directory.")
        self.outputRootBrowse = QtWidgets.QPushButton("Browse...", panel)
        self.outputRootBrowse.clicked.connect(self._browse_job_output_root)

        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.outputRootEdit, stretch=1)
        out_row.addWidget(self.outputRootBrowse)
        out_wrap = QtWidgets.QWidget(panel)
        out_wrap.setLayout(out_row)

        self.customInputBaseDirCheck = QtWidgets.QCheckBox("Custom input base dir", panel)
        self.customInputBaseDirCheck.setToolTip(
            "When disabled, input_base_dir is auto-inferred from this job's primary input file.\n"
            "input_base_dir is used to resolve relative paths embedded inside CryoSPARC .cs / STAR files\n"
            "(e.g. 'J220/extract/...') when running in batch mode."
        )
        self.inputBaseDirEdit = QtWidgets.QLineEdit(panel)
        self.inputBaseDirEdit.setPlaceholderText("(auto)")
        self.inputBaseDirEdit.setToolTip(
            "Base directory used to resolve relative paths stored inside CryoSPARC .cs / STAR files.\n"
            "If your .cs/.star contains paths like 'J220/extract/...', set this to the CryoSPARC project root."
        )
        self.inputBaseDirBrowse = QtWidgets.QPushButton("Browse...", panel)
        self.inputBaseDirBrowse.clicked.connect(self._browse_input_base_dir)

        base_row = QtWidgets.QHBoxLayout()
        base_row.addWidget(self.inputBaseDirEdit, stretch=1)
        base_row.addWidget(self.inputBaseDirBrowse)
        base_wrap = QtWidgets.QWidget(panel)
        base_wrap.setLayout(base_row)

        form.addRow("Job ID", self.jobIdEdit)
        form.addRow("Kind", self.kindCombo)
        form.addRow("GPUs/job", self.gpusSpin)
        form.addRow("Procs/job", self.procsEdit)
        form.addRow(self.customOutCheck, out_wrap)
        form.addRow(self.customInputBaseDirCheck, base_wrap)

        self.kindStack = QtWidgets.QStackedWidget(panel)
        self.kindStack.addWidget(self._build_page_mem_analyze(panel))
        self.kindStack.addWidget(self._build_page_particle_pms(panel))
        self.kindStack.addWidget(self._build_page_micrograph_mms(panel))
        layout.addWidget(self.kindStack, stretch=1)

        # Wire editor change events.
        self.jobIdEdit.editingFinished.connect(self._apply_editor_to_selected_job)
        self.kindCombo.currentIndexChanged.connect(self._on_kind_changed)
        self.gpusSpin.valueChanged.connect(self._apply_editor_to_selected_job)
        self.procsEdit.editingFinished.connect(self._apply_editor_to_selected_job)
        self.customOutCheck.stateChanged.connect(self._on_custom_output_root_changed)
        self.outputRootEdit.editingFinished.connect(self._apply_editor_to_selected_job)
        self.customInputBaseDirCheck.stateChanged.connect(self._on_custom_input_base_dir_changed)
        self.inputBaseDirEdit.editingFinished.connect(self._apply_editor_to_selected_job)

        return panel

    # ------------------------- Job pages -------------------------

    def _build_page_particle_pms(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)

        self.pmsParticleEdit = QtWidgets.QLineEdit(page)
        self.pmsParticleBrowse = QtWidgets.QPushButton("Browse...", page)
        self.pmsParticleBrowse.clicked.connect(lambda: self._browse_file(self.pmsParticleEdit, "CS Files (*.cs)"))

        self.pmsTemplateEdit = QtWidgets.QLineEdit(page)
        self.pmsTemplateBrowse = QtWidgets.QPushButton("Browse...", page)
        self.pmsTemplateBrowse.clicked.connect(lambda: self._browse_file(self.pmsTemplateEdit, "CS Files (*.cs)"))

        self.pmsControlPointsEdit = QtWidgets.QLineEdit(page)
        self.pmsControlPointsBrowse = QtWidgets.QPushButton("Browse...", page)
        self.pmsControlPointsBrowse.clicked.connect(lambda: self._browse_file(self.pmsControlPointsEdit, "JSON Files (*.json)"))

        self.pmsPointsStepEdit = QtWidgets.QLineEdit(page)
        self.pmsPointsStepEdit.setPlaceholderText("0.001")
        self.pmsPointsStepEdit.setToolTip("Bezier curve sampling step. Smaller = more points (slower).")

        self.pmsPhysDistEdit = QtWidgets.QLineEdit(page)
        self.pmsPhysDistEdit.setPlaceholderText("35")
        self.pmsPhysDistEdit.setToolTip("Physical membrane distance in Å.")

        self.pmsBatchSizeEdit = QtWidgets.QLineEdit(page)
        self.pmsBatchSizeEdit.setPlaceholderText("20")
        self.pmsBatchSizeEdit.setToolTip("Minibatch size when running multiple stacks per job.")

        self.pmsOutputDirnameEdit = QtWidgets.QLineEdit(page)
        self.pmsOutputDirnameEdit.setPlaceholderText("subtracted")
        self.pmsOutputDirnameEdit.setToolTip(
            "Output folder name under output_root (default: subtracted). "
            "Use different names to avoid output collisions."
        )

        self.pmsResumeCheck = QtWidgets.QCheckBox("Resume (.mxt)", page)
        self.pmsResumeCheck.setChecked(True)
        self.pmsForceCheck = QtWidgets.QCheckBox("Force recompute", page)
        self.pmsAdoptCheck = QtWidgets.QCheckBox("Adopt existing outputs", page)
        self.pmsSkipFailedCheck = QtWidgets.QCheckBox("Skip failed", page)

        form.addRow("Particle .cs", self._hrow(self.pmsParticleEdit, self.pmsParticleBrowse))
        form.addRow("Template .cs", self._hrow(self.pmsTemplateEdit, self.pmsTemplateBrowse))
        form.addRow("Control points .json", self._hrow(self.pmsControlPointsEdit, self.pmsControlPointsBrowse))
        form.addRow("points_step", self.pmsPointsStepEdit)
        form.addRow("physical_membrane_dist", self.pmsPhysDistEdit)
        form.addRow("output_dirname", self.pmsOutputDirnameEdit)
        form.addRow("batch_size", self.pmsBatchSizeEdit)
        form.addRow(self.pmsResumeCheck)
        form.addRow(self.pmsForceCheck)
        form.addRow(self.pmsAdoptCheck)
        form.addRow(self.pmsSkipFailedCheck)

        for w in [
            self.pmsParticleEdit,
            self.pmsTemplateEdit,
            self.pmsControlPointsEdit,
            self.pmsPointsStepEdit,
            self.pmsPhysDistEdit,
            self.pmsOutputDirnameEdit,
            self.pmsBatchSizeEdit,
        ]:
            w.editingFinished.connect(self._apply_editor_to_selected_job)
        for c in [self.pmsResumeCheck, self.pmsForceCheck, self.pmsAdoptCheck, self.pmsSkipFailedCheck]:
            c.stateChanged.connect(self._apply_editor_to_selected_job)

        return page

    def _build_page_micrograph_mms(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)

        self.mmsParticleStarEdit = QtWidgets.QLineEdit(page)
        self.mmsParticleStarBrowse = QtWidgets.QPushButton("Browse...", page)
        self.mmsParticleStarBrowse.clicked.connect(lambda: self._browse_file(self.mmsParticleStarEdit, "STAR Files (*.star)"))

        self.mmsDependsOnCombo = QtWidgets.QComboBox(page)
        self.mmsDependsOnCombo.setToolTip(
            "Select an upstream PMS job. MMS will wait for this dependency and "
            "inherit particle_output_root/output_dirname by default."
        )

        self.mmsParticleOutputRootEdit = QtWidgets.QLineEdit(page)
        self.mmsParticleOutputRootBrowse = QtWidgets.QPushButton("Browse...", page)
        self.mmsParticleOutputRootBrowse.clicked.connect(self._browse_mms_particle_output_root)
        self.mmsParticleOutputRootEdit.setToolTip(
            "Optional explicit PMS output root for dependency lookup. "
            "Leave empty to auto-inject from depends_on."
        )

        self.mmsOutputDirnameEdit = QtWidgets.QLineEdit(page)
        self.mmsOutputDirnameEdit.setPlaceholderText("subtracted")
        self.mmsOutputDirnameEdit.setToolTip(
            "Output folder name for MMS outputs/dependency lookup (default: subtracted)."
        )

        self.mmsBatchSizeEdit = QtWidgets.QLineEdit(page)
        self.mmsBatchSizeEdit.setPlaceholderText("30")
        self.mmsBatchSizeEdit.setToolTip("Minibatch size for micrograph subtraction.")

        self.mmsResumeCheck = QtWidgets.QCheckBox("Resume (.mxt)", page)
        self.mmsResumeCheck.setChecked(True)
        self.mmsForceCheck = QtWidgets.QCheckBox("Force recompute", page)
        self.mmsAdoptCheck = QtWidgets.QCheckBox("Adopt existing outputs", page)
        self.mmsSkipFailedCheck = QtWidgets.QCheckBox("Skip failed", page)
        self.mmsRequireParticleMxtCheck = QtWidgets.QCheckBox("Require particle-stack .mxt success", page)
        self.mmsRequireParticleMxtCheck.setChecked(True)
        self.mmsStrictDepsCheck = QtWidgets.QCheckBox("Strict dependency preflight (fail-fast)", page)
        self.mmsStrictDepsCheck.setChecked(True)
        self.mmsWriteOutputStarCheck = QtWidgets.QCheckBox("Write output STAR", page)
        self.mmsWriteOutputStarCheck.setChecked(True)

        self.mmsOutputStarPathEdit = QtWidgets.QLineEdit(page)
        self.mmsOutputStarPathEdit.setPlaceholderText("(auto)")
        self.mmsOutputStarPathBrowse = QtWidgets.QPushButton("Browse...", page)
        self.mmsOutputStarPathBrowse.clicked.connect(
            lambda: self._save_file(
                self.mmsOutputStarPathEdit,
                "STAR Files (*.star)",
                default_name="mms_micrograph_output.star",
            )
        )
        self.mmsOutputStarPathEdit.setToolTip("Optional explicit output STAR path; blank uses deterministic default.")

        self.mmsDependencyWarningLabel = QtWidgets.QLabel("", page)
        self.mmsDependencyWarningLabel.setWordWrap(True)
        self.mmsDependencyWarningLabel.setStyleSheet("color: #b26a00;")

        form.addRow("particles_selected.star", self._hrow(self.mmsParticleStarEdit, self.mmsParticleStarBrowse))
        form.addRow("depends_on (PMS)", self.mmsDependsOnCombo)
        form.addRow(
            "particle_output_root",
            self._hrow(self.mmsParticleOutputRootEdit, self.mmsParticleOutputRootBrowse),
        )
        form.addRow("output_dirname", self.mmsOutputDirnameEdit)
        form.addRow("batch_size", self.mmsBatchSizeEdit)
        form.addRow(self.mmsResumeCheck)
        form.addRow(self.mmsForceCheck)
        form.addRow(self.mmsAdoptCheck)
        form.addRow(self.mmsSkipFailedCheck)
        form.addRow(self.mmsRequireParticleMxtCheck)
        form.addRow(self.mmsStrictDepsCheck)
        form.addRow(self.mmsWriteOutputStarCheck)
        form.addRow("output_star_path", self._hrow(self.mmsOutputStarPathEdit, self.mmsOutputStarPathBrowse))
        form.addRow(self.mmsDependencyWarningLabel)

        for w in [
            self.mmsParticleStarEdit,
            self.mmsParticleOutputRootEdit,
            self.mmsOutputDirnameEdit,
            self.mmsBatchSizeEdit,
            self.mmsOutputStarPathEdit,
        ]:
            w.editingFinished.connect(self._apply_editor_to_selected_job)
        for c in [
            self.mmsResumeCheck,
            self.mmsForceCheck,
            self.mmsAdoptCheck,
            self.mmsSkipFailedCheck,
            self.mmsRequireParticleMxtCheck,
            self.mmsStrictDepsCheck,
            self.mmsWriteOutputStarCheck,
        ]:
            c.stateChanged.connect(self._apply_editor_to_selected_job)
        self.mmsDependsOnCombo.currentIndexChanged.connect(self._apply_editor_to_selected_job)

        return page

    def _build_page_mem_analyze(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(parent)
        form = QtWidgets.QFormLayout(page)

        self.anTemplateEdit = QtWidgets.QLineEdit(page)
        self.anTemplateBrowse = QtWidgets.QPushButton("Browse...", page)
        self.anTemplateBrowse.clicked.connect(lambda: self._browse_file(self.anTemplateEdit, "CS Files (*.cs)"))

        self.anParticleEdit = QtWidgets.QLineEdit(page)
        self.anParticleBrowse = QtWidgets.QPushButton("Browse...", page)
        self.anParticleBrowse.clicked.connect(lambda: self._browse_file(self.anParticleEdit, "CS Files (*.cs)"))

        self.anOutputEdit = QtWidgets.QLineEdit(page)
        self.anOutputBrowse = QtWidgets.QPushButton("Browse...", page)
        self.anOutputBrowse.clicked.connect(lambda: self._save_file(self.anOutputEdit, "JSON Files (*.json)", default_name="control_points.json"))

        def line(default: str, tooltip: str) -> QtWidgets.QLineEdit:
            w = QtWidgets.QLineEdit(page)
            w.setText(default)
            w.setToolTip(tooltip)
            w.editingFinished.connect(self._apply_editor_to_selected_job)
            return w

        self.anDegree = line("3", "Bezier curve degree.")
        self.anPhysicalDist = line("35", "Physical membrane distance in Å.")
        self.anNumPoints = line("600", "Coarsefit sample points.")
        self.anCoarseIter = line("300", "Coarsefit iterations.")
        self.anCoarseCpus = line("20", "Coarsefit CPU workers.")
        self.anCurPenalty = line("0.05", "Curvature penalty threshold.")
        self.anDitherRange = line("50", "Dithering range.")
        self.anRefineIter = line("700", "Refine iterations.")
        self.anRefineCpus = line("12", "Refine CPU workers.")

        form.addRow("Template .cs", self._hrow(self.anTemplateEdit, self.anTemplateBrowse))
        form.addRow("Particle .cs", self._hrow(self.anParticleEdit, self.anParticleBrowse))
        form.addRow("Output JSON", self._hrow(self.anOutputEdit, self.anOutputBrowse))
        form.addRow("degree", self.anDegree)
        form.addRow("physical_membrane_dist", self.anPhysicalDist)
        form.addRow("num_points", self.anNumPoints)
        form.addRow("coarsefit_iter", self.anCoarseIter)
        form.addRow("coarsefit_cpus", self.anCoarseCpus)
        form.addRow("cur_penalty_thr", self.anCurPenalty)
        form.addRow("dithering_range", self.anDitherRange)
        form.addRow("refine_iter", self.anRefineIter)
        form.addRow("refine_cpus", self.anRefineCpus)

        for w in [self.anTemplateEdit, self.anParticleEdit, self.anOutputEdit]:
            w.editingFinished.connect(self._apply_editor_to_selected_job)

        return page

    # ------------------------- Run controls + log -------------------------

    def _build_run_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self.runButton = QtWidgets.QPushButton("Run batch", w)
        self.stopButton = QtWidgets.QPushButton("Stop", w)
        self.openOutputButton = QtWidgets.QPushButton("Open selected output folder", w)
        self.commandButton = QtWidgets.QPushButton("Command…", w)

        self.runButton.setToolTip("Generate a batch spec and start the scheduler subprocess.")
        self.stopButton.setToolTip("Terminate the scheduler (and all running jobs).")
        self.openOutputButton.setToolTip("Open the output_root of the selected job in the file browser.")
        self.commandButton.setToolTip("Show the exact command(s) that will be executed.")

        self.runButton.clicked.connect(self._run_batch)
        self.stopButton.clicked.connect(self._stop_batch)
        self.openOutputButton.clicked.connect(self._open_selected_output_root)
        self.commandButton.clicked.connect(self._show_command_preview)

        layout.addWidget(self.runButton)
        layout.addWidget(self.stopButton)
        layout.addWidget(self.openOutputButton)
        layout.addWidget(self.commandButton)
        layout.addStretch(1)
        return w

    def _build_log_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Batch log (scheduler stdout/stderr)", self)
        layout = QtWidgets.QVBoxLayout(box)
        self.logText = QtWidgets.QTextBrowser(box)
        self.logText.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.logText)
        return box

    # ------------------------- Helpers -------------------------

    def _hrow(self, left: QtWidgets.QWidget, right: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(left, stretch=1)
        layout.addWidget(right)
        return w

    def _browse_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", filter_str)
        if path:
            line_edit.setText(path)
            self._apply_editor_to_selected_job()

    def _save_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str, *, default_name: str) -> None:
        base_dir = self._run_root or os.getcwd()
        default_path = str(Path(base_dir) / default_name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output path", default_path, filter_str)
        if path:
            line_edit.setText(path)
            self._apply_editor_to_selected_job()

    def _browse_run_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch run root")
        if path:
            self.runRootLineEdit.setText(path)
            self._on_run_root_changed()

    def _on_run_root_changed(self) -> None:
        raw = self.runRootLineEdit.text().strip()
        if raw:
            try:
                resolved = str(Path(raw).expanduser().resolve())
            except Exception:
                resolved = raw
            self._run_root = resolved
            # Normalize the displayed path to avoid relative-path surprises.
            self.runRootLineEdit.setText(self._run_root)
        else:
            self._run_root = ""
        self._recompute_default_output_roots()

    def _recompute_default_output_roots(self) -> None:
        run_root = self._run_root
        if not run_root:
            return
        for job in self._jobs:
            if not job.custom_output_root:
                job.output_root = str(Path(run_root) / job.job_id)
        self._refresh_table()
        self._load_selected_into_editor()

    def _browse_job_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select job output root")
        if path:
            self.outputRootEdit.setText(path)
            self.customOutCheck.setChecked(True)
            self._apply_editor_to_selected_job()

    def _browse_mms_particle_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select particle_output_root")
        if path:
            self.mmsParticleOutputRootEdit.setText(path)
            self._apply_editor_to_selected_job()

    def _browse_input_base_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input base directory")
        if path:
            self.customInputBaseDirCheck.setChecked(True)
            self.inputBaseDirEdit.setText(path)
            self._apply_editor_to_selected_job()

    def _on_custom_output_root_changed(self) -> None:
        self._apply_editor_to_selected_job()

    def _on_custom_input_base_dir_changed(self) -> None:
        """
        Toggle between auto-inferred and user-provided `input_base_dir`.

        When enabling custom mode, prefill the field with the current auto
        inference to make overrides easy and non-empty by default.
        """
        if self._selected_row is None:
            return
        if not (0 <= self._selected_row < len(self._jobs)):
            return

        job = self._jobs[self._selected_row]
        if bool(self.customInputBaseDirCheck.isChecked()) and not self.inputBaseDirEdit.text().strip():
            inferred = self._auto_infer_input_base_dir(job)
            if inferred:
                self.inputBaseDirEdit.setText(inferred)

        self._apply_editor_to_selected_job()

    def _primary_input_path_for_job(self, job: _GuiJob) -> str | None:
        if job.kind == "bezierfit_particle_pms":
            p = job.args.get("particle")
            return None if p in (None, "") else str(p)
        if job.kind == "bezierfit_micrograph_mms":
            p = job.args.get("particle")
            return None if p in (None, "") else str(p)
        if job.kind == "bezierfit_mem_analyze":
            t = job.args.get("template")
            if t not in (None, ""):
                return str(t)
            p = job.args.get("particle")
            return None if p in (None, "") else str(p)
        return None

    def _normalise_path_for_inference(self, path_str: str) -> str:
        """
        Convert `path_str` into a deterministic absolute path for base-dir inference.

        - If absolute: resolve directly.
        - If relative: resolve relative to run_root (if set), otherwise current cwd.
        """
        s = str(path_str).strip()
        if s == "":
            return ""

        expanded = os.path.expanduser(os.path.expandvars(s))
        p = Path(expanded)
        if p.is_absolute():
            return str(p.resolve())

        base = Path(self._run_root) if self._run_root else Path(os.getcwd())
        return str((base / p).resolve())

    def _auto_infer_input_base_dir(self, job: _GuiJob) -> str:
        primary = self._primary_input_path_for_job(job)
        if primary is None:
            return ""
        try:
            abs_primary = self._normalise_path_for_inference(primary)
            if abs_primary == "":
                return ""
            return infer_input_base_dir(abs_primary)
        except Exception:
            return ""

    def _resolved_input_base_dir(self, job: _GuiJob) -> str:
        """
        Determine the `input_base_dir` that should be written to the spec.
        """
        if bool(job.custom_input_base_dir) and str(job.input_base_dir).strip():
            return str(job.input_base_dir).strip()
        return self._auto_infer_input_base_dir(job)

    def _pms_job_ids(self) -> list[str]:
        return [j.job_id for j in self._jobs if j.kind == "bezierfit_particle_pms"]

    def _default_mms_dependency(self) -> str | None:
        pms_ids = self._pms_job_ids()
        if len(pms_ids) == 1:
            return pms_ids[0]
        return None

    def _refresh_mms_depends_combo(self, *, selected_dep: str | None = None) -> None:
        self.mmsDependsOnCombo.blockSignals(True)
        self.mmsDependsOnCombo.clear()
        self.mmsDependsOnCombo.addItem("(none)", None)
        for pms_id in self._pms_job_ids():
            self.mmsDependsOnCombo.addItem(pms_id, pms_id)
        target = selected_dep
        if target is None:
            target = self._default_mms_dependency()
        idx = self.mmsDependsOnCombo.findData(target)
        if idx < 0:
            idx = 0
        self.mmsDependsOnCombo.setCurrentIndex(idx)
        self.mmsDependsOnCombo.blockSignals(False)

    def _revalidate_mms_dependencies(self) -> list[str]:
        """
        Validate GUI-level MMS dependency selections and expose warnings.
        """
        errors: list[str] = []
        by_id = {j.job_id: j for j in self._jobs}
        for job in self._jobs:
            if job.kind != "bezierfit_micrograph_mms":
                if job.depends_on:
                    job.depends_on = []
                continue
            deps = [str(x) for x in job.depends_on]
            if len(deps) > 1:
                errors.append(
                    f"MMS job {job.job_id!r} has multiple dependencies {deps}. "
                    "Select at most one PMS dependency in the GUI."
                )
            for dep_id in deps:
                dep_job = by_id.get(dep_id)
                if dep_job is None:
                    errors.append(f"MMS job {job.job_id!r} depends on missing job_id {dep_id!r}.")
                elif dep_job.kind != "bezierfit_particle_pms":
                    errors.append(
                        f"MMS job {job.job_id!r} depends on {dep_id!r} (kind={dep_job.kind!r}); "
                        "GUI only allows PMS dependencies."
                    )

        self._dependency_errors = errors
        if self._selected_row is not None and 0 <= self._selected_row < len(self._jobs):
            selected = self._jobs[self._selected_row]
            if selected.kind == "bezierfit_micrograph_mms":
                selected_errs = [e for e in errors if f"{selected.job_id!r}" in e]
                if selected_errs:
                    self.mmsDependencyWarningLabel.setText("Dependency warning: " + selected_errs[0])
                else:
                    self.mmsDependencyWarningLabel.setText("")
            else:
                self.mmsDependencyWarningLabel.setText("")
        else:
            self.mmsDependencyWarningLabel.setText("")
        return errors

    def _sync_input_base_dir_widgets(self, job: _GuiJob, *, set_checkbox: bool = True) -> None:
        """
        Update the input_base_dir UI controls for the given job.
        """
        custom = bool(job.custom_input_base_dir)
        if set_checkbox:
            self.customInputBaseDirCheck.setChecked(custom)

        if custom:
            self.inputBaseDirEdit.setEnabled(True)
            self.inputBaseDirBrowse.setEnabled(True)
            self.inputBaseDirEdit.setText(str(job.input_base_dir or ""))
        else:
            inferred = self._auto_infer_input_base_dir(job)
            self.inputBaseDirEdit.setEnabled(False)
            self.inputBaseDirBrowse.setEnabled(False)
            self.inputBaseDirEdit.setText(inferred)

    def _auto_detect_gpus(self) -> None:
        ok, details = check_cupy_cuda_available()
        if not ok:
            QtWidgets.QMessageBox.critical(self, "GPU detection failed", details)
            return
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        self.gpusLineEdit.setText(",".join(str(i) for i in range(count)))
        self.maxRunningSpin.setValue(max(1, count))

    # ------------------------- Job list operations -------------------------

    def _add_default_job(self) -> None:
        self._jobs.append(
            _GuiJob(
                job_id="job_001",
                kind="bezierfit_particle_pms",
                gpus=1,
                procs=None,
                output_root=str(Path(os.getcwd()) / "mxt_runs" / "job_001"),
                args={
                    "particle": "",
                    "template": "",
                    "control_points": "",
                    "points_step": 0.001,
                    "physical_membrane_dist": 35,
                    "output_dirname": "subtracted",
                    "batch_size": 20,
                    "resume": True,
                    "force": False,
                    "adopt_existing_outputs": False,
                    "skip_failed": False,
                },
                depends_on=[],
            )
        )
        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=0)

    def _next_job_id(self) -> str:
        existing = {j.job_id for j in self._jobs}
        i = 1
        while True:
            cand = f"job_{i:03d}"
            if cand not in existing:
                return cand
            i += 1

    def _refresh_table(self, *, select_row: int | None = None) -> None:
        self._updating_table = True
        self.jobsTable.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            enabled_item = QtWidgets.QTableWidgetItem()
            enabled_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(QtCore.Qt.Checked if job.enabled else QtCore.Qt.Unchecked)
            enabled_item.setToolTip("Enable/disable this job (disabled jobs are not included in the batch spec).")
            self.jobsTable.setItem(row, 0, enabled_item)

            def _item(text: str) -> QtWidgets.QTableWidgetItem:
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                return it

            self.jobsTable.setItem(row, 1, _item(job.job_id))
            self.jobsTable.setItem(row, 2, _item(_KIND_TO_LABEL.get(job.kind, job.kind)))
            self.jobsTable.setItem(row, 3, _item(str(int(job.gpus))))
            self.jobsTable.setItem(row, 4, _item("" if job.procs is None else str(int(job.procs))))
            self.jobsTable.setItem(row, 5, _item(job.output_root))
            self.jobsTable.setItem(row, 6, _item(job.status))
            self.jobsTable.setItem(row, 7, _item(",".join(str(x) for x in job.assigned_gpus)))

        self.jobsTable.resizeColumnsToContents()
        if select_row is not None and 0 <= select_row < len(self._jobs):
            self.jobsTable.selectRow(select_row)
        self._updating_table = False

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_table:
            return
        row = int(item.row())
        col = int(item.column())
        if col != 0:
            return
        if not (0 <= row < len(self._jobs)):
            return
        self._jobs[row].enabled = item.checkState() == QtCore.Qt.Checked

    def _on_table_selection_changed(self) -> None:
        rows = self.jobsTable.selectionModel().selectedRows()
        if not rows:
            self._selected_row = None
            self.mmsDependencyWarningLabel.setText("")
            return
        self._selected_row = int(rows[0].row())
        self._load_selected_into_editor()

    def _load_selected_into_editor(self) -> None:
        if self._selected_row is None:
            return
        if not (0 <= self._selected_row < len(self._jobs)):
            return
        job = self._jobs[self._selected_row]

        self.editorTitle.setText(f"Job editor — {job.job_id}")
        self.jobIdEdit.setText(job.job_id)
        self.kindCombo.setCurrentText(_KIND_TO_LABEL.get(job.kind, job.kind))
        self.gpusSpin.setValue(int(job.gpus))
        self.procsEdit.setText("" if job.procs is None else str(int(job.procs)))
        self.customOutCheck.setChecked(bool(job.custom_output_root))
        self.outputRootEdit.setText(job.output_root)
        self.outputRootEdit.setEnabled(bool(job.custom_output_root))
        self.outputRootBrowse.setEnabled(bool(job.custom_output_root))

        # Pages
        if job.kind == "bezierfit_mem_analyze":
            self.kindStack.setCurrentIndex(0)
            self.anTemplateEdit.setText(str(job.args.get("template", "")))
            self.anParticleEdit.setText(str(job.args.get("particle", "")))
            self.anOutputEdit.setText(str(job.args.get("output", "")))
            self.anDegree.setText(str(job.args.get("degree", "3")))
            self.anPhysicalDist.setText(str(job.args.get("physical_membrane_dist", "35")))
            self.anNumPoints.setText(str(job.args.get("num_points", "600")))
            self.anCoarseIter.setText(str(job.args.get("coarsefit_iter", "300")))
            self.anCoarseCpus.setText(str(job.args.get("coarsefit_cpus", "20")))
            self.anCurPenalty.setText(str(job.args.get("cur_penalty_thr", "0.05")))
            self.anDitherRange.setText(str(job.args.get("dithering_range", "50")))
            self.anRefineIter.setText(str(job.args.get("refine_iter", "700")))
            self.anRefineCpus.setText(str(job.args.get("refine_cpus", "12")))
        elif job.kind == "bezierfit_particle_pms":
            self.kindStack.setCurrentIndex(1)
            self.pmsParticleEdit.setText(str(job.args.get("particle", "")))
            self.pmsTemplateEdit.setText(str(job.args.get("template", "")))
            self.pmsControlPointsEdit.setText(str(job.args.get("control_points", "")))
            self.pmsPointsStepEdit.setText(str(job.args.get("points_step", "0.001")))
            self.pmsPhysDistEdit.setText(str(job.args.get("physical_membrane_dist", "35")))
            self.pmsOutputDirnameEdit.setText(str(job.args.get("output_dirname", "subtracted")))
            self.pmsBatchSizeEdit.setText(str(job.args.get("batch_size", "20")))
            self.pmsResumeCheck.setChecked(bool(job.args.get("resume", True)))
            self.pmsForceCheck.setChecked(bool(job.args.get("force", False)))
            self.pmsAdoptCheck.setChecked(bool(job.args.get("adopt_existing_outputs", False)))
            self.pmsSkipFailedCheck.setChecked(bool(job.args.get("skip_failed", False)))
        else:
            self.kindStack.setCurrentIndex(2)
            self.mmsParticleStarEdit.setText(str(job.args.get("particle", "")))
            depends_value = job.depends_on[0] if job.depends_on else self._default_mms_dependency()
            if not job.depends_on and depends_value is not None:
                job.depends_on = [depends_value]
            self._refresh_mms_depends_combo(selected_dep=depends_value)
            self.mmsParticleOutputRootEdit.setText(str(job.args.get("particle_output_root", "")))
            self.mmsOutputDirnameEdit.setText(str(job.args.get("output_dirname", "subtracted")))
            self.mmsBatchSizeEdit.setText(str(job.args.get("batch_size", "30")))
            self.mmsResumeCheck.setChecked(bool(job.args.get("resume", True)))
            self.mmsForceCheck.setChecked(bool(job.args.get("force", False)))
            self.mmsAdoptCheck.setChecked(bool(job.args.get("adopt_existing_outputs", False)))
            self.mmsSkipFailedCheck.setChecked(bool(job.args.get("skip_failed", False)))
            self.mmsRequireParticleMxtCheck.setChecked(bool(job.args.get("require_particle_mxt", True)))
            self.mmsStrictDepsCheck.setChecked(bool(job.args.get("strict_dependencies", True)))
            self.mmsWriteOutputStarCheck.setChecked(bool(job.args.get("write_output_star", True)))
            self.mmsOutputStarPathEdit.setText(str(job.args.get("output_star_path", "")))

        self._sync_input_base_dir_widgets(job, set_checkbox=True)
        self._revalidate_mms_dependencies()

    def _apply_editor_to_selected_job(self) -> None:
        if self._selected_row is None:
            return
        if not (0 <= self._selected_row < len(self._jobs)):
            return
        job = self._jobs[self._selected_row]
        prev_job_id = str(job.job_id)
        prev_kind = str(job.kind)

        job.job_id = self.jobIdEdit.text().strip() or job.job_id
        job.kind = _LABEL_TO_KIND.get(self.kindCombo.currentText(), job.kind)
        job.gpus = int(self.gpusSpin.value())
        procs = self.procsEdit.text().strip()
        job.procs = None if procs == "" else int(procs)
        job.custom_output_root = bool(self.customOutCheck.isChecked())

        if job.custom_output_root:
            job.output_root = self.outputRootEdit.text().strip() or job.output_root
        else:
            if self._run_root:
                job.output_root = str(Path(self._run_root) / job.job_id)
            self.outputRootEdit.setText(job.output_root)

        job.custom_input_base_dir = bool(self.customInputBaseDirCheck.isChecked())
        if job.custom_input_base_dir:
            raw_base = self.inputBaseDirEdit.text().strip()
            if raw_base:
                try:
                    job.input_base_dir = normalise_dir(raw_base)
                except Exception:
                    # Keep the raw text; spec export will fall back to auto if empty.
                    job.input_base_dir = raw_base
            else:
                job.input_base_dir = ""
        else:
            job.input_base_dir = ""

        if job.kind == "bezierfit_micrograph_mms":
            current_dep = job.depends_on[0] if job.depends_on else None
            if prev_kind != job.kind or prev_job_id != job.job_id:
                self._refresh_mms_depends_combo(selected_dep=current_dep)
            selected_dep = self.mmsDependsOnCombo.currentData()
            if isinstance(selected_dep, str) and selected_dep.strip() != "":
                job.depends_on = [selected_dep]
            else:
                auto_dep = self._default_mms_dependency()
                if auto_dep is not None:
                    job.depends_on = [auto_dep]
                    self._refresh_mms_depends_combo(selected_dep=auto_dep)
                else:
                    job.depends_on = []
        else:
            job.depends_on = []

        # Kind-specific args
        if job.kind == "bezierfit_mem_analyze":
            job.args = {
                "template": self.anTemplateEdit.text().strip(),
                "particle": self.anParticleEdit.text().strip(),
                "output": self.anOutputEdit.text().strip(),
                "degree": _safe_int(self.anDegree.text(), default=3),
                "physical_membrane_dist": _safe_int(self.anPhysicalDist.text(), default=35),
                "num_points": _safe_int(self.anNumPoints.text(), default=600),
                "coarsefit_iter": _safe_int(self.anCoarseIter.text(), default=300),
                "coarsefit_cpus": _safe_int(self.anCoarseCpus.text(), default=20),
                "cur_penalty_thr": _safe_float(self.anCurPenalty.text(), default=0.05),
                "dithering_range": _safe_int(self.anDitherRange.text(), default=50),
                "refine_iter": _safe_int(self.anRefineIter.text(), default=700),
                "refine_cpus": _safe_int(self.anRefineCpus.text(), default=12),
            }
        elif job.kind == "bezierfit_particle_pms":
            job.args = {
                "particle": self.pmsParticleEdit.text().strip(),
                "template": self.pmsTemplateEdit.text().strip(),
                "control_points": self.pmsControlPointsEdit.text().strip(),
                "points_step": _safe_float(self.pmsPointsStepEdit.text(), default=0.001),
                "physical_membrane_dist": _safe_int(self.pmsPhysDistEdit.text(), default=35),
                "output_dirname": self.pmsOutputDirnameEdit.text().strip() or "subtracted",
                "batch_size": _safe_int(self.pmsBatchSizeEdit.text(), default=20),
                "resume": bool(self.pmsResumeCheck.isChecked()),
                "force": bool(self.pmsForceCheck.isChecked()),
                "adopt_existing_outputs": bool(self.pmsAdoptCheck.isChecked()),
                "skip_failed": bool(self.pmsSkipFailedCheck.isChecked()),
            }
        else:
            particle_output_root = self.mmsParticleOutputRootEdit.text().strip()
            output_star_path = self.mmsOutputStarPathEdit.text().strip()
            job.args = {
                "particle": self.mmsParticleStarEdit.text().strip(),
                "particle_output_root": particle_output_root if particle_output_root != "" else None,
                "output_dirname": self.mmsOutputDirnameEdit.text().strip() or "subtracted",
                "batch_size": _safe_int(self.mmsBatchSizeEdit.text(), default=30),
                "resume": bool(self.mmsResumeCheck.isChecked()),
                "force": bool(self.mmsForceCheck.isChecked()),
                "adopt_existing_outputs": bool(self.mmsAdoptCheck.isChecked()),
                "skip_failed": bool(self.mmsSkipFailedCheck.isChecked()),
                "require_particle_mxt": bool(self.mmsRequireParticleMxtCheck.isChecked()),
                "strict_dependencies": bool(self.mmsStrictDepsCheck.isChecked()),
                "write_output_star": bool(self.mmsWriteOutputStarCheck.isChecked()),
                "output_star_path": output_star_path if output_star_path != "" else None,
            }

        self.outputRootEdit.setEnabled(bool(job.custom_output_root))
        self.outputRootBrowse.setEnabled(bool(job.custom_output_root))
        if job.custom_input_base_dir and str(job.input_base_dir).strip():
            self.inputBaseDirEdit.setText(str(job.input_base_dir))
        self._sync_input_base_dir_widgets(job, set_checkbox=False)

        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=self._selected_row)

    def _on_kind_changed(self) -> None:
        kind = _LABEL_TO_KIND.get(self.kindCombo.currentText(), "bezierfit_particle_pms")
        if kind == "bezierfit_mem_analyze":
            self.kindStack.setCurrentIndex(0)
        elif kind == "bezierfit_particle_pms":
            self.kindStack.setCurrentIndex(1)
        else:
            self.kindStack.setCurrentIndex(2)
        self._apply_editor_to_selected_job()

    def _on_add_job(self) -> None:
        jid = self._next_job_id()
        out_root = str(Path(self._run_root) / jid) if self._run_root else str(Path(os.getcwd()) / "mxt_runs" / jid)
        self._jobs.append(
            _GuiJob(
                enabled=True,
                job_id=jid,
                kind="bezierfit_particle_pms",
                gpus=1,
                procs=None,
                output_root=out_root,
                args={
                    "particle": "",
                    "template": "",
                    "control_points": "",
                    "points_step": 0.001,
                    "physical_membrane_dist": 35,
                    "output_dirname": "subtracted",
                    "batch_size": 20,
                    "resume": True,
                    "force": False,
                    "adopt_existing_outputs": False,
                    "skip_failed": False,
                },
                depends_on=[],
            )
        )
        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=len(self._jobs) - 1)

    def _on_duplicate_job(self) -> None:
        if self._selected_row is None:
            return
        base = self._jobs[self._selected_row]
        suffix = 1
        existing = {j.job_id for j in self._jobs}
        while True:
            cand = f"{base.job_id}_copy{suffix}"
            if cand not in existing:
                break
            suffix += 1
        out_root = str(Path(self._run_root) / cand) if self._run_root else str(Path(base.output_root).parent / cand)
        clone = _GuiJob(
            enabled=base.enabled,
            job_id=cand,
            kind=base.kind,
            gpus=base.gpus,
            procs=base.procs,
            output_root=out_root,
            custom_output_root=False,
            input_base_dir=str(base.input_base_dir),
            custom_input_base_dir=bool(base.custom_input_base_dir),
            args=dict(base.args),
            depends_on=list(base.depends_on),
        )
        self._jobs.insert(self._selected_row + 1, clone)
        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=self._selected_row + 1)

    def _on_remove_job(self) -> None:
        if self._selected_row is None:
            return
        self._jobs.pop(self._selected_row)
        new_row = min(self._selected_row, len(self._jobs) - 1)
        self._selected_row = None
        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=new_row if new_row >= 0 else None)

    def _on_sweep_job(self) -> None:
        if self._selected_row is None:
            return
        base = self._jobs[self._selected_row]
        candidates = sorted([k for k in base.args.keys() if k not in {"particle", "template", "control_points", "output"}])
        if not candidates:
            QtWidgets.QMessageBox.warning(self, "Sweep unavailable", "No sweepable parameters found for this job.")
            return
        dlg = SweepBuilderDialog(param_candidates=candidates, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        spec = dlg.sweep_spec()

        inserted: list[_GuiJob] = []
        for v in spec.values:
            suffix = f"_{spec.param}{_sanitise_suffix(v)}"
            jid = f"{base.job_id}{suffix}"
            # Ensure uniqueness
            existing = {j.job_id for j in self._jobs} | {j.job_id for j in inserted}
            if jid in existing:
                jid = f"{jid}_{len(inserted)+1}"
            out_root = str(Path(self._run_root) / jid) if self._run_root else str(Path(base.output_root).parent / jid)
            new_args = dict(base.args)
            new_args[spec.param] = v
            inserted.append(
                _GuiJob(
                    enabled=True,
                    job_id=jid,
                    kind=base.kind,
                    gpus=base.gpus,
                    procs=base.procs,
                    output_root=out_root,
                    custom_output_root=False,
                    input_base_dir=str(base.input_base_dir),
                    custom_input_base_dir=bool(base.custom_input_base_dir),
                    args=new_args,
                    depends_on=list(base.depends_on),
                )
            )

        self._jobs[self._selected_row + 1 : self._selected_row + 1] = inserted
        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=self._selected_row + 1)

    # ------------------------- Import/export -------------------------

    def _on_import_spec(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open batch spec", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            spec = load_spec_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Import failed", f"Failed to load spec: {type(exc).__name__}: {exc}")
            return

        self._jobs = []
        for j in spec.jobs:
            imported_args = dict(j.args)
            imported_base = str(imported_args.pop("input_base_dir", "") or "")
            imported_base = imported_base.strip()
            self._jobs.append(
                _GuiJob(
                    enabled=bool(j.enabled),
                    job_id=j.job_id,
                    kind=j.kind,
                    gpus=int(j.resources.gpus),
                    procs=j.resources.procs,
                    output_root=str(j.output_root),
                    custom_output_root=True,
                    input_base_dir=imported_base,
                    custom_input_base_dir=bool(imported_base),
                    args=imported_args,
                    depends_on=list(j.depends_on),
                    status="queued",
                )
            )

        self.gpusLineEdit.setText(",".join(str(x) for x in spec.scheduler.gpus))
        self.maxRunningSpin.setValue(int(spec.scheduler.max_running_jobs))
        self.policyCombo.setCurrentText("fill_first (Recommended)" if spec.scheduler.policy == "fill_first" else "round_robin")

        # Suggest run_root based on spec location if all output_roots share a common parent.
        try:
            parents = {str(Path(j.output_root).parent) for j in self._jobs}
            if len(parents) == 1:
                self.runRootLineEdit.setText(list(parents)[0])
                self._on_run_root_changed()
        except Exception:
            pass

        self._revalidate_mms_dependencies()
        self._refresh_table(select_row=0 if self._jobs else None)

    def _on_export_spec(self) -> None:
        try:
            spec_dict = self._generate_spec_dict()
            # Validate spec to fail fast on missing fields.
            parse_spec_dict(spec_dict, base_dir=Path(self._run_root or os.getcwd()))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Invalid spec: {type(exc).__name__}: {exc}")
            return

        base_dir = self._run_root or os.getcwd()
        default_path = str(Path(base_dir) / "bezierfit_batch_spec.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save batch spec", default_path, "JSON Files (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(spec_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Failed to write spec: {type(exc).__name__}: {exc}")

    # ------------------------- Scheduler run/stop -------------------------

    def _generate_spec_dict(self) -> dict[str, Any]:
        dep_errors = self._revalidate_mms_dependencies()
        if dep_errors:
            preview = "\n".join(dep_errors[:8])
            if len(dep_errors) > 8:
                preview += f"\n... and {len(dep_errors) - 8} more"
            raise ValueError(
                "Invalid MMS dependency selections in GUI. Please fix before export/run.\n"
                f"{preview}"
            )

        gpus = parse_gpu_list(self.gpusLineEdit.text().strip())
        policy = "fill_first" if self.policyCombo.currentText().startswith("fill_first") else "round_robin"
        max_running = int(self.maxRunningSpin.value())

        jobs = []
        for row, job in enumerate(self._jobs):
            # Sync enabled from table checkbox (authoritative UI control).
            enabled_item = self.jobsTable.item(row, 0)
            if enabled_item is not None:
                job.enabled = enabled_item.checkState() == QtCore.Qt.Checked

            if not job.enabled:
                continue

            args = dict(job.args)
            resolved_base = self._resolved_input_base_dir(job)
            if resolved_base:
                args["input_base_dir"] = resolved_base

            jobs.append(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "enabled": True,
                    "output_root": job.output_root,
                    "depends_on": list(job.depends_on),
                    "resources": {"gpus": int(job.gpus), "procs": job.procs},
                    "args": args,
                }
            )

        return {
            "spec_schema_version": 2,
            "scheduler": {
                "gpus": list(gpus),
                "policy": policy,
                "max_running_jobs": max_running,
                "fail_fast": True,
            },
            "jobs": jobs,
        }

    def _run_batch(self) -> None:
        if self._batch_popen and self._batch_popen.poll() is None:
            QtWidgets.QMessageBox.warning(self, "Batch running", "A batch is already running.")
            return

        run_root = self.runRootLineEdit.text().strip()
        if not run_root:
            QtWidgets.QMessageBox.warning(self, "Missing run root", "Please choose a run root directory.")
            return
        try:
            run_root = str(Path(run_root).expanduser().resolve())
        except Exception:
            pass
        Path(run_root).mkdir(parents=True, exist_ok=True)

        try:
            spec_dict = self._generate_spec_dict()
            parse_spec_dict(spec_dict, base_dir=Path(run_root))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid batch", f"{type(exc).__name__}: {exc}")
            return

        self._run_root = run_root
        self._spec_path = str(Path(run_root) / "bezierfit_batch_spec.json")
        self._state_path = str(Path(run_root) / "scheduler_state.json")
        self._log_path = str(Path(run_root) / "bezierfit_batch.run.out")
        self._pid_path = str(Path(run_root) / "bezierfit_batch.pid")

        # Write spec file (reproducibility).
        Path(self._spec_path).write_text(json.dumps(spec_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        cmd = [
            python_executable_for_subprocess(),
            "-u",
            "-m",
            "memxterminator.bezierfit.scheduler.cli",
            "--spec",
            self._spec_path,
            "--state",
            self._state_path,
        ]

        # Reset log tail state.
        self._log_last_pos = 0
        with open(self._log_path, "w", encoding="utf-8") as f:
            f.write(f">>> Command: {shlex.join(cmd)}\n")
            f.flush()
            self._batch_popen = subprocess.Popen(
                cmd,
                cwd=str(run_root),
                stdout=f,
                stderr=subprocess.STDOUT,
                **popen_kwargs_for_new_session(),
            )

        Path(self._pid_path).write_text(str(int(self._batch_popen.pid)) + "\n", encoding="utf-8")
        self.logText.append(f">>> Started batch scheduler pid={self._batch_popen.pid}")

    def _stop_batch(self) -> None:
        pid = None
        if self._batch_popen and self._batch_popen.poll() is None:
            pid = int(self._batch_popen.pid)
        elif self._pid_path and os.path.exists(self._pid_path):
            try:
                pid = int(Path(self._pid_path).read_text(encoding="utf-8").strip())
            except Exception:
                pid = None

        if pid is None:
            self.logText.append(">>> No running batch found.")
            return

        terminate_pid(int(pid))
        self.logText.append(f">>> Sent SIGTERM to batch pid={pid}")

    def _open_selected_output_root(self) -> None:
        if self._selected_row is None:
            return
        if not (0 <= self._selected_row < len(self._jobs)):
            return
        _open_folder(self._jobs[self._selected_row].output_root)

    def _show_command_preview(self) -> None:
        if self._selected_row is not None and 0 <= self._selected_row < len(self._jobs):
            job = self._jobs[self._selected_row]
            # Preview the job command as the scheduler would run it.
            args = dict(job.args)
            if job.kind in {"bezierfit_particle_pms", "bezierfit_micrograph_mms"} and "procs" not in args:
                args["procs"] = job.procs if job.procs is not None else int(job.gpus)
            resolved_base = self._resolved_input_base_dir(job)
            if resolved_base:
                args["input_base_dir"] = resolved_base
            bjob = BezierfitJob(
                job_id=job.job_id,
                kind=job.kind,  # type: ignore[arg-type]
                args=args,
                output_root=str(job.output_root),
                resources=JobResources(gpus=int(job.gpus), procs=job.procs),
                enabled=job.enabled,
                depends_on=tuple(job.depends_on),
            )
            argv = build_job_argv(bjob)
            CommandPreviewDialog(argv, self, title=f"Job Command: {job.job_id}").exec_()
            return

        # Otherwise show the batch command (if configured).
        if not self._spec_path:
            QtWidgets.QMessageBox.information(self, "Command", "Run root/spec not configured yet.")
            return
        cmd = [
            python_executable_for_subprocess(),
            "-u",
            "-m",
            "memxterminator.bezierfit.scheduler.cli",
            "--spec",
            self._spec_path,
            "--state",
            self._state_path,
        ]
        CommandPreviewDialog(cmd, self, title="Batch Command").exec_()

    # ------------------------- Timers: log + state -------------------------

    def _update_log_view(self) -> None:
        if not self._log_path:
            return
        if not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                f.seek(self._log_last_pos)
                new = f.read()
                self._log_last_pos = f.tell()
            if new:
                self.logText.append(new)
        except Exception:
            return

    def _update_state_view(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return

        jobs = state.get("jobs")
        if not isinstance(jobs, list):
            return
        by_id = {j.get("job_id"): j for j in jobs if isinstance(j, dict) and j.get("job_id")}

        for gui_job in self._jobs:
            rec = by_id.get(gui_job.job_id)
            if not rec:
                continue
            gui_job.status = str(rec.get("status", gui_job.status))
            assigned = rec.get("assigned_gpus", [])
            if isinstance(assigned, list):
                try:
                    gui_job.assigned_gpus = [int(x) for x in assigned]
                except Exception:
                    pass
            gui_job.pid = rec.get("pid") if isinstance(rec.get("pid"), int) else gui_job.pid
            gui_job.returncode = rec.get("returncode") if isinstance(rec.get("returncode"), int) else gui_job.returncode

        self._refresh_table(select_row=self._selected_row)
