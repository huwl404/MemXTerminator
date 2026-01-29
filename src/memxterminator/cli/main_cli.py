import argparse
import sys
import mrcfile

def _run_gui() -> None:
    from PyQt5 import QtWidgets
    from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

    from ..GUI.mainwindow_gui import Ui_MainWindow

    class MainWindow(QMainWindow, Ui_MainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setupUi(self)
            self.radon_button.clicked.connect(self.open_radon_analysis)
            self.mem_analyze_button.clicked.connect(self.open_membrane_analyzer)
            self.mem_subtraction_button.clicked.connect(self.open_membrane_subtraction)
            self.micrograph_membrane_subtraction_pushButton.clicked.connect(
                self.open_radon_micrograph_membrane_subtraction
            )
            self.mem_analyze_bezier_button.clicked.connect(self.open_membrane_analyzer_bezier)
            self.mem_subtraction_bezier_button.clicked.connect(
                self.open_membrane_subtraction_bezier
            )
            self.micrograph_membrane_subtraction_bezier_pushButton.clicked.connect(
                self.open_micrograph_membrane_subtraction_bezier
            )

            # Bezierfit: Batch scheduler (job-level parallelism)
            self.bezier_batch_scheduler_button = QtWidgets.QPushButton(
                "Batch Scheduler", self.verticalLayoutWidget_2
            )
            self.bezier_batch_scheduler_button.setToolTip(
                "Run multiple Bezierfit jobs in parallel (parameter sweeps and/or multiple datasets).\n\n"
                "Tip: choose a new empty 'run root' folder per batch run."
            )
            self.verticalLayout_2.addWidget(self.bezier_batch_scheduler_button)
            self.bezier_batch_scheduler_button.clicked.connect(self.open_bezier_batch_scheduler)

        def _show_feature_import_error(self, feature_name: str, exc: Exception) -> None:
            QMessageBox.critical(
                self,
                "Feature unavailable",
                (
                    f"Failed to load '{feature_name}'.\n\n"
                    "This usually means an optional dependency (e.g. GPU/CUDA libraries) "
                    "is missing or not usable on this system.\n\n"
                    f"Details: {type(exc).__name__}: {exc}"
                ),
            )

        def open_radon_analysis(self):
            try:
                from .radonfit_cli import RadonApp
            except Exception as exc:
                self._show_feature_import_error("Radon analysis (Radonfit)", exc)
                return
            self.radon_dialog = RadonApp(self)
            self.radon_dialog.show()

        def open_membrane_analyzer(self):
            reply = QMessageBox.question(
                self,
                "Membrane Analyzer - Radonfit - MemXTerminator",
                "Ensure you have completed Radon Analysis Blinking. \nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                try:
                    from .radonfit_cli import MembraneAnalyzerApp
                except Exception as exc:
                    self._show_feature_import_error("Membrane Analyzer (Radonfit)", exc)
                    return
                self.membrane_analyzer_dialog = MembraneAnalyzerApp(self)
                self.membrane_analyzer_dialog.show()

        def open_membrane_subtraction(self):
            reply = QMessageBox.question(
                self,
                "Particles Membrane Subtraction - Radonfit - MemXTerminator",
                "Ensure you have completed Membrane Analysis and the obtained averaged membranes and masks are as expected. \nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                try:
                    from .radonfit_cli import MembraneSubtractionApp
                except Exception as exc:
                    self._show_feature_import_error("Membrane Subtraction (Radonfit)", exc)
                    return
                self.membrane_subtraction_dialog = MembraneSubtractionApp(self)
                self.membrane_subtraction_dialog.show()

        def open_radon_micrograph_membrane_subtraction(self):
            reply = QMessageBox.question(
                self,
                "Micrograph Membrane Subtraction - MemXTerminator",
                "Ensure you have completed Particles Membrane Subtraction. \nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                try:
                    from .radonfit_cli import MicrographMembraneSubtraction_Radon_App
                except Exception as exc:
                    self._show_feature_import_error(
                        "Micrograph Membrane Subtraction (Radonfit)", exc
                    )
                    return
                self.micrograph_membrane_subtraction_dialog = (
                    MicrographMembraneSubtraction_Radon_App(self)
                )
                self.micrograph_membrane_subtraction_dialog.show()

        def open_membrane_analyzer_bezier(self):
            try:
                from .bezierfit_cli import MembraneAnalyzer_Bezier_App
            except Exception as exc:
                self._show_feature_import_error("Membrane Analyzer (Bezierfit)", exc)
                return
            self.membrane_analyzer_bezier_dialog = MembraneAnalyzer_Bezier_App(self)
            self.membrane_analyzer_bezier_dialog.show()

        def open_membrane_subtraction_bezier(self):
            reply = QMessageBox.question(
                self,
                "Particles Membrane Subtraction - Bezierfit - MemXTerminator",
                "Ensure you have completed Membrane Analysis. \nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                try:
                    from .bezierfit_cli import ParticleMembraneSubtraction_Bezier_App
                except Exception as exc:
                    self._show_feature_import_error("Membrane Subtraction (Bezierfit)", exc)
                    return
                self.membrane_subtraction_bezier_dialog = (
                    ParticleMembraneSubtraction_Bezier_App(self)
                )
                self.membrane_subtraction_bezier_dialog.show()

        def open_micrograph_membrane_subtraction_bezier(self):
            reply = QMessageBox.question(
                self,
                "Micrograph Membrane Subtraction - MemXTerminator",
                "Ensure you have completed Particles Membrane Subtraction, and converted the particles_selected.cs file to particles_selected.star file for Micrograph Membrane Subtraction. \nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                try:
                    from .bezierfit_cli import MicrographMembraneSubtraction_Bezier_App
                except Exception as exc:
                    self._show_feature_import_error(
                        "Micrograph Membrane Subtraction (Bezierfit)", exc
                    )
                    return
                self.micrograph_membrane_subtraction_bezier_dialog = (
                    MicrographMembraneSubtraction_Bezier_App(self)
                )
                self.micrograph_membrane_subtraction_bezier_dialog.show()

        def open_bezier_batch_scheduler(self):
            try:
                from .bezierfit_batch_scheduler_gui import BezierfitBatchSchedulerDialog
            except Exception as exc:
                self._show_feature_import_error("Bezierfit Batch Scheduler", exc)
                return
            self.bezier_batch_scheduler_dialog = BezierfitBatchSchedulerDialog(self)
            self.bezier_batch_scheduler_dialog.show()

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

def fix_mrc_file(fp):
    """Try to fix .mrc file Map ID."""
    try:
        with mrcfile.open(fp, "r+", permissive=True) as mrc:
            if not mrc.header.map == mrcfile.constants.MAP_ID:
                mrc.header.map = mrcfile.constants.MAP_ID
            if mrc.data is not None:
                mrc.update_header_from_data()
            else:
                print(f"ERROR with {fp}: data is None!")
        try:
            with mrcfile.open(fp, "r") as mrc:
                print(f"File {fp} opened successfully after fix.")
        except ValueError as e:
            print(f"ERROR with {fp}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {fp}: {e}")

def main():
    """main client"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_gui = subparsers.add_parser('gui')
    parser_fixmapid = subparsers.add_parser('fixmapid')
    parser_fixmapid.add_argument('file', nargs='+', help='Path to .mrc file(s)')
    parser_bezier_batch = subparsers.add_parser(
        'bezierfit-batch',
        help='Run multiple Bezierfit jobs in parallel from a JSON spec (batch scheduler).',
    )
    parser_bezier_batch.add_argument('--spec', required=True, help='Path to batch spec JSON file.')
    parser_bezier_batch.add_argument('--state', default=None, help='Path to write scheduler_state.json.')
    parser_bezier_batch.add_argument('--gpus', default=None, help="Override GPU list, e.g. '0,1,2' or 'auto'.")
    parser_bezier_batch.add_argument('--policy', choices=['fill_first', 'round_robin'], default=None)
    parser_bezier_batch.add_argument('--max_running_jobs', type=int, default=None)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'gui':
        _run_gui()

    if args.command == 'fixmapid':
        for file_path in args.file:
            fix_mrc_file(file_path)

    if args.command == 'bezierfit-batch':
        from memxterminator.bezierfit.scheduler.cli import main as scheduler_main

        argv = ['--spec', args.spec]
        if args.state:
            argv += ['--state', args.state]
        if args.gpus:
            argv += ['--gpus', args.gpus]
        if args.policy:
            argv += ['--policy', args.policy]
        if args.max_running_jobs is not None:
            argv += ['--max_running_jobs', str(int(args.max_running_jobs))]
        scheduler_main(argv)
