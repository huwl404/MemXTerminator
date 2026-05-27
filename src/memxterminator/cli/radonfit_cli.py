from ._gui_runtime import configure_gui_runtime

configure_gui_runtime()

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication, QDialog
from ..GUI.radon_gui import Ui_Form
from ..GUI.radonfit_membrane_analyzer_gui import Ui_MembraneAnalyzer
from ..GUI.radonfit_membrane_subtraction_gui import Ui_MembraneSubtraction
from ..GUI.MicrographMembraneSubtraction_Radonfit import Ui_MicrographMembraneSubtraction_Radonfit
import os
import json
import mrcfile
import subprocess
import shlex

from ._deps import check_cupy_cuda_available
from ._process import popen_kwargs_for_new_session, python_executable_for_subprocess, terminate_pid
from ..mxt_state import validate_output_dirname


class RadonApp(QtWidgets.QDialog, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        self.mrcPathLineEdit.setToolTip("Path to the 2D average MRC file you want to analyze.")
        self.sectionSpinBox.setToolTip("Select the section number you want to analyze from the MRC file.")
        self.cropRateLineEdit.setToolTip("Enter the crop rate value (between 0 and 1). This determines how much of the image is considered for analysis.")
        self.thrLineEdit.setToolTip("Enter the threshold value (between 0 and 1). This determines the sensitivity of the analysis.")
        self.jsonPathLineEdit.setToolTip("Path where the radonanalysis_info.json file will be saved. Default is the directory of the MRC file.")
        
        self.browseButton.clicked.connect(self.browse_file)
        self.previewButton.clicked.connect(self.preview_section)
        self.analyzeButton.clicked.connect(self.analyze_section)
        self.saveButton.clicked.connect(self.save_results)
        self.jsonBrowseButton.clicked.connect(self.browse_json_save_path)
        self.mrc_file = None
        self.image = None
        self.analyzer = None
        self.params = {}

    def _show_gpu_required(self, details: str) -> None:
        QMessageBox.critical(
            self,
            "GPU feature unavailable",
            (
                "This feature requires a CUDA-capable GPU and a working CuPy (`cupy`) installation.\n\n"
                f"{details}\n\n"
                "Tip: start the GUI on a GPU node (or ensure CUDA libraries are available) "
                "and try again."
            ),
        )

    def _import_radonfit_gpu(self):
        ok, details = check_cupy_cuda_available()
        if not ok:
            self._show_gpu_required(details)
            return None, None

        try:
            from ..radonfit.lib._utils import readmrc
            from ..radonfit.lib.radonanalyser import RadonAnalyzer
        except Exception as exc:
            self._show_gpu_required(f"Failed to import Radonfit GPU modules: {type(exc).__name__}: {exc}")
            return None, None

        return readmrc, RadonAnalyzer

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open MRC File", "", "MRC Files (*.mrc)")
        if filepath:
            self.mrcPathLineEdit.setText(filepath)
            self.mrc_file = filepath
            total_sections = mrcfile.open(self.mrc_file).data.shape[0]
            self.sectionSpinBox.setMaximum(total_sections - 1)
            for i in range(total_sections):
                self.params[str(i)] = {
                    'crop_rate': 0.6,
                    'thr': 0.7, 
                    'theta_start': 0,
                    'theta_end': 180
                }

    def preview_section(self):
        if self.mrc_file:
            readmrc, _RadonAnalyzer = self._import_radonfit_gpu()
            if readmrc is None:
                return
            section = self.sectionSpinBox.value()
            self.image = readmrc(self.mrc_file, section=section, mode='gpu')
            import matplotlib.pyplot as plt

            plt.imshow(self.image.get(), cmap='gray')
            plt.show()

    def analyze_section(self):
        try:
            readmrc, RadonAnalyzer = self._import_radonfit_gpu()
            if readmrc is None or RadonAnalyzer is None:
                return
            if self.image is None and self.mrc_file:
                section = self.sectionSpinBox.value()
                self.image = readmrc(self.mrc_file, section=section, mode='gpu')
            
            if self.image is not None:
                crop_rate = float(self.cropRateLineEdit.text())
                thr = float(self.thrLineEdit.text())
                theta_start = int(self.theta_start_lineEdit.text())
                theta_end = int(self.theta_end_lineEdit.text())
                self.analyzer = RadonAnalyzer('None', 0, self.image, crop_rate, thr, theta_start=theta_start, theta_end=theta_end)
                self.analyzer.visualize_analyze()
                section = self.sectionSpinBox.value()
                self.params[str(section)] = {
                    'crop_rate': crop_rate,
                    'thr': thr, 
                    'theta_start': theta_start,
                    'theta_end': theta_end
                }
        except Exception as e:
            print(f"Error: {e}")
            QMessageBox.warning(self, "Error", "Radon analysis failed. Please try again with different parameters.")
    def browse_json_save_path(self):
        default_dir = QtCore.QFileInfo(self.mrcPathLineEdit.text()).absolutePath()
        save_path, _ = QFileDialog.getSaveFileName(self, "Select JSON Save Path", default_dir, "JSON Files (*.json);;All Files (*)")
        if save_path:
            self.jsonPathLineEdit.setText(save_path)
    def save_results(self):
        save_path = self.jsonPathLineEdit.text() or 'radonanalysis_info.json'
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}
        data.update(self.params)
        with open(save_path, 'w') as file:
            json.dump(data, file, indent=4)
        QMessageBox.information(self, "Info", "Results saved successfully!")

class MembraneAnalyzerApp(QtWidgets.QDialog, Ui_MembraneAnalyzer):
    PID_FILE = "radonfit_membrane_analysis.pid"
    LOG_FILE = "radonfit_membrane_analysis.run.out"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.templates_starfile_browse_button.clicked.connect(self.browse_templates_starfile)
        self.particles_starfile_browse_button.clicked.connect(self.browse_particles_starfile)
        self.output_starfile_browse_button.clicked.connect(self.output_starfile)
        self.info_JSON_file_browse_button.clicked.connect(self.info_json_file)
        self.kappa_templates_checkBox.stateChanged.connect(self.generate_kappa_templates)


        self.templates_starfile_name = None
        self.particles_starfile_name = None
        self.output_starfile_name = None
        self.info_json_name = None
        self.kappa_template = False
        self.process = None
        self.kappa_number = None
        self.kappa_start_value = None
        self.kappa_end_value = None

        self.initialsigma1 = None
        self.initialsigma2 = None
        self.template_size = None
        self.sigmarange = None
        self.sigma_step = None
        self.curve_kappa_start = None
        self.curve_kappa_end = None
        self.curve_kappa_step = None
        self.edge_sigma_for_mask = None
        self.extra_mem_dist = None
        self.mem_edge_sigma = None

        self._process_pid = None
        self.check_running_process()
        self.Launch_button.clicked.connect(self.start_process)
        self.Kill_button.clicked.connect(self.kill_process)
        
        with open(self.LOG_FILE, "a") as f:
            f.write("Radonfit Membrane Analysis ready.\n")
        self.last_read_position = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)
        

    def browse_templates_starfile(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.templates_starfile_path_textedit.setText(filepath)
            self.templates_starfile_name = filepath

    def browse_particles_starfile(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.particles_starfile_path_textedit.setText(filepath)
            self.particles_starfile_name = filepath
    def output_starfile(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.output_starfile_path_textedit.setText(filepath)
            self.output_starfile_name = filepath
    def info_json_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json)")
        if filepath:
            self.info_json_file_textedit.setText(filepath)
            self.info_json_name = filepath
    def generate_kappa_templates(self, state):
        if state == QtCore.Qt.Checked:
            self.kappa_template = self.whichtemplate_spinBox.value()
        elif state == QtCore.Qt.Unchecked:
            self.kappa_template = False
    def start_process(self):
        ok, details = check_cupy_cuda_available()
        if not ok:
            QMessageBox.critical(
                self,
                "GPU feature unavailable",
                (
                    "Membrane Analyzer (Radonfit) requires a CUDA-capable GPU and CuPy.\n\n"
                    f"{details}"
                ),
            )
            return

        self.kappa_number = self.kappa_num_textedit.text()
        self.kappa_start_value = self.kappa_start_textedit.text()
        self.kappa_end_value = self.kappa_end_textedit.text()

        self.initialsigma1 = self.initialsigma1_textedit.text()
        self.initialsigma2 = self.initialsigma2_textedit.text()
        self.template_size = self.templatesize.text()
        self.sigmarange = self.select_range_spinbox.value()
        self.sigma_step = self.sigma_step_for_centerfit_textedit.text()
        self.curve_kappa_start = self.curve_kappa_start_textedit.text()
        self.curve_kappa_end = self.curve_kappa_end_textedit.text()
        self.curve_kappa_step = self.curve_kappa_step_textedit.text()
        self.edge_sigma_for_mask = self.edge_sigma_for_mask_textedit.text()
        self.extra_mem_dist = self.extra_mem_dist_textedit.text()
        self.mem_edge_sigma = self.edge_sigma_for_mem_average_textedit.text()
        params = ['--templates_starfile_name', f'{self.templates_starfile_name}',
                  '--output_filename', f'{self.output_starfile_name}',
                  '--particle_starfile_name', f'{self.particles_starfile_name}',
                  '--kappa_template', f'{int(self.kappa_template)}',
                  '--kappanum', f'{self.kappa_number}',
                  '--kappastart', f'{self.kappa_start_value}',
                  '--kappaend', f'{self.kappa_end_value}',
                  '--info_json', f'{self.info_json_name}',
                  '--sigma1', f'{self.initialsigma1}',
                  '--sigma2', f'{self.initialsigma2}',
                  '--template_size', f'{self.template_size}',
                  '--sigma_range', f'{self.sigmarange}',
                  '--sigma_step', f'{self.sigma_step}',
                  '--curve_kappa_start', f'{self.curve_kappa_start}',
                  '--curve_kappa_end', f'{self.curve_kappa_end}',
                  '--curve_kappa_step', f'{self.curve_kappa_step}',
                  '--edge_sigma', f'{self.edge_sigma_for_mask}',
                  '--extra_mem_dist', f'{self.extra_mem_dist}',
                  '--mem_edge_sigma', f'{self.mem_edge_sigma}']
        # self.process = subprocess.Popen(['python', 'membrane_analysis-main.py'] + params)

        with open(self.LOG_FILE, 'w') as f:
            # self.process = subprocess.Popen(['python','-u', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'membrane_analysis-main.py')] + params, stdout=f, stderr=subprocess.STDOUT)
            cmd = [python_executable_for_subprocess(), '-u', '-m', 'memxterminator.radonfit.bin.membrane_analysis-main'] + params
            try:
                f.write(f">>> Command: {shlex.join(cmd)}\n")
                f.flush()
            except Exception:
                pass
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                **popen_kwargs_for_new_session(),
            )
        print("Radonfit Membrane Analysis started with PID:", self.process.pid)
        self._process_pid = int(self.process.pid)
        with open(self.PID_FILE, 'w') as f:
            f.write(str(self.process.pid))
    def kill_process(self):
        pid = None
        if isinstance(self.process, subprocess.Popen):
            pid = int(self.process.pid)
        elif self._process_pid is not None:
            pid = int(self._process_pid)
        else:
            return

        terminate_pid(pid)
        print(f"Process PID {pid} terminated")
        self.process = None
        self._process_pid = None
        if os.path.exists(self.PID_FILE):
            os.remove(self.PID_FILE)
        self.timer.stop()
    def update_log(self):
        try:
            with open(self.LOG_FILE, 'r') as f:
                f.seek(self.last_read_position)
                new_content = f.read()
                self.last_read_position = f.tell()
            if new_content:
                self.textBrowser_log.append(new_content)
        except FileNotFoundError:
            self.textBrowser_log.append(f"Error: '{self.LOG_FILE}' file not found.")
        self._refresh_process_state()

    def _refresh_process_state(self) -> None:
        # If we launched the process in this GUI session, we have a Popen handle.
        if isinstance(self.process, subprocess.Popen):
            ret = self.process.poll()
            if ret is None:
                return
            pid = int(self.process.pid)
            self.process = None
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.textBrowser_log.append(f">>> DONE (PID={pid}, exit_code={ret})")
            return

        # If we only have a PID from a previous GUI session, check if it's still alive.
        if self._process_pid is None:
            return
        pid = int(self._process_pid)
        try:
            os.kill(pid, 0)
        except OSError:
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.textBrowser_log.append(f">>> DONE (PID={pid})")
    def check_running_process(self):
        if os.path.exists(self.PID_FILE):
            with open(self.PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if process is running
                self.process = None
                self._process_pid = pid
                print(f"Process with PID {pid} is still running!")
                # Here, you can also update the GUI to show that the process is running
            except OSError:
                print(f"Process with PID {pid} is not running.")
                os.remove(self.PID_FILE)

class MembraneSubtractionApp(QtWidgets.QDialog, Ui_MembraneSubtraction):
    PID_FILE = "radonfit_particle_pms.pid"
    LOG_FILE = "radonfit_particle_pms.run.out"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.particles_selected_starfile_button.clicked.connect(self.browse_particles_starfile)
        self.membrane_analysis_results_button.clicked.connect(self.browse_mem_analysis_starfile)

        worker_tip = (
            "Procs = worker processes for GPU computation (this is NOT CPU cores).\n"
            "0 = auto-detect from visible GPUs.\n"
            "Recommendation: for 1 GPU, use 1. Do not exceed the number of GPUs."
        )
        self.CPU_label.setText("Procs")
        self.CPU_label.setToolTip(worker_tip)
        self.CPU_lineEdit.setToolTip(worker_tip)
        batch_tip = (
            "Progress/reporting window for processed particle stacks.\n"
            "Actual GPU concurrency is controlled by Procs."
        )
        self.Batch_size_label.setToolTip(batch_tip)
        self.Batch_size_lineEdit.setToolTip(batch_tip)
        output_dir_tip = (
            "Single output directory name replacing extract/ in particle stack paths.\n"
            "Use a different name to compare membrane subtraction parameter sets."
        )
        self.Output_dirname_label.setToolTip(output_dir_tip)
        self.Output_dirname_lineEdit.setToolTip(output_dir_tip)


        self.mem_analysis_starfile_name = None
        self.particles_starfile_name = None
        self.bias = self.Bias_lineEdit.text()
        self.extra_mem_dist = self.Extra_mem_dist_lineEdit.text()
        self.scaling_factor_start = self.Scaling_factor_start_lineEdit.text()
        self.scaling_factor_end = self.Scaling_factor_end_lineEdit.text()
        self.scaling_factor_step = self.Step_lineEdit.text()
        self.cpu = self.CPU_lineEdit.text()
        self.batch_size = self.Batch_size_lineEdit.text()
        self.output_dirname = self.Output_dirname_lineEdit.text()
        self.process = None
        self._process_pid = None

        self.check_running_process()
        self.launch_button.clicked.connect(self.start_process)
        self.kill_button.clicked.connect(self.kill_process)

        self.show_command_button = QtWidgets.QPushButton("Command...", self.horizontalLayoutWidget)
        self.show_command_button.setToolTip("Show the exact CLI command that will be executed.")
        self.horizontalLayout.insertWidget(1, self.show_command_button)
        self.show_command_button.clicked.connect(self.show_command)
        
        with open(self.LOG_FILE, "a") as f:
            f.write("Radonfit Membrane Subtraction ready.\n")
        self.last_read_position = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)
        

    def browse_mem_analysis_starfile(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.membrane_analysis_file_lineEdit.setText(filepath)
            self.mem_analysis_starfile_name = filepath

    def browse_particles_starfile(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.particles_selected_starfile_lineEdit.setText(filepath)
            self.particles_starfile_name = filepath

    def _build_cmd(self) -> list[str]:
        params = [
            "--particles_selected_filename",
            self.particles_selected_starfile_lineEdit.text(),
            "--membrane_analysis_filename",
            self.membrane_analysis_file_lineEdit.text(),
            "--bias",
            self.Bias_lineEdit.text(),
            "--extra_mem_dist",
            self.Extra_mem_dist_lineEdit.text(),
            "--scaling_factor_start",
            self.Scaling_factor_start_lineEdit.text(),
            "--scaling_factor_end",
            self.Scaling_factor_end_lineEdit.text(),
            "--scaling_factor_step",
            self.Step_lineEdit.text(),
            "--procs",
            self.CPU_lineEdit.text(),
            "--batch_size",
            self.Batch_size_lineEdit.text(),
            "--output_dirname",
            self.Output_dirname_lineEdit.text(),
        ]
        return [python_executable_for_subprocess(), "-u", "-m", "memxterminator.radonfit.bin.membrane_subtract-main"] + params

    def show_command(self) -> None:
        from ._command_preview import CommandPreviewDialog

        CommandPreviewDialog(self._build_cmd(), self).exec_()

    def start_process(self):
        ok, details = check_cupy_cuda_available()
        if not ok:
            QMessageBox.critical(
                self,
                "GPU feature unavailable",
                (
                    "Membrane Subtraction (Radonfit) requires a CUDA-capable GPU and CuPy.\n\n"
                    f"{details}"
                ),
            )
            return

        self.mem_analysis_starfile_name = self.membrane_analysis_file_lineEdit.text()
        self.particles_starfile_name = self.particles_selected_starfile_lineEdit.text()
        self.bias = self.Bias_lineEdit.text()
        self.extra_mem_dist = self.Extra_mem_dist_lineEdit.text()
        self.scaling_factor_start = self.Scaling_factor_start_lineEdit.text()
        self.scaling_factor_end = self.Scaling_factor_end_lineEdit.text()
        self.scaling_factor_step = self.Step_lineEdit.text()
        self.cpu = self.CPU_lineEdit.text()
        self.batch_size = self.Batch_size_lineEdit.text()
        try:
            self.output_dirname = validate_output_dirname(self.Output_dirname_lineEdit.text())
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid output directory name", str(exc))
            return
        with open(self.LOG_FILE, 'w') as f:
            cmd = self._build_cmd()
            particles_star = self.particles_selected_starfile_lineEdit.text()
            base, ext = os.path.splitext(particles_star)
            if ext.lower() != ".star":
                base = particles_star
            out_star_all = f"{base}_{self.output_dirname}.star"
            out_star_completed = f"{base}_{self.output_dirname}_completed.star"
            f.write(f">>> STAR outputs (for cryoSPARC import): {out_star_all}\n")
            f.write(f">>> STAR outputs (completed-only): {out_star_completed}\n")
            try:
                f.write(f">>> Command: {shlex.join(cmd)}\n")
                f.flush()
            except Exception:
                pass
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                **popen_kwargs_for_new_session(),
            )
        print("Radonfit Membrane Subtraction started with PID:", self.process.pid)
        print(
            "Radonfit Membrane Subtraction started using "
            f"{self.cpu} procs, batch size {self.batch_size}, output_dirname={self.output_dirname}"
        )
        self._process_pid = int(self.process.pid)
        with open(self.PID_FILE, 'w') as f:
            f.write(str(self.process.pid))
    def kill_process(self):
        pid = None
        if isinstance(self.process, subprocess.Popen):
            pid = int(self.process.pid)
        elif self._process_pid is not None:
            pid = int(self._process_pid)
        else:
            return

        terminate_pid(pid)
        print(f"Process PID {pid} terminated")
        self.process = None
        self._process_pid = None
        if os.path.exists(self.PID_FILE):
            os.remove(self.PID_FILE)
        self.timer.stop()
    def update_log(self):
        # 读取日志文件内容
        try:
            with open(self.LOG_FILE, 'r') as f:
                f.seek(self.last_read_position)  # 跳转到上次读取的位置
                new_content = f.read()  # 读取新内容
                self.last_read_position = f.tell()  # 更新读取的位置
            if new_content:
                self.textBrowser_log.append(new_content)
        except FileNotFoundError:
            self.textBrowser_log.append(f"Error: '{self.LOG_FILE}' file not found.")
        self._refresh_process_state()

    def _refresh_process_state(self) -> None:
        if isinstance(self.process, subprocess.Popen):
            ret = self.process.poll()
            if ret is None:
                return
            pid = int(self.process.pid)
            self.process = None
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.textBrowser_log.append(f">>> DONE (PID={pid}, exit_code={ret})")
            return

        if self._process_pid is None:
            return
        pid = int(self._process_pid)
        try:
            os.kill(pid, 0)
        except OSError:
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.textBrowser_log.append(f">>> DONE (PID={pid})")
    def check_running_process(self):
        if os.path.exists(self.PID_FILE):
            with open(self.PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if process is running
                self.process = None
                self._process_pid = pid
                print(f"Process with PID {pid} is still running!")
                # Here, you can also update the GUI to show that the process is running
            except OSError:
                print(f"Process with PID {pid} is not running.")
                os.remove(self.PID_FILE)


class MicrographMembraneSubtraction_Radon_App(QtWidgets.QDialog, Ui_MicrographMembraneSubtraction_Radonfit):
    PID_FILE = "radonfit_micrograph_mms.pid"
    LOG_FILE = "radonfit_micrograph_mms.run.out"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.particles_selected_starfile_browse_pushButton.clicked.connect(self.particle_browse)
        self.process = None
        self._process_pid = None
        self.cpus = None
        self.batch_size = None
        self.output_dirname = self.output_dirname_lineEdit.text()
        self.particle = None
        output_dir_tip = (
            "Output directory name used by RadonFit particle membrane subtraction.\n"
            "Use the same value here so micrograph subtraction can find the PMS stacks and .mxt files."
        )
        self.output_dirname_label.setToolTip(output_dir_tip)
        self.output_dirname_lineEdit.setToolTip(output_dir_tip)
        self.check_running_process()
        self.launch_pushButton.clicked.connect(self.start_process)
        self.kill_pushButton.clicked.connect(self.kill_process)

        self.show_command_button = QtWidgets.QPushButton("Command...", self.horizontalLayoutWidget_2)
        self.show_command_button.setToolTip("Show the exact CLI command that will be executed.")
        self.horizontalLayout_2.insertWidget(1, self.show_command_button)
        self.show_command_button.clicked.connect(self.show_command)
        
        with open(self.LOG_FILE, "a") as f:
            f.write("Micrograph Membrane Subtraction ready.\n")
        self.last_read_position = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)
    
    def particle_browse(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open STAR File", "", "STAR Files (*.star)")
        if filepath:
            self.particles_selected_starfile_lineEdit.setText(filepath)
            self.particle = filepath

    def _build_cmd(self, *, coerce_numbers: bool) -> list[str]:
        particles_star = self.particles_selected_starfile_lineEdit.text()
        cpus = self.cpus_lineEdit.text()
        batch_size = self.batch_size_lineEdit.text()
        output_dirname = self.output_dirname_lineEdit.text()

        if coerce_numbers:
            cpus = str(int(cpus))
            batch_size = str(int(batch_size))
            output_dirname = validate_output_dirname(output_dirname)

        params = [
            "--particles_selected_filename",
            particles_star,
            "--procs",
            cpus,
            "--batch_size",
            batch_size,
            "--output_dirname",
            output_dirname,
        ]
        return [python_executable_for_subprocess(), "-u", "-m", "memxterminator.radonfit.bin.micrograph_mem_subtraction"] + params

    def show_command(self) -> None:
        from ._command_preview import CommandPreviewDialog

        try:
            cmd = self._build_cmd(coerce_numbers=True)
        except Exception:
            cmd = self._build_cmd(coerce_numbers=False)
        CommandPreviewDialog(cmd, self).exec_()
    
    def start_process(self):
        ok, details = check_cupy_cuda_available()
        if not ok:
            QMessageBox.critical(
                self,
                "GPU feature unavailable",
                (
                    "Micrograph Membrane Subtraction (Radonfit) requires a CUDA-capable GPU and CuPy.\n\n"
                    f"{details}"
                ),
            )
            return

        self.particle = self.particles_selected_starfile_lineEdit.text()
        self.cpus = self.cpus_lineEdit.text()
        self.batch_size = self.batch_size_lineEdit.text()
        try:
            self.output_dirname = validate_output_dirname(self.output_dirname_lineEdit.text())
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid output directory name", str(exc))
            return
        with open(self.LOG_FILE, 'w') as f:
            cmd = self._build_cmd(coerce_numbers=True)
            try:
                f.write(f">>> Command: {shlex.join(cmd)}\n")
                f.flush()
            except Exception:
                pass
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                **popen_kwargs_for_new_session(),
            )
        print("Micrograph Membrane Subtraction started with PID:", self.process.pid)
        print(
            "Micrograph Membrane Subtraction started using "
            f"{self.cpus} procs, batch size {self.batch_size}, output_dirname={self.output_dirname}"
        )
        self._process_pid = int(self.process.pid)
        with open(self.PID_FILE, 'w') as f:
            f.write(str(self.process.pid))
    def kill_process(self):
        pid = None
        if isinstance(self.process, subprocess.Popen):
            pid = int(self.process.pid)
        elif self._process_pid is not None:
            pid = int(self._process_pid)
        else:
            return

        terminate_pid(pid)
        print(f"Process PID {pid} terminated")
        self.process = None
        self._process_pid = None
        if os.path.exists(self.PID_FILE):
            os.remove(self.PID_FILE)
        self.timer.stop()
    def update_log(self):
        try:
            with open(self.LOG_FILE, 'r') as f:
                f.seek(self.last_read_position)
                new_content = f.read()
                self.last_read_position = f.tell()
            if new_content:
                self.LOG_textBrowser.append(new_content)
        except FileNotFoundError:
            self.LOG_textBrowser.append(f"Error: '{self.LOG_FILE}' file not found.")
        self._refresh_process_state()

    def _refresh_process_state(self) -> None:
        if isinstance(self.process, subprocess.Popen):
            ret = self.process.poll()
            if ret is None:
                return
            pid = int(self.process.pid)
            self.process = None
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.LOG_textBrowser.append(f">>> DONE (PID={pid}, exit_code={ret})")
            return

        if self._process_pid is None:
            return
        pid = int(self._process_pid)
        try:
            os.kill(pid, 0)
        except OSError:
            self._process_pid = None
            if os.path.exists(self.PID_FILE):
                os.remove(self.PID_FILE)
            self.LOG_textBrowser.append(f">>> DONE (PID={pid})")
    def check_running_process(self):
        if os.path.exists(self.PID_FILE):
            with open(self.PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Check if process is running
                self.process = None
                self._process_pid = pid
                print(f"Process with PID {pid} is still running!")
            except OSError:
                print(f"Process with PID {pid} is not running.")
                os.remove(self.PID_FILE)
