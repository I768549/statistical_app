import sys
import numpy as np
from scipy.stats import t
import math
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, QTabWidget,QTableWidget, QHeaderView, QTableWidgetItem,
                             QRadioButton, QButtonGroup, QGroupBox, QTextEdit,
                             QMessageBox, QFileDialog, QSizePolicy, QStackedWidget, QComboBox)
from PyQt6.QtGui import QIcon, QFontDatabase, QDoubleValidator, QIntValidator

from main_functions import *
from dist_generation import *
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ExpHelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exponential window")
        self.setGeometry(800, 200, 700, 700)  # Position to right of main window
        main_layout = QVBoxLayout()
        plot_layout = QHBoxLayout()

        self.exp_porb_widget = pg.PlotWidget()
        self.exp_porb_widget.setLabel('bottom', 'Values')
        self.exp_porb_widget.showGrid(x=True, y=True, alpha=0.3)

        plot_layout.addWidget(self.exp_porb_widget)
        main_layout.addLayout(plot_layout)

        self.setLayout(main_layout)


class StatisticalApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        self.raw_dist_data = None
        self.processed_data = None
        self.current_file_path = ""
        self.modifications_log = []
        self.intervals_array = []

        self._initialize_ui()
        self._connect_signals()
        self._update_ui_state()
        self.ExpHelpWindow = None

        
    def _initialize_ui(self):
        self.setWindowTitle('Statistical Application')

        try:
            self.setWindowIcon(QIcon('smile.png'))
        except Exception as e:
            print(f"Could't load an icon: {e}")
        self.setGeometry(600, 100, 1000, 800)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        # File load section
        file_layout = QHBoxLayout()
        self.file_button = QPushButton(" Load a distribution")
        self.file_label = QLabel("(file isn't loaded yet)")
        self.update_button = QPushButton("🔄 Update Analysis")
        file_layout.addWidget(self.update_button)
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label)
        main_layout.addLayout(file_layout)

        # Analysis controls section
        controls_layout = QVBoxLayout()

        # Upper controls row
        controls_layout1 = QHBoxLayout()

        # Bins Group Box
        class_group_box = QGroupBox("Bins")
        class_layout = QHBoxLayout(class_group_box)

        self.rd_default = QRadioButton("Default")
        self.rd_custom = QRadioButton("Custom:")
        self.rd_default.setChecked(True)

        self.custom_entry = QLineEdit()
        self.custom_entry.setPlaceholderText("bins amount")
        self.custom_entry.setEnabled(False)

        self.rd_custom.toggled.connect(lambda checked: self.custom_entry.setEnabled(checked))

        class_layout.addWidget(self.rd_default)
        class_layout.addWidget(self.rd_custom)
        class_layout.addWidget(self.custom_entry)
        controls_layout1.addWidget(class_group_box)

        outliers_anomalies_group_box = QGroupBox("Outliers and anomalies")
        outliers_anomalies_layout = QHBoxLayout(outliers_anomalies_group_box)

        trim_layout = QHBoxLayout()
        self.rd_del_outliers = QRadioButton("Deleting quantile")
        self.rd_del_outliers.setChecked(True)
        self.rd_del_outliers.setAutoExclusive(False)

        self.quantile_entry = QLineEdit("0")
        self.quantile_entry.setPlaceholderText("0 - 0.5")
        self.quantile_entry.setEnabled(False)

        self.rd_del_outliers.toggled.connect(lambda checked: self.quantile_entry.setEnabled(checked))

        trim_layout.addWidget(self.rd_del_outliers)
        trim_layout.addWidget(self.quantile_entry)

        # Anomaly Removal by Z-score
        anomaly_layout = QHBoxLayout()
        self.rd_del_anomalies = QRadioButton("Z-score anomaly del")
        self.rd_del_anomalies.setAutoExclusive(False)
        self.z_score_entry = QLineEdit("1.96")
        self.z_score_entry.setEnabled(False)

        self.rd_del_anomalies.toggled.connect(lambda checked: self.z_score_entry.setEnabled(checked))

        anomaly_layout.addWidget(self.rd_del_anomalies)
        anomaly_layout.addWidget(self.z_score_entry)

        # Anomaly removal by Kurtosis and Asymmetry
        anomaly_kurtosis_layout = QVBoxLayout()
        self.rd_del_anomalies_kurtosis = QRadioButton("Delete anomalies")
        self.rd_del_anomalies_kurtosis.setAutoExclusive(False)
        anomaly_kurtosis_layout.addWidget(self.rd_del_anomalies_kurtosis)

        prob_paper_layout = QHBoxLayout()
        self.prob_paper_button = QPushButton("Show exp. prob. paper")
        prob_paper_layout.addWidget(self.prob_paper_button)

        outliers_anomalies_layout.addLayout(trim_layout)
        outliers_anomalies_layout.addLayout(anomaly_layout)
        outliers_anomalies_layout.addLayout(anomaly_kurtosis_layout)
        outliers_anomalies_layout.addLayout(prob_paper_layout)

        controls_layout1.addWidget(outliers_anomalies_group_box)
        controls_layout1.addStretch()

        controls_layout.addLayout(controls_layout1)
        main_layout.addLayout(controls_layout)

        plot_layout = QHBoxLayout()


        #Histogramm
        self.histogram_widget = pg.PlotWidget()
        self.histogram_widget.setLabel('left', 'Relative frequencies')
        self.histogram_widget.setLabel('bottom', 'Values')
        self.histogram_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.histogram_widget)

        # Виджет ECDF
        self.ecdf_widget = pg.PlotWidget()
        self.ecdf_widget.setLabel('right', 'Cumulative probabillity')
        self.ecdf_widget.setLabel('bottom', 'Values')
        self.ecdf_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.ecdf_widget)
        main_layout.addLayout(plot_layout)

        #Mode selection buttons
        mode_buttons_layout = QHBoxLayout()

        self.analysis_button = QPushButton("📊 Analysis View")
        self.analysis_button.setCheckable(True)
        self.analysis_button.setChecked(True)

        self.transform_button = QPushButton("🔄 Transform View")
        self.transform_button.setCheckable(True)

        self.generate_dist_button = QPushButton("🎲Random distributions and hypotheses test")
        self.generate_dist_button.setCheckable(True)

        # Create button group for exclusive selection
        self.view_button_group = QButtonGroup()
        self.view_button_group.addButton(self.analysis_button, 1)
        self.view_button_group.addButton(self.transform_button, 2) 
        self.view_button_group.addButton(self.generate_dist_button, 3)

        mode_buttons_layout.addWidget(self.analysis_button) 
        mode_buttons_layout.addWidget(self.transform_button)     
        mode_buttons_layout.addWidget(self.generate_dist_button)

        mode_buttons_layout.addStretch()
        main_layout.addLayout(mode_buttons_layout)
        
        #stacked widget to switch between analysis transformation and generation
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create analysis panel (statistics output)
        self.analysis_panel = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_panel)

        stats_label = QLabel("Statistics:")
        self.statistics_output = QTextEdit()
        self.statistics_output.setReadOnly(True)
        self.statistics_output.setMinimumHeight(250)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.statistics_output.setFont(font)
        analysis_layout.addWidget(stats_label)
        analysis_layout.addWidget(self.statistics_output)
        
        # Create transformation panel
        self.transform_panel = QWidget()
        transform_layout = QVBoxLayout(self.transform_panel)
        
        # Shift Data
        shift_group = QGroupBox("Shift Data")
        shift_layout = QHBoxLayout(shift_group)
        self.shift_input = QLineEdit()
        self.shift_input.setPlaceholderText("")
        self.apply_shift_button = QPushButton("Apply shift")
        shift_layout.addWidget(QLabel("Shift value at + n position:"))
        shift_layout.addWidget(self.shift_input)
        shift_layout.addWidget(self.apply_shift_button)
        transform_layout.addWidget(shift_group)

        # Logarithm
        log_group = QGroupBox("Logarithmize data")
        log_layout = QHBoxLayout(log_group)
        self.apply_log_button = QPushButton("Apply ln")
        log_layout.addWidget(self.apply_log_button)
        log_layout.addStretch()
        transform_layout.addWidget(log_group)

        # Standardize data
        standardize_group = QGroupBox("Standardize data")
        standardize_layout = QHBoxLayout(standardize_group)
        self.apply_standartization_button = QPushButton("Standardize data")
        standardize_layout.addWidget(self.apply_standartization_button)
        standardize_layout.addStretch()
        transform_layout.addWidget(standardize_group)

        # Reset data
        reset_group = QGroupBox("Get back to initial data")
        reset_layout = QHBoxLayout(reset_group)
        self.reset_data_button = QPushButton("❌ Restore all transformations")
        reset_layout.addWidget(self.reset_data_button)
        reset_layout.addStretch()
        transform_layout.addWidget(reset_group)
        
        transform_layout.addStretch()

        #generation panel
        self.generation_panel = QWidget()
        generation_layout = QVBoxLayout(self.generation_panel)
        label_arb_gen = QLabel("You can generate distribution of an available type here:")
        generation_layout.addWidget(label_arb_gen)

        
        dist_chooser_layout = QHBoxLayout()
        self.generate_distribution_push_button = QPushButton("Generate distribution of a choosen type")
        self.falling_list = QComboBox()
        self.falling_list.addItems(["Exponential", "Normal", "Weibull", "Uniform", "Laplace"])
        size_label = QLabel("Size n: ")
        self.size_line = QLineEdit()

        dist_chooser_layout.addWidget(self.generate_distribution_push_button)
        dist_chooser_layout.addWidget(self.falling_list)
        dist_chooser_layout.addWidget(size_label)
        dist_chooser_layout.addWidget(self.size_line)

        #params 
        self.params_layout = QHBoxLayout()

        #Exp
        self.lambda_label = QLabel("Lambda: ")
        self.lambda_line = QLineEdit("0")
        self.params_layout.addWidget(self.lambda_label)
        self.params_layout.addWidget(self.lambda_line)
        self.lambda_label.hide()
        self.lambda_line.hide()

        #Uniform
        self.a_label = QLabel("a: ")
        self.b_label = QLabel("b: ")
        self.a_line = QLineEdit("0")
        self.b_line = QLineEdit("1")

        self.params_layout.addWidget(self.a_label)
        self.params_layout.addWidget(self.a_line)
        self.params_layout.addWidget(self.b_label)
        self.params_layout.addWidget(self.b_line)

        self.a_label.hide()
        self.a_line.hide()
        self.b_label.hide()
        self.b_line.hide()

        #Laplace
        self.lam2_label = QLabel("lambda: ")
        self.mean2_label = QLabel("mean: ")
        self.lam2_line = QLineEdit("0")
        self.mean2_line = QLineEdit("1")

        self.params_layout.addWidget(self.lam2_label)
        self.params_layout.addWidget(self.lam2_line)
        self.params_layout.addWidget(self.mean2_label)
        self.params_layout.addWidget(self.mean2_line)

        self.lam2_label.hide()
        self.lam2_line.hide()
        self.mean2_label.hide()
        self.mean2_line.hide()


        #Weibull
        self.alpha_label = QLabel("alpha: ")
        self.alpha_line = QLineEdit("1")
        self.beta_label = QLabel("beta: ")
        self.beta_line = QLineEdit("1")

        self.params_layout.addWidget(self.alpha_label)
        self.params_layout.addWidget(self.alpha_line)
        self.params_layout.addWidget(self.beta_label)
        self.params_layout.addWidget(self.beta_line)

        self.alpha_label.hide()
        self.alpha_line.hide()
        self.beta_label.hide()
        self.beta_line.hide()

        #Normal Box-Muller
        self.mean_label = QLabel("mean: ")
        self.mean_line = QLineEdit("0")
        self.std_label = QLabel("std: ")
        self.std_line = QLineEdit("1")

        self.params_layout.addWidget(self.mean_label)
        self.params_layout.addWidget(self.mean_line)
        self.params_layout.addWidget(self.std_label)
        self.params_layout.addWidget(self.std_line)

        self.mean_label.hide()
        self.mean_line.hide()
        self.std_label.hide()
        self.std_line.hide()
        dist_chooser_layout.addLayout(self.params_layout)

        #t-test layout
        t_test_layout = QVBoxLayout()
        t_test_choosing_layout = QHBoxLayout()

        label_t_test = QLabel("Perfome a t-test for a choosen distribution here:")
        significance_level_label = QLabel("alpha:")
        self.significance_level_line = QLineEdit("0")
        self.falling_list_2 = QComboBox()
        self.falling_list_2.addItems(["Exponential"])
        param_value_label = QLabel("Parameter value:")
        self.param_value_line = QLineEdit("5")
        self.perfom_t_test_qpushbut = QPushButton("Perfom a t-test")

        #T_test Table
        self.t_test_result_table = QTableWidget()
        self.t_test_result_table.setRowCount(7) 
        self.t_test_result_table.setColumnCount(6)

        # Set row headers (sample sizes)
        sample_sizes = ["20", "50", "100", "400", "1000", "2000", "5000"]
        self.t_test_result_table.setVerticalHeaderLabels(sample_sizes)

        # Set column headers - modify these based on your exact needs
        column_headers = ["E{λ'}", "σ{λ'}", "E{t-statistics}", "σ{T-statistics}", "T-value(1-a/2)", "alpha"]  # Example column headers
        self.t_test_result_table.setHorizontalHeaderLabels(column_headers)

        # Set reasonable size for the table
        self.t_test_result_table.setMinimumHeight(238)
        self.t_test_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        #t-test_widgets_adding
        t_test_layout.addWidget(label_t_test)
        t_test_choosing_layout.addWidget(self.falling_list_2)
        t_test_choosing_layout.addWidget(param_value_label)
        t_test_choosing_layout.addWidget(self.param_value_line)
        t_test_choosing_layout.addWidget(significance_level_label)
        t_test_choosing_layout.addWidget(self.significance_level_line)
        t_test_choosing_layout.addWidget(self.perfom_t_test_qpushbut)


        t_test_layout.addLayout(t_test_choosing_layout)
        t_test_layout.addWidget(self.t_test_result_table)

        self.perfom_t_test_qpushbut.clicked.connect(self.handle_t_test)

        #Adding layouts
        generation_layout.addLayout(dist_chooser_layout)
        generation_layout.addLayout(t_test_layout)
        generation_layout.addStretch()

        # Add both panels to the stacked widget
        self.stacked_widget.addWidget(self.analysis_panel)
        self.stacked_widget.addWidget(self.transform_panel)
        self.stacked_widget.addWidget(self.generation_panel)
        
        # Connect view switching buttons
        self.analysis_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.transform_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.generate_dist_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))


    def _connect_signals(self):
        self.file_button.clicked.connect(self._load_data)
        self.update_button.clicked.connect(self._update_analysis)

        self.rd_custom.toggled.connect(self.custom_entry.setEnabled)
        self.rd_del_outliers.toggled.connect(self.quantile_entry.setEnabled)
        self.rd_del_anomalies.toggled.connect(self.z_score_entry.setEnabled)

        #Exp button 
        self.prob_paper_button.clicked.connect(self.show_prob_paper)

        # Connect transformation buttons
        self.apply_shift_button.clicked.connect(self._apply_shift)
        self.apply_log_button.clicked.connect(self._apply_logarithm)
        self.apply_standartization_button.clicked.connect(self._standartise_data)
        self.generate_distribution_push_button.clicked.connect(self._generate_distribution)
        self.reset_data_button.clicked.connect(self._reset_data)
        self.falling_list.currentTextChanged.connect(self._update_ui_state)


    def _reset_data(self):
        if self.raw_dist_data is None:
            QMessageBox.warning(self, "No data", "Initial data isn't loaded to reset it")
            return

        reply = QMessageBox.question(self, "Reset Confirmation",
                                     "This action will undo all transformations (shift, logarithm) and restore the data to its original state loaded from the file.\nContinue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                     QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            self.processed_data = self.raw_dist_data.copy()
            self.modifications_log = ["Data loaded from file.", "Reset to original data."]
            QMessageBox.information(self, "Data Reset",
                                    "Data has been restored to its original state. Updating analysis...")
            self.shift_input.clear()
            self.stacked_widget.setCurrentIndex(0)  # Switch to analysis view
            self._update_analysis()

    def _update_ui_state(self):
        data_loaded = self.processed_data is not None
        self.rd_default.setEnabled(data_loaded)
        self.rd_custom.setEnabled(data_loaded)
        self.custom_entry.setEnabled(data_loaded and self.rd_custom.isChecked())
        self.rd_del_outliers.setEnabled(data_loaded)
        self.quantile_entry.setEnabled(data_loaded and self.rd_del_outliers.isChecked())
        self.rd_del_anomalies.setEnabled(data_loaded)
        self.z_score_entry.setEnabled(data_loaded and self.rd_del_anomalies.isChecked())
        self.rd_del_anomalies_kurtosis.setEnabled(data_loaded)
        self.update_button.setEnabled(data_loaded)
        self.shift_input.setEnabled(data_loaded)
        self.apply_shift_button.setEnabled(data_loaded)
        self.apply_log_button.setEnabled(data_loaded)
        self.reset_data_button.setEnabled(data_loaded)
        self.apply_standartization_button.setEnabled(data_loaded)

        self.current_dist = self.falling_list.currentText()
        #Hide lambda
        self.lambda_label.hide()
        self.lambda_line.hide()
        #Hide uniform
        self.a_label.hide()
        self.a_line.hide()
        self.b_label.hide()
        self.b_line.hide()
        #Hide Weibull
        self.alpha_label.hide()
        self.alpha_line.hide()
        self.beta_label.hide()
        self.beta_line.hide()
        #Hide Normal
        self.mean_label.hide()
        self.mean_line.hide()
        self.std_label.hide()
        self.std_line.hide()     
        #Hide Laplace
        self.lam2_label.hide()
        self.lam2_line.hide()
        self.mean2_label.hide()
        self.mean2_line.hide()   

        if self.current_dist == "Exponential":
            self.lambda_label.show()
            self.lambda_line.show()
        elif self.current_dist == "Uniform":
            self.a_label.show()
            self.a_line.show()
            self.b_label.show()
            self.b_line.show()
        elif self.current_dist == "Weibull":
            self.alpha_label.show()
            self.alpha_line.show()
            self.beta_label.show()
            self.beta_line.show()
        elif self.current_dist == "Normal":
            self.mean_label.show()
            self.mean_line.show()
            self.std_label.show()
            self.std_line.show()
        elif self.current_dist == "Laplace":
            self.lam2_label.show()
            self.lam2_line.show()
            self.mean2_label.show()
            self.mean2_line.show()

    def _load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file with distribution data",
            "",
            "Text files (*.txt);;Data files (*.dat);;All files (*)"
        )
        if not file_name:
            return

        try:
            self.raw_dist_data = read_distribution(file_name)
            self.processed_data = self.raw_dist_data.copy()
            self.current_file_path = file_name
            self.file_label.setText(f"Loaded: {QtCore.QFileInfo(file_name).fileName()}")
            self.file_label.setToolTip(file_name)
            self.modifications_log = ["Data loaded from file."]
            self._update_ui_state()
            self._update_analysis()
            self.stacked_widget.setCurrentIndex(0)  # Switch to analysis view

        except Exception as e:
            error_message = f"File loading error:\n{str(e)}"
            QMessageBox.critical(self, "File Error", error_message)
            self.raw_dist_data = None
            self.processed_data = None
            self.current_file_path = ""
            self.file_label.setText("File loading error.")
            self.file_label.setToolTip("")
            self.modifications_log = []
            self._clear_plots_and_stats()
            self._update_ui_state()

    def handle_t_test(self):
        try:
            true_param_value = float(self.param_value_line.text())
            significance_level = float(self.significance_level_line.text())
            if true_param_value <= 0:
                QMessageBox.warning(self, "Error","Lambda value must be > 0")
        except ValueError:
            QMessageBox.warning(self, "input error", "Enter a number")
        alpha_values = [0.80, 0.70, 0.40, 0.20, 0.10, 0.05, 0.05]
        #alpha = significance_level
        sample_sizes_str = ["20", "50", "100", "400", "1000", "2000", "5000"]
        sample_sizes_int =[int(element) for element in sample_sizes_str]
        experiment_amount = 450
        for idx, sample_size in enumerate(sample_sizes_int):
            if significance_level <= 0:
                alpha = alpha_values[idx]
            else:
                alpha = significance_level
            estimated_lambdas = []
            estimated_t_statistics = []
            for _ in range(experiment_amount):
                #sample
                simulated_exp_distr = generate_exp_theoretical_dist(sample_size, true_param_value)
                sample_mean = arithmetic_mean(simulated_exp_distr)
                sample_std = math.sqrt(unbiased_sample_variance(simulated_exp_distr, sample_mean))
                #estimated lambda

                estimated_lambda = 1/sample_mean
                estimated_lambdas.append(estimated_lambda)
                #se_estimated_lambda = estimated_lambda/math.sqrt(sample_size)
                true_mean = 1/true_param_value
                #t-statistics
                t_stat = (sample_mean-true_mean)/(sample_std/math.sqrt(sample_size))
                estimated_t_statistics.append(t_stat)
            #lambdas values
            mean_estimated_lambdas = arithmetic_mean(estimated_lambdas)
            std_estimated_lambdas = math.sqrt(unbiased_sample_variance(estimated_lambdas, mean_estimated_lambdas))
            #t-statistics values
            mean_estimated_t_statistics = arithmetic_mean(estimated_t_statistics)
            std_estimated_t_statistics = math.sqrt(unbiased_sample_variance(estimated_t_statistics, mean_estimated_t_statistics))
            #TODO: разобраться с этой хуйней
            t_critical = t.ppf(1-alpha/2, sample_size-1)

            self.t_test_result_table.setItem(idx, 0, QTableWidgetItem(f"{mean_estimated_lambdas:.4f}"))
            self.t_test_result_table.setItem(idx, 1, QTableWidgetItem(f"{std_estimated_lambdas:.4f}"))
            self.t_test_result_table.setItem(idx, 2, QTableWidgetItem(f"{mean_estimated_t_statistics:.4f}"))
            self.t_test_result_table.setItem(idx, 3, QTableWidgetItem(f"{std_estimated_t_statistics:.4f}"))
            self.t_test_result_table.setItem(idx, 4, QTableWidgetItem(f"{t_critical:.4f}"))
            self.t_test_result_table.setItem(idx, 5, QTableWidgetItem(f"{alpha}"))
        
    def show_prob_paper(self):
        if self.ExpHelpWindow is None:
            self.ExpHelpWindow = ExpHelpWindow()
        
        # Get the current data for plotting
        if self.processed_data is not None:
            # Calculate ECDF for the current processed data
            x_axis_ecdf, y_axis_ecdf = ecdf(self.processed_data)
            
            # Update the probability paper plot in the secondary window
            self.plot_qq_exponential(self.processed_data)
            #self._update_exp_prob_paper(x_axis_ecdf, y_axis_ecdf)
        
        self.ExpHelpWindow.show()
        self.ExpHelpWindow.activateWindow()

    def _apply_shift(self):
        if self.processed_data is None:
            QMessageBox.warning(self, "No Data", "Please load the data first.")
            return

        try:
            shift_value_str = self.shift_input.text().strip().replace(',', '.')
            if not shift_value_str:
                shift_value = 0.0
            else:
                shift_value = float(shift_value_str)

            self.processed_data = shift_data(self.processed_data, shift_value)
            self.modifications_log.append(f"Shift applied: {shift_value:+.4f}")
            QMessageBox.information(self, "Shift Applied",
                                    f"Shift of {shift_value:+.4f} applied.\nUpdating analysis...")
            self._update_analysis()

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the shift value.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying the shift: {e}")

    def _apply_logarithm(self):
        if self.processed_data is None:
            QMessageBox.warning(self, "No Data", "Please load the data first.")
            return

        current_data_array = np.array(self.processed_data)

        if np.any(current_data_array <= 0):
            reply = QMessageBox.warning(self, "Invalid Data for Logarithm",
                                        "The data contains non-positive values (<= 0).\n"
                                        "Logarithm requires strictly positive values (> 0).\n\n"
                                        "It is recommended to apply a positive shift first.",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                                        QMessageBox.StandardButton.Ok)
            return

        try:
            print(type(self.processed_data), len(self.processed_data))

            self.processed_data = logarithmize_data(self.processed_data)
            self.modifications_log.append("Natural logarithm (ln) applied.")
            QMessageBox.information(self, "Logarithm Applied", "Natural logarithm applied.\nUpdating analysis...")
            self._update_analysis()

        except ValueError as e:
            QMessageBox.critical(self, "Logarithm Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying the logarithm: {e}")

    def _standartise_data(self):
        if self.processed_data is None:
            QMessageBox.warning(self, "No Data", "Please load the data first.")
            return
        
        try:
            self.processed_data = standartise_data(self.processed_data)
            self.modifications_log.append("Standardization applied.")
            QMessageBox.information(self, "Standardization Applied", 
                                   "Standardization applied.\nUpdating analysis...")
            self._update_analysis()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while standardizing the data: {e}")
    
    def _generate_distribution(self):
        try:
            n_str = self.size_line.text().strip()
            if not n_str or int(n_str) == 0:
                QMessageBox.warning(self, "Generating Error", "You must enter n")
            else:
                n = int(n_str)
        except:
            QMessageBox.warning(self, "Invalid Input","Please enter a valid number for dist generation")

        if self.current_dist == "Exponential":
            lam = float(self.lambda_line.text().strip())
            dist_array = generate_exp_theoretical_dist(n, lam)
        elif self.current_dist == "Uniform":
            a = float(self.a_line.text().strip())
            b = float(self.b_line.text().strip()) 
            dist_array = generate_uniform_theoretical_dist(n, a, b)
        elif self.current_dist == "Weibull":
            alpha = float(self.alpha_line.text().strip())
            beta = float(self.beta_line.text().strip())
            dist_array = generate_weibull_theoretical_dist(n, alpha, beta)
        elif self.current_dist == "Normal":
            mean = float(self.mean_line.text().strip())
            std = float(self.std_line.text().strip())
            dist_array = generate_normal_box_muller_distribution(n, mean, std)
        elif self.current_dist == "Laplace":
            lam = float(self.lam2_line.text().strip())
            mean = float(self.mean2_line.text().strip())
            dist_array = generate_laplace(mu=mean, b=lam, size=n)

            

        if dist_array is not None:
            # Open file save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Distribution Data",
                "",  # Default directory
                "Text Files (*.txt);;All Files (*)"
            )
            if file_path:  # If user didn't cancel the dialog
                try:
                    with open(file_path, 'w') as file:
                        # Convert all numbers to strings and join with spaces
                        file.write(' '.join(map(str, dist_array)))
                    QMessageBox.information(self, "Success", f"Distribution saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
        
    
    def _update_analysis(self):
        if self.processed_data is None:
            self._clear_plots_and_stats()
            return

        current_data = self.processed_data.copy()
        temp_modifications = []
        try:
            if self.rd_del_outliers.isChecked():
                try:
                    quantile_str = self.quantile_entry.text().strip().replace(',', '.')
                    quantile_val = float(quantile_str) if quantile_str else 0.1
                    if not (0.0 <= quantile_val < 0.5):
                        QMessageBox.warning(self, "Invalid Input",
                                            f"Trimming fraction ({quantile_val}) must be between 0.0 and 0.5. Using 0.1.")
                        quantile_val = 0.1
                        self.quantile_entry.setText(str(quantile_val))

                    original_len = len(current_data)
                    current_data = trim_data(current_data, quantile_val)
                    removed = original_len - len(current_data)
                    if removed > 0:
                        temp_modifications.append(f"Trimmed {removed} points ({quantile_val * 100:.1f}% on each side).")
                    if len(current_data) == 0:
                        QMessageBox.warning(self, "Processing Error", "All data was removed after trimming.")
                        self._clear_plots_and_stats()
                        self.statistics_output.setPlainText("No data after trimming.")
                        return

                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        "Invalid value for trimming fraction. Enter a number between 0.0 and 0.5.")
                    return

            if self.rd_del_anomalies.isChecked():
                try:
                    z_threshold_str = self.z_score_entry.text().strip().replace(',', '.')
                    z_threshold = abs(float(z_threshold_str)) if z_threshold_str else 1.96
                    if z_threshold <= 0:
                        QMessageBox.warning(self, "Invalid Input",
                                            f"Z-score threshold ({z_threshold}) must be > 0. Using 1.96.")
                        z_threshold = 1.96
                        self.z_score_entry.setText(str(z_threshold))

                    if len(current_data) > 1:
                        mean_val = arithmetic_mean(current_data)
                        unbiased_var = unbiased_sample_variance(current_data, mean_val)
                        unbiased_std = np.sqrt(unbiased_var) if unbiased_var >= 0 else 0

                        if unbiased_std is not np.nan and unbiased_std > 0:
                            original_len = len(current_data)
                            current_data = del_anomaly_data_Z_score(current_data, mean_val, unbiased_std, -z_threshold,
                                                                    z_threshold)
                            removed = original_len - len(current_data)
                            if removed > 0:
                                temp_modifications.append(f"Removed {removed} points with |Z| > {z_threshold:.2f}.")
                            if not current_data:
                                QMessageBox.warning(self, "Processing Error",
                                                    "All data was removed after Z-score filtering.")
                                self._clear_plots_and_stats()
                                self.statistics_output.setPlainText("No data after Z-score filtering.")
                                return
                        elif unbiased_std == 0:
                            temp_modifications.append("Skipped Z-filtering (standard deviation is 0).")
                        else:
                            temp_modifications.append("Skipped Z-filtering (unable to calculate standard deviation).")

                    elif len(current_data) <= 1:
                        temp_modifications.append("Skipped Z-filtering (too few data points).")

                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        "Invalid value for Z-score threshold. Enter a number > 0.")
                    return
                    
            if self.rd_del_anomalies_kurtosis.isChecked():
                try:
                    mean_val = arithmetic_mean(current_data)
                    unbiased_var = unbiased_sample_variance(current_data, mean_val)
                    unbiased_std = np.sqrt(unbiased_var) if unbiased_var >= 0 else 0

                    biased_var = biased_sample_variance(current_data, mean_val)
                    biased_std = np.sqrt(biased_var)

                    biased_asym = biased_asymmetry(current_data, mean_val, biased_std)
                    unbiased_asym = unbiased_asymmetry(len(current_data), biased_asym)

                    biased_kurt = biased_kurtosis(current_data, mean_val, biased_std)
                    unbiased_kurt = unbiased_kurtosis(len(current_data), biased_kurt)

                    original_len = len(current_data)
                    current_data = anomaly_deletion_by_unbiased_kurtosis(current_data, unbiased_std, unbiased_kurt, unbiased_asym, mean_val)
                    removed = original_len - len(current_data)
                    if removed > 0:
                        temp_modifications.append(f"Removed {removed} anomalies using kurtosis and asymmetry criteria.")
                except Exception as e:
                    temp_modifications.append(f"Error applying kurtosis anomaly detection: {str(e)}")

            if not current_data or len(current_data) == 0:
                if not temp_modifications:
                    QMessageBox.warning(self, "No Data", "No data available for analysis.")
                self._clear_plots_and_stats()
                self.statistics_output.setPlainText("No data available for analysis after preprocessing.")
                return

            bins_amount = 0
            if self.rd_custom.isChecked():
                try:
                    bins_str = self.custom_entry.text().strip()
                    if bins_str:
                        bins_amount = int(bins_str)
                        if bins_amount <= 0:
                            raise ValueError("Number of intervals must be positive.")
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input",
                                        f"Invalid number of intervals: {e}. Choosing automatically.")
                    bins_amount = 0
                    self.rd_default.setChecked(True)

            hist_info = midpoint_intervals_forming(current_data, bins=bins_amount)
            self.intervals_array = hist_info['intervals_array']
            delta_h = float(hist_info['delta_h'])
            bins_amount = hist_info["bins_amount"]

            frequencies_array = frequencies(current_data, delta_h, bins_amount)
            relative_frequencies_array = relative_frequencies(frequencies_array, len(current_data))
            x_axis_ecdf, y_axis_ecdf = ecdf(current_data)

            print(f"intervals_array size: {len(self.intervals_array)}")
            print(f"relative_frequencies_array size: {len(relative_frequencies_array)}")
            print(f"frequencies_array size: {len(frequencies_array)}")
            assert len(self.intervals_array) == len(relative_frequencies_array), "Array size mismatch!"

            self._plot_histogram(self.intervals_array, relative_frequencies_array, delta_h)
            self._plot_ecdf(x_axis_ecdf, y_axis_ecdf)
                # overlay the binned (step) CDF
            self._plot_discrete_ecdf(
                self.intervals_array,
                relative_frequencies_array,
                delta_h
            )

            if self.ExpHelpWindow is not None and self.ExpHelpWindow.isVisible():
                #self._update_exp_prob_paper(x_axis_ecdf, y_axis_ecdf)
                self.plot_qq_exponential(self.processed_data)
            
            # Use the display_statistics_table method 
            self._display_statistics(
                data=current_data,
                current_modifications=temp_modifications,
                intervals_array =self.intervals_array,
                delta_h=delta_h
            )

        except ValueError as e:
            QMessageBox.critical(self, "Processing Error", f"Error during analysis: {str(e)}")
            self._clear_plots_and_stats()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")
            self._clear_plots_and_stats()

    def _plot_histogram(self, intervals_array, relative_frequencies_array, delta_h):
        self.histogram_widget.clear()
        if not intervals_array or not relative_frequencies_array:
            self.histogram_widget.setTitle("Histogram of Relative Frequencies (no data)")
            return

        bar_width = delta_h
        if bar_width <= 0: bar_width = 1.0

        histogram_item = pg.BarGraphItem(
            x=intervals_array,
            height=relative_frequencies_array,
            width=bar_width,
            brush='cornflowerblue',
            pen=pg.mkPen('k', width=0.2)
        )
        self.histogram_widget.addItem(histogram_item)
         #Set y-axis range based on max bin height
        max_height = max(relative_frequencies_array) if relative_frequencies_array else 1.0
        self.histogram_widget.setYRange(0, max_height * 1.1)  # Add 10% padding on top
        
        # Set x-axis range if needed
        if intervals_array:
            min_x = min(intervals_array) - bar_width/2
            max_x = max(intervals_array) + bar_width/2
            self.histogram_widget.setXRange(min_x, max_x)


    def _plot_ecdf(self, x_axis, y_axis):
        self.ecdf_widget.clear()
        if not x_axis or not y_axis:
            self.ecdf_widget.setTitle("ECDF (no data)")
            return
        ecdf_item = pg.PlotCurveItem(
            x=x_axis,
            y=y_axis,
            pen=pg.mkPen('blue', width=0.5)
        )
        self.ecdf_widget.addItem(ecdf_item)

        scatter = pg.ScatterPlotItem(x=x_axis, y=y_axis, size=2, brush='blue')
        self.ecdf_widget.addItem(scatter)

        self.ecdf_widget.autoRange()


    def _plot_discrete_ecdf(self, intervals_array, relative_frequencies_array, delta_h):
        """
        Draw only the horizontal steps of the binned CDF,
        as separate line segments (no vertical connectors).
        """
        # define your custom pen once
        my_pen = pg.mkPen(color=(0, 150, 0),    # dark green
                  width=4,             # 4px thick
                  style=QtCore.Qt.PenStyle.DashLine)
        RED_PEN = pg.mkPen('r', width=2)   # solid red, 2px thick          
        if not intervals_array or not relative_frequencies_array:
            return

        # build bin edges from midpoints
        edges = [mid - delta_h/2 for mid in intervals_array]
        edges.append(intervals_array[-1] + delta_h/2)

        # cumulative relative frequencies
        cum_rel = np.cumsum(relative_frequencies_array)

        # for each bin, draw one horizontal segment at height=cum_rel[i]
        for i, level in enumerate(cum_rel):
            left = edges[i]
            right = edges[i+1]
            seg = pg.PlotCurveItem(
                x=[left, right],
                y=[level, level],
                pen=RED_PEN
            )
            self.ecdf_widget.addItem(seg)

    def plot_qq_exponential(self, data):
        if self.ExpHelpWindow is None or not data:
            return
        w = self.ExpHelpWindow.exp_porb_widget
        w.clear()
        
        x = sorted([float(item) for item in data])
        n = len(x)
        # Емпіричні ймовірності
        p_emp = []
        for i in range(1, n + 1):
            prob = (i - 0.5) / n
            p_emp.append(min(prob, 0.99))
        phi = []
        for p in p_emp:
            phi.append(math.log(1 / (1 - p)))
        
        scatter = pg.ScatterPlotItem(x=x, y=phi, size=5, brush=pg.mkBrush('b'))
        w.addItem(scatter)

        #ЛІНІЙНА РЕГРЕСІЯ ФОРМУЛИ
        sum_x = sum(x)
        sum_y = sum(phi)
        sum_x_squared = sum(x_val * x_val for x_val in x)
        sum_xy = sum(x_val * y_val for x_val, y_val in zip(x, phi))
        # a, b
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        b = (sum_y - a * sum_x) / n
        min_x = min(x)
        max_x = max(x)
        line_x = []
        line_y = []
        step = (max_x - min_x) / 99 
        for i in range(100):
            x_val = min_x + step * i
            line_x.append(x_val)
            line_y.append(b + a * x_val)
        line = pg.PlotCurveItem(x=line_x, y=line_y, pen=pg.mkPen('r', width=2))
        w.addItem(line)
        
        # значення p
        tick_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        tick_positions = []
        for p in tick_probs:
            if p < 1:
                tick_positions.append(math.log(1 / (1 - p)))
        
        yticks = [(val, f"{p:.2f}") for val, p in zip(tick_positions, tick_probs)]
        w.getPlotItem().getAxis('left').setTicks([yticks])
        
        max_phi = math.log(1 / (1 - 0.99))
        w.setYRange(0, max_phi)
        
        w.setXRange(min(x), max(x))
        
        w.setLabels(bottom='x', left='Ймовірність p')
        w.showGrid(x=True, y=True, alpha=0.3)

    def _clear_plots_and_stats(self):
        """Clears all plots and statistics text area."""
        self.histogram_widget.clear()
        self.ecdf_widget.clear()
        self.statistics_output.clear()

    def _display_statistics(self, data, current_modifications=None, confidence_level=0.95, intervals_array=None, delta_h=0, bins_count=0):
        """Displays statistical metrics in a formatted table.""" 
        self.statistics_output.clear()
        try:
            if len(data) == 0:
                self.statistics_output.setPlainText("No data for analysis.")
                return
            
            output_lines = []
            confidence_level = 0.95

            # Calculate all statistics
            length = len(data)
            mean = arithmetic_mean(data)
            median = sample_median(data)
            trim_mean_10 = trimmed_mean(data, a=0.1)
            unbiased_var = unbiased_sample_variance(data, mean)
            unbiased_std = np.sqrt(unbiased_var)

            biased_var = biased_sample_variance(data, mean)
            biased_std = np.sqrt(biased_var)

            biased_asym = biased_asymmetry(data, mean, biased_std)
            unbiased_asym = unbiased_asymmetry(length, biased_asym)

            biased_kurt = biased_kurtosis(data, mean, biased_std)
            unbiased_exc_kurt = unbiased_kurtosis(length, biased_kurt)

            count_kurt = counter_kurtosis(unbiased_exc_kurt)
            if mean > 10**(-3):
                pearson_cv = pirson_coeff(unbiased_std, mean)
            #MAD/MED 
            non_parametric_coef_of_variation = median_absolute_deviation(data)/median
            
            # Add Walsh median calculation if intervals_array and delta_h are provided
            walsh_med = walsh_median(data, intervals_array, delta_h)
            # Calculate intervals and standard errors
            intervals_dict = calculate_all_intervals(data)
            
            # Create table header
            table_lines = []
            table_lines.append(f"--- Data Summary ---")
            table_lines.append(f"Sample size (after processing): n = {length}")
            header = f"| {'Parameter Estimate':<20} | {'Standard Error':<15} | {'Lower CI Bound':<15} | {'Value':<12} | {'Upper CI Bound':<15} |"
            separator = f"|{'-'*22}|{'-'*17}|{'-'*17}|{'-'*14}|{'-'*17}|"
            table_lines.append(separator)
            table_lines.append(header)
            table_lines.append(separator)

            # Add rows for each statistic
            # Mean
            se_mean = intervals_dict["Standard Errors"]["SE Mean"]
            ci_mean = intervals_dict["Confidence Intervals"]["CI Mean"]
            table_lines.append(f"| {'Mean':<20} | {se_mean:15.6f} | {ci_mean[0]:15.6f} | {mean:12.6f} | {ci_mean[1]:15.6f} |")
            # Variance
            se_var = intervals_dict["Standard Errors"]["SE Variance"]
            ci_var = intervals_dict["Confidence Intervals"]["CI Variance"]
            #table_lines.append(f"| {'Unbiased Variance':<20} | {se_var:15.6f} | {ci_var[0]:15.6f} | {unbiased_var:12.6f} | {ci_var[1]:15.6f} |")
            # Standard Deviation
            ci_std = intervals_dict["Confidence Intervals"]["CI Std Dev"]
            table_lines.append(f"| {'Unbiased Std Dev':<20} | {se_var**(1/2):15.6f} | {ci_std[0]:15.6f} | {unbiased_std:12.6f} | {ci_std[1]:15.6f} |")

            # Skewness
            se_asym = intervals_dict["Standard Errors"]["SE Skewness"]
            ci_asym = intervals_dict["Confidence Intervals"]["CI Skewness"]
            table_lines.append(f"| {'Unbiased Assymetry':<20} | {se_asym:15.6f} | {ci_asym[0]:15.6f} | {unbiased_asym:12.6f} | {ci_asym[1]:15.6f} |")
            # Kurtosis
            se_kurt = intervals_dict["Standard Errors"]["SE Excess Kurtosis"]
            ci_kurt = intervals_dict["Confidence Intervals"]["CI Excess Kurtosis"]
            table_lines.append(f"| {'Unbiased Excess Kurt':<20} | {se_kurt:15.6f} | {ci_kurt[0]:15.6f} | {unbiased_exc_kurt:12.6f} | {ci_kurt[1]:15.6f} |")
            # Median (no SE or CI in the calculation)
            table_lines.append(f"| {'Median':<20} | {'':<15} | {'':<15} | {median:12.6f} | {'':<15} |")
            # Trimmed Mean (no SE or CI in the calculation)
            table_lines.append(f"| {'Trimmed Mean (10%)':<20} | {'':<15} | {'':<15} | {trim_mean_10:12.6f} | {'':<15} |")

            #Walsh Median
            table_lines.append(f"| {'Walsh Median':<20} | {'':<15} | {'':<15} | {walsh_med:12.6f} | {'':<15} |")
            #Counter Excess
            table_lines.append(f"| {'Counter Excess':<20} | {'':<15} | {'':<15} | {count_kurt:12.6f} | {'':<15} |")

            if mean > 10**(-3):
                #Pearson cv
                table_lines.append(f"| {'Pearson cv':<20} | {'':<15} | {'':<15} | {pearson_cv:12.6f} | {'':<15} |")
            #non_parametric_coef_of_variation
            table_lines.append(f"| {'MAD/MED':<20} | {'':<15} | {'':<15} | {non_parametric_coef_of_variation:12.6f} | {'':<15} |")
            # Add footer
            table_lines.append(separator)
            
            # Add prediction intervals in a separate section
            pi_single = intervals_dict["Prediction Intervals"]["PI Single Observation"]
            
            table_lines.append(f"\nPrediction Intervals (Confidence Level: {confidence_level*100:.0f}%):")
            table_lines.append(f"Single Observation: ({pi_single[0]:.6f}, {pi_single[1]:.6f})")
            
            table_lines.append(f"\nBiased statistics (just for fun)")
            output_lines.append(f"{'Biased Variance:':<30} {biased_var: 12.6f}")
            output_lines.append(f"{'Biased Std Dev:':<30} {biased_std: 12.6f}")
            output_lines.append(f"{'Biased Asymmetry:':<30} {biased_asym: 12.6f}")

            output_lines.append(f"{'Biased Excess:':<30} {biased_kurt: 12.6f}")
            # Display the table
            self.statistics_output.append("\n".join(table_lines))
            self.statistics_output.append("\n".join(output_lines))

        except Exception as e:
            import traceback
            print("Error displaying statistics table:")
            print(traceback.format_exc())
            self.statistics_output.append(f"Error displaying statistics table:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    styles_main = ("""
        QMainWindow, QWidget {
            background-color: #f0f0f0; /* Светло-серый фон */
        }
        QLabel, QRadioButton, QGroupBox {
            font-size: 10pt; /* Чуть увеличим шрифт */
        }
        QLineEdit, QTextEdit {
            font-size: 10pt;
            padding: 4px;
            border: 1px solid #bdc3c7; /* Серая рамка */
            border-radius: 3px;
            background-color: #ffffff; /* Белый фон для полей */
        }
        QPushButton {
            background-color: #3498db; /* Синий */
            color: white;
            font-size: 10pt;
            font-weight: bold;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            min-width: 80px; /* Минимальная ширина кнопки */
        }
        QPushButton:hover {
            background-color: #2980b9; /* Темно-синий при наведении */
        }
        QPushButton:disabled {
            background-color: #bdc3c7; /* Серый для неактивных кнопок */
            color: #7f8c8d;
        }
        QRadioButton {
            padding: 4px;
        }
        QGroupBox {
            margin-top: 10px; /* Отступ сверху для групп */
            padding: 10px;
             border: 1px solid #bdc3c7;
             border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left; /* Position at the top left */
            padding: 0 3px 0 3px;
            left: 10px; /* Move title slightly to the right */
            color: #2c3e50; /* Darker title color */
            font-weight: bold;
        }
        QTabWidget::pane { /* The tab widget frame */
            border-top: 2px solid #bdc3c7;
            margin-top: -2px; /* Align pane border with tab bottom border */
        }
        QTabBar::tab { /* The tab titles */
            background: #e4e4e4; /* Lighter gray for inactive tabs */
            border: 1px solid #bdc3c7;
            border-bottom-color: #bdc3c7; /* Same as top border */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 100px; /* Min width for tabs */
            padding: 5px 10px;
            margin-right: 2px; /* Space between tabs */
        }
        QTabBar::tab:selected {
            background: #f0f0f0; /* Background matches window */
            border-color: #bdc3c7;
            border-bottom-color: #f0f0f0; /* Make bottom border match background */
            margin-bottom: -1px; /* Raise selected tab slightly */
        }
        QTabBar::tab:!selected:hover {
            background: #d4d4d4; /* Slightly darker on hover */
        }
        QTextEdit {
             font-family: "Courier New", Courier, monospace; /* Моноширинный шрифт */
             background-color: #fdfdfd; /* Чуть-чуть не белый */
        }
        QLabel#file_label { /* Если бы у file_label был objectName='file_label' */
            font-style: italic;
            color: #555;f
        }

    """)

    app.setStyle("Fusion")
    #app.setStyleSheet(styles_main)
    window = StatisticalApplication()
    window.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()