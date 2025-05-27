import sys
import numpy as np
from scipy.stats import norm, weibull_min, laplace, chi2, f, t, gaussian_kde


import math
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, QTabWidget,QTableWidget, QHeaderView, QTableWidgetItem,
                             QRadioButton, QButtonGroup, QGroupBox, QTextEdit,
                             QMessageBox, QFileDialog, QSizePolicy, QStackedWidget, QComboBox, QCheckBox, QFormLayout)
from PyQt6.QtGui import QIcon, QFontDatabase, QDoubleValidator, QIntValidator

from main_functions import *
from dist_generation import *
from criterias import *

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

        self.generate_dist_button = QPushButton("🎲 Random distributions and hypotheses test")
        self.generate_dist_button.setCheckable(True)

        self.homogeneity_button = QPushButton("⚖️ Homogeneity tests")
        self.homogeneity_button.setCheckable(True)

        self.reproduction_button = QPushButton("🧪 Experiments on distributions")
        self.reproduction_button.setCheckable(True)


        # Create button group for exclusive selection
        self.view_button_group = QButtonGroup()
        self.view_button_group.addButton(self.analysis_button, 1)
        self.view_button_group.addButton(self.transform_button, 2) 
        self.view_button_group.addButton(self.generate_dist_button, 3)
        self.view_button_group.addButton(self.homogeneity_button, 4)
        self.view_button_group.addButton(self.reproduction_button, 5)


        mode_buttons_layout.addWidget(self.analysis_button) 
        mode_buttons_layout.addWidget(self.transform_button)     
        mode_buttons_layout.addWidget(self.generate_dist_button)
        mode_buttons_layout.addWidget(self.homogeneity_button)
        mode_buttons_layout.addWidget(self.reproduction_button)


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
        #TODO self.statistics_output.setMaximumHeight(250)
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
        self.size_line = QLineEdit("5000")

        dist_chooser_layout.addWidget(self.generate_distribution_push_button)
        dist_chooser_layout.addWidget(self.falling_list)
        dist_chooser_layout.addWidget(size_label)
        dist_chooser_layout.addWidget(self.size_line)

        #params 
        self.params_layout = QHBoxLayout()

        #Exp
        self.lambda_label = QLabel("Lambda: ")
        self.lambda_line = QLineEdit("6")
        self.params_layout.addWidget(self.lambda_label)
        self.params_layout.addWidget(self.lambda_line)
        self.lambda_label.hide()
        self.lambda_line.hide()

        #Uniform
        self.a_label = QLabel("a: ")
        self.b_label = QLabel("b: ")
        self.a_line = QLineEdit("5")
        self.b_line = QLineEdit("3")

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
        self.lam2_line = QLineEdit("30")
        self.mean2_line = QLineEdit("0")

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
        self.alpha_line = QLineEdit("1.5")
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

        self.homogeneity_panel = QWidget()
        homogeneity_layout = QHBoxLayout(self.homogeneity_panel)

        # --- Left side: Controls/Inputs ---
        self.choosing_layout = QVBoxLayout()

        # Section: Choose two distributions
        two_distros_label = QLabel("Choose two arbitrary distributions:")
        self.choosing_layout.addWidget(two_distros_label)

        self.choose_1_dist = QPushButton("FIRST")
        self.file_dist_status_1 = QLabel("(file isn't loaded yet)")
        dist1_layout = QHBoxLayout()
        dist1_layout.addWidget(self.choose_1_dist)
        dist1_layout.addWidget(self.file_dist_status_1)

        self.choose_2_dist = QPushButton("SECOND")
        self.file_dist_status_2 = QLabel("(file isn't loaded yet)")
        dist2_layout = QHBoxLayout()
        dist2_layout.addWidget(self.choose_2_dist)
        dist2_layout.addWidget(self.file_dist_status_2)

        choose_btns_layout = QVBoxLayout()
        choose_btns_layout.addLayout(dist1_layout)
        choose_btns_layout.addLayout(dist2_layout)

        self.choosing_layout.addLayout(choose_btns_layout)

        # Section: Two-sample analysis options
        two_sample_section_label = QLabel("Two-sample options:")
        #note_1 = QLabel("Note: Suitable for normal-like distributions.")
        self.choosing_layout.addWidget(two_sample_section_label)
        #self.choosing_layout.addWidget(note_1)

        two_sample_checkboxes = QVBoxLayout()
        self.two_disp_mean_compare = QCheckBox("Compare dispersions and means. !Note: Suitable for normal-like distributions only!")
        self.two_disp_mean_criterias = QCheckBox("Perform criterias")
        self.perform_button_one = QPushButton("Perform")
        self.perform_button_one.setEnabled(False)
        two_sample_checkboxes.addWidget(self.two_disp_mean_compare)
        two_sample_checkboxes.addWidget(self.two_disp_mean_criterias)
        self.choosing_layout.addLayout(two_sample_checkboxes)
        self.choosing_layout.addWidget(self.perform_button_one)

        # Section: Many-sample analysis options
        many_sample_section_label = QLabel("Many-sample criterias:")
        note_2 = QLabel("Note: Parametric tests need normal-like distributions.")
        self.choosing_layout.addWidget(many_sample_section_label)
        self.choosing_layout.addWidget(note_2)

        many_sample_checkboxes = QVBoxLayout()
        self.many_samples_normal_criteria = QCheckBox("Normal distribution criteria")
        self.many_samples_non_normal_criteria = QCheckBox("Non-normal distribution criteria")
        self.perform_button_two = QPushButton("Perform")

        many_sample_checkboxes.addWidget(self.many_samples_normal_criteria)
        many_sample_checkboxes.addWidget(self.many_samples_non_normal_criteria)
        self.choosing_layout.addLayout(many_sample_checkboxes)
        self.choosing_layout.addWidget(self.perform_button_two)


        # Add choosing layout to the main homogeneity layout (left side)
        homogeneity_layout.addLayout(self.choosing_layout, stretch=1)


        # --- Right side: Output display ---
        self.output_dist_layout = QVBoxLayout()

        self.output_1_dist_text = QTextEdit()
        self.output_1_dist_text.setReadOnly(True)
        self.output_dist_layout.addWidget(self.output_1_dist_text)

        self.output_2_dist_text = QTextEdit()
        self.output_2_dist_text.setReadOnly(True)
        self.output_dist_layout.addWidget(self.output_2_dist_text)

        # Add output layout to the main homogeneity layout (right side)
        homogeneity_layout.addLayout(self.output_dist_layout, stretch=2)


        # --- Add to stacked widget ---
        self.stacked_widget.addWidget(self.analysis_panel)
        self.stacked_widget.addWidget(self.transform_panel)
        self.stacked_widget.addWidget(self.generation_panel)
        self.stacked_widget.addWidget(self.homogeneity_panel)

        # --- Connect buttons to switch views ---
        self.analysis_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.transform_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.generate_dist_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.homogeneity_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        self.reproduction_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

        self.reproduction_panel = QWidget()
        self.reproduction_layout = QVBoxLayout(self.reproduction_panel)
        self.reproduction_layout.addWidget(QLabel("Choose distribution type, size of it and specify the parameters to perform a simulation"))
        # --- Top Layout: all inline ---
        top_layout = QHBoxLayout()

        # Distribution selector
        self.falling_list_1 = QComboBox()
        self.falling_list_1.addItems(["Exponential", "Normal", "Weibull", "Uniform", "Laplace"])
        top_layout.addWidget(QLabel("Distribution:"))
        top_layout.addWidget(self.falling_list_1)

        # Size
        self.size_label_1 = QLabel("Size n:")
        self.size_line_1 = QLineEdit("5000")
        self.size_line_1.setFixedWidth(60)
        top_layout.addWidget(self.size_label_1)
        top_layout.addWidget(self.size_line_1)

        # --- Parameters (compact) ---
        self.params_form_layout_1 = QFormLayout()
        self.params_form_layout_1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Exponential
        self.lambda_label_1 = QLabel("Lambda:")
        self.lambda_line_1 = QLineEdit("6")
        self.params_form_layout_1.addRow(self.lambda_label_1, self.lambda_line_1)

        # Uniform
        self.a_label_1 = QLabel("a:")
        self.a_line_1 = QLineEdit("5")
        self.b_label_1 = QLabel("b:")
        self.b_line_1 = QLineEdit("3")
        self.params_form_layout_1.addRow(self.a_label_1, self.a_line_1)
        self.params_form_layout_1.addRow(self.b_label_1, self.b_line_1)

        # Laplace
        self.lam2_label_1 = QLabel("lambda:")
        self.lam2_line_1 = QLineEdit("20")
        self.mean2_label_1 = QLabel("mean:")
        self.mean2_line_1 = QLineEdit("0")
        self.params_form_layout_1.addRow(self.lam2_label_1, self.lam2_line_1)
        self.params_form_layout_1.addRow(self.mean2_label_1, self.mean2_line_1)

        # Weibull
        self.alpha_label_1 = QLabel("alpha:")
        self.alpha_line_1 = QLineEdit("1")
        self.beta_label_1 = QLabel("beta:")
        self.beta_line_1 = QLineEdit("1")
        self.params_form_layout_1.addRow(self.alpha_label_1, self.alpha_line_1)
        self.params_form_layout_1.addRow(self.beta_label_1, self.beta_line_1)

        # Normal
        self.mean_label_1 = QLabel("mean:")
        self.mean_line_1 = QLineEdit("0")
        self.std_label_1 = QLabel("std:")
        self.std_line_1 = QLineEdit("1")
        self.params_form_layout_1.addRow(self.mean_label_1, self.mean_line_1)
        self.params_form_layout_1.addRow(self.std_label_1, self.std_line_1)

        # Hide all by default
        for w in [
            self.lambda_label_1, self.lambda_line_1,
            self.a_label_1, self.a_line_1, self.b_label_1, self.b_line_1,
            self.lam2_label_1, self.lam2_line_1, self.mean2_label_1, self.mean2_line_1,
            self.alpha_label_1, self.alpha_line_1, self.beta_label_1, self.beta_line_1,
            self.mean_label_1, self.mean_line_1, self.std_label_1, self.std_line_1
        ]:
            w.hide()

        # Put form into a container
        param_container = QWidget()
        param_container.setLayout(self.params_form_layout_1)
        top_layout.addWidget(param_container)

        # Button
        self.generate_distribution_push_button_1 = QPushButton("Perform a simulation")
        top_layout.addWidget(self.generate_distribution_push_button_1)

        self.reproduction_layout.addLayout(top_layout)
        self.shit = QHBoxLayout()
        output_simul_label = QLabel("Parameters estimation and evaluation of their error ")
        self.alpha_value_line = QLineEdit("0.05")
        self.alpha_value_line.setMaximumWidth(60)
        self.shit.addWidget(output_simul_label)
        self.shit.addWidget(self.alpha_value_line)

        self.simul_output = QTextEdit()
        self.simul_output.setReadOnly(True)
        self.simul_output.setMaximumHeight(70)
        font_1 = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.simul_output.setFont(font_1)
        self.reproduction_layout.addLayout(self.shit)
        self.reproduction_layout.addWidget(self.simul_output)

        self.kolm_pearson_output = QTextEdit()
        self.kolm_pearson_output.setReadOnly(True)
        self.kolm_pearson_output.setMaximumHeight(200)
        self.kolm_pearson_output.setFont(font_1)
        self.reproduction_layout.addWidget(self.kolm_pearson_output)





        # --- Add to stacked widget ---
        self.stacked_widget.addWidget(self.analysis_panel)
        self.stacked_widget.addWidget(self.transform_panel)
        self.stacked_widget.addWidget(self.generation_panel)
        self.stacked_widget.addWidget(self.homogeneity_panel)
        self.stacked_widget.addWidget(self.reproduction_panel)

        # --- Connect buttons to switch views ---
        self.analysis_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.transform_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.generate_dist_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.homogeneity_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        self.reproduction_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

    def _connect_signals(self):
        self.file_button.clicked.connect(lambda: self._load_data(external_data=None))  

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
        self.generate_distribution_push_button_1.clicked.connect(self._perform_lab2)
        self.reset_data_button.clicked.connect(self._reset_data)
        # lab4 
        #self.perform_button_one.clicked.connect(lambda: self._load_distr(2))
        self.choose_1_dist.clicked.connect(lambda: self._load_distr(1))
        self.choose_2_dist.clicked.connect(lambda: self._load_distr(2))
        self.perform_button_one.clicked.connect(self._perform_action_one)

        self.falling_list.currentTextChanged.connect(self._update_ui_state)
        self.falling_list_1.currentTextChanged.connect(self._update_ui_state)


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
        
        self.current_dist_1 = self.falling_list_1.currentText()

        # Hide all param widgets
        self.lambda_label_1.hide()
        self.lambda_line_1.hide()

        self.a_label_1.hide()
        self.a_line_1.hide()
        self.b_label_1.hide()
        self.b_line_1.hide()

        self.alpha_label_1.hide()
        self.alpha_line_1.hide()
        self.beta_label_1.hide()
        self.beta_line_1.hide()

        self.mean_label_1.hide()
        self.mean_line_1.hide()
        self.std_label_1.hide()
        self.std_line_1.hide()

        self.lam2_label_1.hide()
        self.lam2_line_1.hide()
        self.mean2_label_1.hide()
        self.mean2_line_1.hide()

        # Show relevant param widgets
        if self.current_dist_1 == "Exponential":
            self.lambda_label_1.show()
            self.lambda_line_1.show()
        elif self.current_dist_1 == "Uniform":
            self.a_label_1.show()
            self.a_line_1.show()
            self.b_label_1.show()
            self.b_line_1.show()
        elif self.current_dist_1 == "Weibull":
            self.alpha_label_1.show()
            self.alpha_line_1.show()
            self.beta_label_1.show()
            self.beta_line_1.show()
        elif self.current_dist_1 == "Normal":
            self.mean_label_1.show()
            self.mean_line_1.show()
            self.std_label_1.show()
            self.std_line_1.show()
        elif self.current_dist_1 == "Laplace":
            self.lam2_label_1.show()
            self.lam2_line_1.show()
            self.mean2_label_1.show()
            self.mean2_line_1.show()

    def _load_data(self, external_data=None):
        print(f"DEBUG: _load_data called with external_data={external_data}, type={type(external_data)}")
        if external_data is not None:
            try:
                # Make sure external_data is a list or array that supports copy
                if isinstance(external_data, (list, tuple)):
                    # Convert to list if it's not already
                    self.raw_dist_data = list(external_data)
                    # Use list slicing to copy a list
                    self.processed_data = self.raw_dist_data[:]
                elif hasattr(external_data, 'copy'):
                    # For NumPy arrays or other objects with copy method
                    self.raw_dist_data = external_data
                    self.processed_data = self.raw_dist_data.copy()
                else:
                    # Handle unexpected types
                    raise TypeError(f"Cannot load data of type {type(external_data).__name__}. Expected list, tuple, or array.")
                
                self.current_file_path = "Internal: Simulated Data"
                self.file_label.setText("Loaded: Simulated Data")
                self.file_label.setToolTip("Generated from Lab 2")
                self.modifications_log = ["Data loaded from simulation."]
                self._update_ui_state()
                self._update_analysis()
            except Exception as e:
                QMessageBox.critical(self, "Simulation Error", f"Failed to load simulation data:\n{str(e)}")
                self.raw_dist_data = None
                self.processed_data = None
                self.file_label.setText("Simulation load error.")
                self.file_label.setToolTip("")
                self.modifications_log = []
                self._clear_plots_and_stats()
                self._update_ui_state()
                return
        else:
            # Fallback to file dialog
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
                
                # Same type checking for file-loaded data
                if isinstance(self.raw_dist_data, (list, tuple)):
                    self.processed_data = list(self.raw_dist_data)
                elif hasattr(self.raw_dist_data, 'copy'):
                    self.processed_data = self.raw_dist_data.copy()
                else:
                    raise TypeError(f"Unexpected data type: {type(self.raw_dist_data).__name__}")
                    
                self.current_file_path = file_name
                self.file_label.setText(f"Loaded: {QtCore.QFileInfo(file_name).fileName()}")
                self.file_label.setToolTip(file_name)
                self.modifications_log = ["Data loaded from file."]
                self._update_ui_state()
                self._update_analysis()
                self.stacked_widget.setCurrentIndex(0)
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
        experiment_amount = 600
        for idx, sample_size in enumerate(sample_sizes_int):
                if significance_level <= 0:
                    alpha = alpha_values[idx]
                else:
                    alpha = significance_level
                estimated_lambdas     = []
                estimated_t_statistics = []
                for _ in range(experiment_amount):
                    simulated_exp_distr = generate_exp_theoretical_dist(sample_size, true_param_value)
                    sample_mean = arithmetic_mean(simulated_exp_distr)
                    sample_std  = math.sqrt(unbiased_sample_variance(simulated_exp_distr, sample_mean))

                    estimated_lambda = 1.0 / sample_mean
                    estimated_lambdas.append(estimated_lambda)
                    # H₀: μ = 1/λ
                    true_mean = 1.0 / true_param_value
                    #t_stat = (true_param_value - estimated_lambda) / (estimated_lambda / math.sqrt(sample_size))
                    t_stat = (sample_mean - true_mean) / (sample_std / math.sqrt(sample_size))
                    estimated_t_statistics.append(t_stat)

                mean_estimated_lambdas = arithmetic_mean(estimated_lambdas)
                std_estimated_lambdas = math.sqrt(unbiased_sample_variance(estimated_lambdas, mean_estimated_lambdas))

                mean_estimated_t = arithmetic_mean(estimated_t_statistics)
                std_estimated_t  = math.sqrt(unbiased_sample_variance(estimated_t_statistics,mean_estimated_t))

                df = sample_size - 1
                t_crit = t.ppf(1 - alpha/2, df)

                self.t_test_result_table.setItem(idx, 0, QTableWidgetItem(f"{mean_estimated_lambdas:.4f}"))
                self.t_test_result_table.setItem(idx, 1, QTableWidgetItem(f"{std_estimated_lambdas:.4f}"))
                self.t_test_result_table.setItem(idx, 2, QTableWidgetItem(f"{mean_estimated_t:.4f}"))
                self.t_test_result_table.setItem(idx, 3, QTableWidgetItem(f"{std_estimated_t:.4f}"))
                self.t_test_result_table.setItem(idx, 4, QTableWidgetItem(f"{t_crit:.4f}"))
                self.t_test_result_table.setItem(idx, 5, QTableWidgetItem(f"{alpha:.2f}"))
        
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
    def _perform_lab2(self):
        try:
            n = int(self.size_line_1.text().strip())
            if n <= 0:
                raise ValueError("n must be > 0")
        except ValueError:
            QMessageBox.warning(self, "Generating Error", "Please enter a valid n")
            return

        try:
            alpha_input = self.alpha_value_line.text().strip()
            alpha = float(alpha_input) if alpha_input else 0.05
            if not (0 < alpha < 1):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid alpha in (0, 1)")
            return

        # Set alpha based on user input and sample size
        if alpha != 0.05:  # If user specifies anything other than 0.05
            alpha_value = 0.3 if n < 30 else 0.05
        else:
            alpha_value = alpha  # Use user-specified alpha (0.05)
        z_alpha = norm.ppf(1 - alpha_value / 2)

        dist_name = self.falling_list_1.currentText()
        dist_array = None
        estimates, se_estimates, ci_estimates = {}, {}, {}

        try:
            if dist_name == "Exponential":
                lam = float(self.lambda_line_1.text().strip())
                dist_array = generate_exp_theoretical_dist(n, lam)
                lambda_hat = 1 / np.mean(dist_array)
                se = lambda_hat / np.sqrt(n)
                estimates['λ'] = lambda_hat
                se_estimates['λ'] = se
                ci_estimates['λ'] = (lambda_hat - z_alpha * se, lambda_hat + z_alpha * se)

            elif dist_name == "Uniform":
                a = float(self.a_line_1.text().strip())
                b = float(self.b_line_1.text().strip())
                dist_array = generate_uniform_theoretical_dist(n, a, b)
                a_hat, b_hat = min(dist_array), max(dist_array)
                range_est = b_hat - a_hat
                se = range_est / math.sqrt(n * (n + 1))
                estimates.update({'a': a_hat, 'b': b_hat})
                se_estimates.update({'a': se, 'b': se})
                ci_estimates['a'] = (a_hat - z_alpha * se, a_hat + z_alpha * se)
                ci_estimates['b'] = (b_hat - z_alpha * se, b_hat + z_alpha * se)

            elif dist_name == "Weibull":
                alpha_weibull = float(self.alpha_line_1.text().strip())
                beta = float(self.beta_line_1.text().strip())
                dist_array = generate_weibull_theoretical_dist(n, alpha_weibull, beta)
                alpha_hat, beta_hat = estimate_weibull_moments(dist_array)
                se_alpha, se_beta = alpha_hat / math.sqrt(n), beta_hat / math.sqrt(n)
                estimates.update({'α (shape)': alpha_hat, 'β (scale)': beta_hat})
                se_estimates.update({'α (shape)': se_alpha, 'β (scale)': se_beta})
                ci_estimates['α (shape)'] = (alpha_hat - z_alpha * se_alpha, alpha_hat + z_alpha * se_alpha)
                ci_estimates['β (scale)'] = (beta_hat - z_alpha * se_beta, beta_hat + z_alpha * se_beta)

            elif dist_name == "Normal":
                mean = float(self.mean_line_1.text().strip())
                std = float(self.std_line_1.text().strip())
                if std <= 0:
                    raise ValueError("Standard deviation must be positive")
                dist_array = generate_normal_box_muller_distribution(n, mean, std)
                mu_hat = arithmetic_mean(dist_array)
                sigma_hat = biased_sample_variance(dist_array, mu_hat)
                se_mu, se_sigma = sigma_hat / math.sqrt(n), sigma_hat * math.sqrt(2 / n)
                estimates.update({'μ': mu_hat, 'σ': sigma_hat})
                se_estimates.update({'μ': se_mu, 'σ': se_sigma})
                ci_estimates['μ'] = (mu_hat - z_alpha * se_mu, mu_hat + z_alpha * se_mu)
                ci_estimates['σ'] = (sigma_hat - z_alpha * se_sigma, sigma_hat + z_alpha * se_sigma)

            elif dist_name == "Laplace":
                lam = float(self.lam2_line_1.text().strip())
                mean = float(self.mean2_line_1.text().strip())
                dist_array = generate_laplace(mu=mean, b=lam, size=n)
                mu_hat = sample_median(dist_array)
                b_hat = np.mean([abs(x - mu_hat) for x in dist_array])
                se = b_hat / math.sqrt(n)
                estimates.update({'μ': mu_hat, 'b': b_hat})
                se_estimates.update({'μ': se, 'b': se})
                ci_estimates['μ'] = (mu_hat - z_alpha * se, mu_hat + z_alpha * se)
                ci_estimates['b'] = (b_hat - z_alpha * se, b_hat + z_alpha * se)

        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", f"Error with {dist_name} parameters: {str(e)}")
            return

        # Output: estimates + intervals
        output = "Parameters Estimation:\n"
        for param in estimates:
            output += f"{param}: {estimates[param]:.4f} (SE: {se_estimates[param]:.4f}, CI: [{ci_estimates[param][0]:.4f}, {ci_estimates[param][1]:.4f}])\n"
        self.simul_output.setPlainText(output)

        # Prepare ECDF
        self.ecdf_sorted_data = sorted(dist_array)
        self.ecdf_y_ = [i / n for i in range(1, n + 1)]
        self._load_data(external_data=dist_array)
        self._plot_ecdf_theor(dist_name, estimates, se_estimates, alpha_value, n, dist_array)
        # Pearson Chi² test
        delta_h = float(self.hist_info["delta_h"])
        minimum = min(dist_array)
        k = self.bins_amount
        bins_edges = np.linspace(minimum, minimum + k * delta_h, k + 1)
        observed_freq = np.array(self.frequencies_array, dtype=float)

        try:
            if dist_name == "Normal":
                mu, sigma = estimates['μ'], estimates['σ']
                expected_freq = n * (norm.cdf(bins_edges[1:], mu, sigma) - norm.cdf(bins_edges[:-1], mu, sigma))
            elif dist_name == "Exponential":
                lam = estimates['λ']
                expected_freq = n * (np.exp(-lam * bins_edges[:-1]) - np.exp(-lam * bins_edges[1:]))
            elif dist_name == "Uniform":
                a, b = estimates['a'], estimates['b']
                expected_freq = n * (bins_edges[1:] - bins_edges[:-1]) / (b - a)
            elif dist_name == "Weibull":
                alpha_w, beta = estimates['α (shape)'], estimates['β (scale)']
                expected_freq = n * (weibull_min.cdf(bins_edges[1:], alpha_w, scale=beta) -
                                    weibull_min.cdf(bins_edges[:-1], alpha_w, scale=beta))
            elif dist_name == "Laplace":
                mu, b = estimates['μ'], estimates['b']
                expected_freq = n * (laplace.cdf(bins_edges[1:], mu, b) - laplace.cdf(bins_edges[:-1], mu, b))
        except:
            expected_freq = np.zeros_like(observed_freq)

        # ALWAYS calculate chi2, even if expected freq < 5
        chi2_stat = np.sum((observed_freq - expected_freq)**2 / np.where(expected_freq == 0, 1e-9, expected_freq))
        df = len(observed_freq) - 1 - len(estimates)
        critical_chi2 = chi2.ppf(1 - alpha_value, df) if df > 0 else np.nan

        # Kolmogorov test
        x_sorted = np.sort(dist_array)
        if dist_name == "Normal":
            theoretical_cdf = norm.cdf(x_sorted, mu, sigma)
        elif dist_name == "Exponential":
            theoretical_cdf = 1 - np.exp(-lam * x_sorted)
        elif dist_name == "Uniform":
            a, b = estimates['a'], estimates['b']
            theoretical_cdf = np.clip((x_sorted - a) / (b - a), 0, 1)
        elif dist_name == "Weibull":
            theoretical_cdf = weibull_min.cdf(x_sorted, alpha_w, scale=beta)
        elif dist_name == "Laplace":
            theoretical_cdf = laplace.cdf(x_sorted, mu, b)

        empirical_cdf = np.arange(1, n + 1) / n
        D_n = np.max(np.abs(empirical_cdf - theoretical_cdf))
        z = np.sqrt(n) * D_n
        p_value_kol = 1 - np.exp(-2 * z**2)

        # Format test results
        results = []
        results.append("+--------------------------+---------------------+--------------------------+\n")
        results.append("| Test                     | Criteria-value      | Compare-value            |\n")
        results.append("+--------------------------+---------------------+--------------------------+\n")

        if not np.isnan(chi2_stat) and df > 0:
            results.append(f"| Pearson χ²               | {chi2_stat:>19.4f} | {critical_chi2:>24.4f} |\n")
        else:
            results.append(f"| Pearson χ²               | {'Invalid':>19} | {'-':>24} |\n")

        results.append(f"| Kolmogorov               | {p_value_kol:>19.4f} | {alpha_value:>24.4f} |\n")
        results.append("+--------------------------+---------------------+--------------------------+\n")

        if not np.isnan(chi2_stat) and df > 0:
            chi2_decision = "Accept" if chi2_stat <= critical_chi2 else "Reject"
            results.append(f"\nPearson χ² test: {chi2_decision} H0 (χ² = {chi2_stat:.4f}, critical = {critical_chi2:.4f})\n")
        else:
            results.append("\nPearson χ² test: Invalid (not enough bins or df <= 0)\n")

        kolmogorov_decision = "Accept" if p_value_kol > alpha_value else "Reject"
        results.append(f"Kolmogorov test: {kolmogorov_decision} H0 (p = {p_value_kol:.4f}, α = {alpha_value:.4f})\n")

        if not np.isnan(chi2_stat) and df > 0:
            overall = "Accept" if (chi2_stat <= critical_chi2 and p_value_kol > alpha_value) else "Reject"
            results.append(f"Overall Decision (α = {alpha_value:.2f}): {overall} H0\n")
        else:
            results.append(f"Overall (α = {alpha_value:.2f}): Based on Kolmogorov only - {kolmogorov_decision} H0\n")

        self.kolm_pearson_output.setPlainText(''.join(results))

    def _load_distr(self, distr_number):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select a file with distribution data",
            "", 
            "Text files (*.txt);;Data files (*.dat);;All files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if distr_number == 1:
                    self.file1_content = content
                    self.file1_path = file_path
                    self.file_dist_status_1.setText(f"File 1: {file_path.split('/')[-1]}")
                else:
                    self.file2_content = content
                    self.file2_path = file_path
                    self.file_dist_status_2.setText(f"File 2: {file_path.split('/')[-1]}")
                
                # Enable perform button if both files are loaded
                if self.file1_content and self.file2_content:
                    self.perform_button_one.setEnabled(True)
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file: {str(e)}")        
    def _perform_action_one(self):
        if self.file1_content and self.file2_content:
            # Call your function here
            result = self.process_files_one(self.file1_content, self.file2_content)
            
            #QMessageBox.information(self, "Result", f"Processing completed!\n{result}")
        else: 
            print("fuck me")
    def process_files_one(self, first_str, second_str):
        #checking disperrsion and mean for !!!normal distributions!!!
        first = [float(element) for element in first_str.split()]
        second = [float(element) for element in second_str.split()]
        size_first = len(first)
        size_second = len(second)
        if size_first < 60 or size_second < 60:
            alpha = 0.2
        else:
            alpha = 0.05

        results = []
        if self.two_disp_mean_compare.isChecked():
            results.append("--- PARAMETRIC TESTS (normal data)---")
            #first distr
            first_mean = arithmetic_mean(first)
            var_1 = unbiased_sample_variance(first, first_mean)
            #second distr
            second_mean = arithmetic_mean(second)
            var_2 = unbiased_sample_variance(second, second_mean)
            #check disperssion first
            results.append(f"Check disspersions first.\nComparing {var_1:.4f} and {var_2:.4f}.")
            if var_1 >= var_2:
                f_stat = var_1 / var_2
                dfn, dfd = size_first - 1, size_second - 1
            else:
                f_stat = var_2 / var_1
                dfn, dfd = size_second - 1, size_first - 1
            #critical values
            f_critical = f.ppf(1 - alpha, dfn, dfd)
            results.append(f"f statistics value is {f_stat:.4f} f critical value is {f_critical:.4f}")
            if f_stat <= f_critical:
                results.append(f"{f_stat:.4f} <= {f_critical:.4f} => H0+ (dispersions are the same)")
                results.append(f"Comparing means {first_mean:.4f} and {second_mean:.4f}")

                if size_first + size_second > 25:
                    S_2 = var_1/size_first + var_2/size_second
                    t_stat_mean_comparing = (first_mean - second_mean)/math.sqrt(S_2)
                else:
                    pooled_var = ((size_first - 1) * var_1 + (size_second - 1) * var_2) / (size_first + size_second - 2)
                    S_2 = pooled_var * (1/size_first + 1/size_second)
                    multiply = math.sqrt((size_first*size_second)/(size_first + size_second))
                    t_stat_mean_comparing = ((first_mean - second_mean)/math.sqrt(S_2)) * multiply
                #critical value
                t_critical_mean_comparing = t.ppf(1 - alpha/2, size_first + size_second - 2)
                results.append(f"t statistics value is {t_stat_mean_comparing:.4f} t critical value is {t_critical_mean_comparing:.4f}")
                if abs(t_stat_mean_comparing) <= t_critical_mean_comparing:
                    results.append(f"|{t_stat_mean_comparing:.4f}| <= {t_critical_mean_comparing:.4f} => H0 + (means are the same)")
                else:
                    results.append(f"|{t_stat_mean_comparing:.4f}| > {t_critical_mean_comparing:.4f} => H0 - (means are not the same)")
            else:
                results.append("H0 - (dispersions are not the same) => comparing means has no sense")
        if  self.two_disp_mean_criterias.isChecked():
            results.append("\n--- NON-PARAMETRIC TESTS ---")
            #Wilcoxon criteria
            results.append("Wilcoxon test:")
            try:
                w_statistics = wilcoxon_w(first, second)
                w_critical_value = norm.ppf(1 - alpha/2)
                results.append(f"Comparing w statistics {w_statistics:.4f} and u critical value {w_critical_value:.4f}")
                if abs(w_statistics) <= w_critical_value:
                    results.append(f"|{w_statistics:.4f}| <= {w_critical_value:.4f} => H0 +")
                else:
                    results.append(f"|{w_statistics:.4f}| > {w_critical_value:.4f} => H0 -")
            
            except Exception as e:
                results.append(f"Error in Wilcoxon test: {e}")
            #Mann-Whitney criteria
            results.append("Mann-Whitney test:")
            try:
                u_statistics = mann_whitney_u(first, second)
                u_critical_value = norm.ppf(1 - alpha/2)
                results.append(f"Comparing u {u_statistics:.4f} and u critical value {u_critical_value:.4f}")
                if abs(u_statistics) <= u_critical_value:
                    results.append(f"|{u_statistics:.4f}| <= {u_critical_value:.4f} => H0 +")
                else:
                    results.append(f"|{u_statistics:.4f}| > {u_critical_value:.4f} => H0 -")
        
            except Exception as e:
                results.append(f"Error in Mann-Whitney test: (str{e})")
            #Sign test
            results.append("Sign test:")
            try:
                if len(first) != len(second):
                    results.append("Sign test requires paired data - skipping")
                else:
                    s_statistics, pos_size = sign_test(first, second)
                    if pos_size > 15:
                        s_critical_value = norm.ppf(1 - alpha/2)
                        results.append(f"Comparing S statistics {s_statistics:.4f} and u critical value {s_critical_value:.4f}")
                        if abs(s_statistics) < s_critical_value:
                            results.append(f"|{s_statistics:.4f}| < {s_critical_value:.4f} => H0 +")
                        else:
                            results.append(f"|{s_statistics:.4f}| >= {s_critical_value:.4f} => H0 -")
                    else:
                        s_critical_value = alpha
                        results.append(f"Comparing a0 {s_statistics:.4f} and critical value a {s_critical_value:.4f}")
                        if s_statistics >= s_critical_value:
                            results.append(f"{s_statistics:.4f} >= {s_critical_value:.4f} => H0 +")
                        else:
                            results.append(f"{s_statistics:.4f} < {s_critical_value:.4f} => H0 -")
            except Exception as e:
                results.append(f"Error in Sign test: {e}") 
            #mid_rank_diff_criteria
            results.append("Mid rank diff criteria:")
            try:
                v_statistics = mid_rank_diff_criteria(first, second)
                u_critical_value = norm.ppf(1 - alpha/2)
                results.append(f"Comparing v {v_statistics:.4f} and u critical value {u_critical_value:.4f}")
                if abs(v_statistics) <= u_critical_value:
                    results.append(f"|{v_statistics:.4f}| <= {u_critical_value:.4f} => H0 +")
                else:
                    results.append(f"|{v_statistics:.4f}| > {u_critical_value:.4f} => H0 -")
            except Exception as e:
                results.append(f"Error in Mid rank diff criteria: {e}") 
            """
            #mid rank diff criteria
            results.append("Mid rank difference test:")
            try:
                v_statistics = mid_rank_diff_criteria(first, second)
                s_critical_value = norm.ppf(1 - alpha/2)
                results.append(f"Comparing v statistics {v_statistics:.4f} and u critical value {s_critical_value:.4f}")
                if abs(v_statistics) <= s_critical_value:
                    results.append(f"|{v_statistics:.4f}| <= {s_critical_value:.4f} => H0 +")
                else:
                    results.append(f"|{v_statistics:.4f}| > {s_critical_value:.4f} => H0 -")
            except Exception as e:
                results.append(f"Error in Mid rank test: {e}")
            """        
            results.append("Abbe criteria (Independence check) for the first entered distribution:")
            try:
                # Calculate P-value using norm.cdf
                U = abbe_independence_criteria(first)
                P = 2 * (1 - norm.cdf(abs(U)))
                results.append(f"Comparing P value {P:.4f} and critical value a {alpha:.4f}")
                if P > alpha:
                    results.append(f"P={P:.4f} > {alpha} => H0 + (Data appears independent)")
                else:
                    results.append(f"P={P:.4f} <= {alpha} => H0 - (Data appears dependent)")
                    
            except Exception as e:
                results.append(f"Error in Abbe criterion: {str(e)}")
          
            

        self.output_1_dist_text.setPlainText('\n'.join(results))

    def _lab4_one(self):
        print("Isn't implemented yes")
    def _lab4_two(self):
        print("Isn't implemented yes")

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

            self.bins_amount = 0
            if self.rd_custom.isChecked():
                try:
                    bins_str = self.custom_entry.text().strip()
                    if bins_str:
                        self.bins_amount = int(bins_str)
                        if self.bins_amount <= 0:
                            raise ValueError("Number of intervals must be positive.")
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input",
                                        f"Invalid number of intervals: {e}. Choosing automatically.")
                    self.bins_amount = 0
                    self.rd_default.setChecked(True)

            self.hist_info = midpoint_intervals_forming(current_data, bins=self.bins_amount)
            self.intervals_array = self.hist_info['intervals_array']
            delta_h = float(self.hist_info['delta_h'])
            self.bins_amount = self.hist_info["bins_amount"]

            self.frequencies_array = frequencies(current_data, delta_h, self.bins_amount)
            relative_frequencies_array = relative_frequencies(self.frequencies_array, len(current_data))
            x_axis_ecdf, y_axis_ecdf = ecdf(current_data)

            print(f"intervals_array size: {len(self.intervals_array)}")
            print(f"relative_frequencies_array size: {len(relative_frequencies_array)}")
            print(f"frequencies_array size: {len(self.frequencies_array)}")
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
        if bar_width <= 0: 
            bar_width = 1.0
        
        # Plot histogram bars
        histogram_item = pg.BarGraphItem(
            x=intervals_array,
            height=relative_frequencies_array,
            width=bar_width,
            brush='cornflowerblue',
            pen=pg.mkPen('k', width=0.2)
        )
        self.histogram_widget.addItem(histogram_item)
        
        # Generate smooth density curve
        if len(intervals_array) > 1:
            # Create data points weighted by frequencies for KDE
            data_points = []
            for x, freq in zip(intervals_array, relative_frequencies_array):
                # Add points proportional to frequency (multiply by large number for better sampling)
                n_points = max(1, int(freq * 1000))
                data_points.extend([x] * n_points)
            
            if len(data_points) > 1:
                # Kernel Density Estimation
                kde = gaussian_kde(data_points)
                
                # Generate smooth x values for the curve
                x_min, x_max = min(intervals_array), max(intervals_array)
                x_smooth = np.linspace(x_min - bar_width, x_max + bar_width, 200)
                
                # Calculate density values
                density_values = kde(x_smooth)
                
                # Scale density to match histogram scale (approximate)
                # This assumes your relative_frequencies represent density values
                scale_factor = max(relative_frequencies_array) / max(density_values) if max(density_values) > 0 else 1
                density_values *= scale_factor
                
                # Plot the smooth curve
                curve_item = pg.PlotCurveItem(
                    x=x_smooth, 
                    y=density_values,
                    pen=pg.mkPen('red', width=2)
                )
                self.histogram_widget.addItem(curve_item)
        
        # Set y-axis range based on max bin height
        max_height = max(relative_frequencies_array) if relative_frequencies_array else 1.0
        self.histogram_widget.setYRange(0, max_height * 1.1)
        
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
    
    def _plot_ecdf_theor(self, dist_name, estimates, se_estimates, alpha, n, dist_array):
        # Do NOT clear the widget, so existing plots (histogram, empirical ECDF, discrete ECDF) remain
        if dist_array is None or len(dist_array) == 0:
            self.ecdf_widget.setTitle("Theoretical CDF (no data)")
            return

        # Generate smooth x values
        x_theor = np.linspace(min(dist_array), max(dist_array), 1000)
        z_alpha = norm.ppf(1 - alpha / 2)  # Critical value for CI

        # Compute theoretical CDF and variance based on distribution
        if dist_name == "Exponential":
            lambda_hat = estimates['λ']
            y_theor = 1 - np.exp(-lambda_hat * x_theor)
            var_theor = (x_theor * np.exp(-lambda_hat * x_theor))**2 * (lambda_hat**2 / n)

        elif dist_name == "Uniform":
            a_hat = estimates['a']
            b_hat = estimates['b']
            y_theor = np.clip((x_theor - a_hat) / (b_hat - a_hat), 0, 1)
            se = se_estimates['a']  # Using 'a' SE as a proxy
            var_theor = np.ones_like(x_theor) * (se**2 / n)  # Crude approximation

        elif dist_name == "Weibull":
            alpha_hat = estimates['α (shape)']
            beta_hat = estimates['β (scale)']
            y_theor = 1 - np.exp(- (x_theor / beta_hat)**alpha_hat)
            temp = (x_theor / beta_hat)**alpha_hat
            partial_alpha = temp * np.log(x_theor / beta_hat + 1e-10) * np.exp(-temp)
            partial_beta = (alpha_hat / beta_hat) * (x_theor / beta_hat)**(alpha_hat - 1) * np.exp(-temp)
            var_theor = (partial_alpha * se_estimates['α (shape)'])**2 + (partial_beta * se_estimates['β (scale)'])**2

        elif dist_name == "Normal":
            mu_hat = estimates['μ']
            sigma_hat = estimates['σ']
            z = (x_theor - mu_hat) / sigma_hat
            y_theor = norm.cdf(z)
            phi_z = norm.pdf(z)
            var_theor = (phi_z**2) * (1 / n + (z**2) / (2 * n))

        elif dist_name == "Laplace":
            mu_hat = estimates['μ']
            b_hat = estimates['b']
            y_theor = 0.5 + 0.5 * np.sign(x_theor - mu_hat) * (1 - np.exp(-np.abs(x_theor - mu_hat) / b_hat))
            diff = np.abs(x_theor - mu_hat)
            partial_b = (diff / (b_hat**2)) * np.exp(-diff / b_hat) * (0.5 * np.sign(x_theor - mu_hat))
            var_theor = (partial_b * se_estimates['b'])**2

        else:
            self.ecdf_widget.setTitle("Unsupported Distribution")
            return

        # Compute CI bounds
        lower = np.clip(y_theor - z_alpha * np.sqrt(var_theor), 0, 1)
        upper = np.clip(y_theor + z_alpha * np.sqrt(var_theor), 0, 1)

        # Plot smooth theoretical CDF and CI curves on top of existing plots
        ecdf_theor_item = pg.PlotCurveItem(
            x=x_theor, y=y_theor, pen=pg.mkPen('blue', width=2)
        )
        lower_curve = pg.PlotCurveItem(
            x=x_theor, y=lower, pen=pg.mkPen('red', style=Qt.PenStyle.DashLine)
        )
        upper_curve = pg.PlotCurveItem(
            x=x_theor, y=upper, pen=pg.mkPen('red', style=Qt.PenStyle.DashLine)
        )

        self.ecdf_widget.addItem(ecdf_theor_item)
        self.ecdf_widget.addItem(lower_curve)
        self.ecdf_widget.addItem(upper_curve)
        self.ecdf_widget.setTitle(f"Empirical and Theoretical CDF ({dist_name}) with {1-alpha:.2%} CI")
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