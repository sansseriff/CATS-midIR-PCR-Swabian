# from snspd_measure.inst.keysight33622A import keysight33622A

# PySide2 for the UI
from PySide2.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QMessageBox
from PySide2.QtCore import QTimer
from PySide2.QtGui import QPalette, QColor

from snspd_measure.inst.sim900 import sim928

# Import client instruments
from client_keysight33622A import ClientKeysight33622A
from client_keysightE36312A import ClientKeysightE36312A

# matplotlib for the plots, including its Qt backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import time
# to generate new UI file: pyside2-uic CoincidenceExampleWindow_XXX.ui > CoincidenceExampleWindow_mx.py
# Please use the QtDesigner to edit the ui interface file
from CoincidenceExampleWindow_m4 import Ui_CoincidenceExample

# numpy and math for statistical analysis
import numpy
import math
import warnings
warnings.filterwarnings('ignore')

# for scope trace
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import yaml

# all required TimeTagger dependencies
from TimeTagger import Coincidences, Histogram2D, Counter, Correlation, createTimeTagger, freeTimeTagger, Histogram, FileWriter, FileReader, TT_CHANNEL_FALLING_EDGES, Resolution, DelayedChannel, GatedChannel, Countrate, CHANNEL_UNUSED
from time import sleep

import json
import csv
import os.path

import serial # Import serial for exception handling
import termios # Import termios for catching specific OS error

# from awgClient import AWGClient

class SIM928ControlDialog(QDialog):
    """Modal dialog for controlling the SIM928 voltage source"""
    
    def __init__(self, parent=None):
        super(SIM928ControlDialog, self).__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("SIM928 Voltage Source Control")
        self.setModal(True)
        # Debounce timer for spinbox-driven voltage updates
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(100)  # 0.1 s debounce
        self.debounce_timer.timeout.connect(self.set_voltage)
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # Voltage control section
        voltage_group = QGroupBox("Voltage Control")
        voltage_layout = QHBoxLayout()
        
        voltage_layout.addWidget(QLabel("Voltage:"))
        
        self.voltage_spinbox = QDoubleSpinBox()
        self.voltage_spinbox.setRange(0.0, 15.0)
        self.voltage_spinbox.setDecimals(3)
        self.voltage_spinbox.setSingleStep(0.10)  # millivolt increments
        self.voltage_spinbox.setSuffix(" V")
        self.voltage_spinbox.setValue(0.0)
        # Debounced update when the value changes
        self.voltage_spinbox.valueChanged.connect(self._on_voltage_spinbox_changed)
        voltage_layout.addWidget(self.voltage_spinbox)
        
        self.set_voltage_button = QPushButton("Set")
        self.set_voltage_button.clicked.connect(self.set_voltage)
        voltage_layout.addWidget(self.set_voltage_button)
        
        voltage_group.setLayout(voltage_layout)
        layout.addWidget(voltage_group)
        
        # Power control section
        power_group = QGroupBox("Power Control")
        power_layout = QHBoxLayout()
        
        self.turn_on_button = QPushButton("Turn On")
        self.turn_on_button.clicked.connect(self.turn_on_source)
        power_layout.addWidget(self.turn_on_button)
        
        self.turn_off_button = QPushButton("Turn Off")
        self.turn_off_button.clicked.connect(self.turn_off_source)
        power_layout.addWidget(self.turn_off_button)
        
        power_group.setLayout(power_layout)
        layout.addWidget(power_group)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        self.setLayout(layout)

    def _on_voltage_spinbox_changed(self, _value):
        """Restart debounce timer on every change; set_voltage runs on timeout."""
        # Restart the timer so set_voltage only fires after changes settle
        self.debounce_timer.start()
        
    def set_voltage(self):
        """Set the voltage on the SIM928 source"""
        # Prevent a pending debounce timeout from firing again after manual Set
        try:
            self.debounce_timer.stop()
        except Exception:
            pass

        voltage = self.voltage_spinbox.value()
        if self.parent_window:
            success = self.parent_window._set_source_voltage_robustly(voltage)
            # Silent operation (no dialog). Log to console for traceability.
            if success:
                print(f"SIM928: Voltage set to {voltage:.3f} V")
            else:
                print(f"SIM928: Failed to set voltage to {voltage:.3f} V")
        else:
            # No parent available; keep silent to avoid intrusive dialogs
            print("SIM928: No parent window available to set voltage")
            
    def turn_on_source(self):
        """Turn on the SIM928 source"""
        if self.parent_window and self.parent_window.source:
            try:
                self.parent_window.source.turnOn()
                QMessageBox.information(self, "Success", "SIM928 source turned on")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to turn on source: {e}")
        else:
            QMessageBox.warning(self, "Error", "SIM928 source not available")
            
    def turn_off_source(self):
        """Turn off the SIM928 source"""
        if self.parent_window and self.parent_window.source:
            try:
                self.parent_window.source.turnOff()
                QMessageBox.information(self, "Success", "SIM928 source turned off")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to turn off source: {e}")
        else:
            QMessageBox.warning(self, "Error", "SIM928 source not available")

class Keysight33622AControlDialog(QDialog):
    """Modal dialog for controlling the Keysight 33622A function generator"""
    
    def __init__(self, parent=None):
        super(Keysight33622AControlDialog, self).__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Keysight 33622A Function Generator Control")
        self.setModal(True)
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # High level control section
        level_group = QGroupBox("High Level Control")
        level_layout = QHBoxLayout()
        
        level_layout.addWidget(QLabel("High Level:"))
        
        self.level_spinbox = QDoubleSpinBox()
        self.level_spinbox.setRange(0.0, 3.0)
        self.level_spinbox.setDecimals(3)
        self.level_spinbox.setSingleStep(0.001)  # millivolt increments
        self.level_spinbox.setSuffix(" V")
        self.level_spinbox.setValue(0.0)
        level_layout.addWidget(self.level_spinbox)
        
        self.set_level_button = QPushButton("Set")
        self.set_level_button.clicked.connect(self.set_high_level)
        level_layout.addWidget(self.set_level_button)
        
        level_group.setLayout(level_layout)
        layout.addWidget(level_group)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
    def set_high_level(self):
        """Set the high level on the Keysight 33622A function generator"""
        high_level = self.level_spinbox.value()
        if self.parent_window and self.parent_window.function_gen:
            try:
                # Set amplitude and offset to achieve the desired high level
                # High level = amplitude + offset, Low level = 0 = offset - amplitude
                # Therefore: amplitude = high_level/2, offset = high_level/2
                
                offset = high_level / 2.0
                
                self.parent_window.function_gen.set_amplitude(2, high_level)  # Channel 2
                self.parent_window.function_gen.set_offset(2, offset)        # Channel 2
                
                QMessageBox.information(self, "Success", f"High level set to {high_level:.3f} V\n(Amplitude: {high_level:.3f} V, Offset: {offset:.3f} V)")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to set high level: {e}")
        else:
            QMessageBox.warning(self, "Error", "Function generator not available")

class CoincidenceExample(QMainWindow):
    ''' Small example of how to create a UI for the TimeTagger with the PySide2 framework'''

    def __init__(self, tagger):
        '''Constructor of the coincidence example window
        The TimeTagger object must be given as arguments to support running many windows at once.'''

        # Create the UI from the designer file and connect its action buttons
        super(CoincidenceExample, self).__init__()
        self.ui = Ui_CoincidenceExample()
        self.ui.setupUi(self)
        self.ui.PCRButton.clicked.connect(self.PCR)
        self.ui.triggerScanButton.clicked.connect(self.open_sim928_control)
        self.ui.clearButton.clicked.connect(self.open_keysight33622A_control)
        self.ui.saveButton.clicked.connect(self.saveHistogram)
        # self.ui.saveTagsButton.clicked.connect(self.saveTagsSimple)
        # self.ui.TraceGen.clicked.connect(self.saveTrace)

        self.ui.fromFile.clicked.connect(self.fromFile)
        self.ui.toFileButton.clicked.connect(self.toFile)


        # Update the measurements whenever any input configuration changes
        self.ui.channelA.valueChanged.connect(self.updateMeasurements)
        self.ui.channelB.valueChanged.connect(self.updateMeasurements)
        self.ui.channelC.valueChanged.connect(self.updateMeasurements)
        self.ui.channelD.valueChanged.connect(self.updateMeasurements)
        self.ui.delayA.valueChanged.connect(self.updateMeasurements)
        self.ui.delayB.valueChanged.connect(self.updateMeasurements)
        self.ui.delayC.valueChanged.connect(self.updateMeasurements)
        self.ui.delayD.valueChanged.connect(self.updateMeasurements)
        self.ui.triggerA.valueChanged.connect(self.updateMeasurements)
        self.ui.triggerB.valueChanged.connect(self.updateMeasurements)
        self.ui.triggerC.valueChanged.connect(self.updateMeasurements)
        self.ui.triggerD.valueChanged.connect(self.updateMeasurements)
        self.ui.deadTimeA.valueChanged.connect(self.updateMeasurements)
        self.ui.deadTimeB.valueChanged.connect(self.updateMeasurements)
        self.ui.deadTimeC.valueChanged.connect(self.updateMeasurements)
        self.ui.deadTimeD.valueChanged.connect(self.updateMeasurements)

        self.ui.testsignalA.stateChanged.connect(self.updateMeasurements)
        self.ui.testsignalB.stateChanged.connect(self.updateMeasurements)
        self.ui.testsignalB.stateChanged.connect(self.updateMeasurements)
        self.ui.coincidenceWindow.valueChanged.connect(self.updateMeasurements)
        self.ui.IntType.currentTextChanged.connect(self.updateMeasurements)
        self.ui.LogScaleCheck.stateChanged.connect(self.updateMeasurements)
        self.ui.IntTime.valueChanged.connect(self.updateMeasurements)

        self.ui.correlationBinwidth.valueChanged.connect(
            self.updateMeasurements)
        self.ui.correlationBins.valueChanged.connect(self.updateMeasurements)

        # Create the matplotlib figure with its subplots for the counter and correlation
        self.fig = Figure()
        self.counterAxis = self.fig.add_subplot(211)
        self.correlationAxis = self.fig.add_subplot(212)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ui.plotLayout.addWidget(self.toolbar)
        self.ui.plotLayout.addWidget(self.canvas)

        self.masked_hist_bins = 2

        self.fudge_factor = 1.0



        # --- Added for robust connection ---
        self.source_port = '/dev/ttyUSB0' # Initial port
        self.possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2'] # Ports to try
        self.source_gpib_addr = 2 # Assuming fixed GPIB address
        self.source_slot = 1      # Assuming fixed slot
        # --- End Added ---

        # Initialize source
        try:
            self.source = sim928(self.source_port, self.source_gpib_addr, self.source_slot)
            self.source.connect()
            self.source.turnOn()
        except serial.SerialException as e:
            print(f"Initial connection to SIM928 failed on {self.source_port}: {e}")
            # Optionally try the other port immediately
            current_index = self.possible_ports.index(self.source_port)
            next_index = (current_index + 1) % len(self.possible_ports)
            self.source_port = self.possible_ports[next_index]
            print(f"Trying alternative port: {self.source_port}")
            try:
                self.source = sim928(self.source_port, self.source_gpib_addr, self.source_slot)
                self.source.connect()
                self.source.turnOn()
                print(f"Successfully connected to {self.source_port}")
            except serial.SerialException as e2:
                 print(f"Connection failed on alternative port {self.source_port}: {e2}")
                 # Handle failure - maybe disable PCR button or show error message
                 self.source = None # Indicate source is not available
                 # You might want to disable the PCR button here
                 # self.ui.PCRButton.setEnabled(False)

        # Initialize Keysight instruments
        try:
            self.function_gen = ClientKeysight33622A()
            self.function_gen.connect()
            print("Function generator (33622A) connected successfully")
        except Exception as e:
            print(f"Failed to connect to function generator: {e}")
            self.function_gen = None

        try:
            self.power_supply = ClientKeysightE36312A()
            self.power_supply.connect()
            print("Power supply (E36312A) connected successfully")
        except Exception as e:
            print(f"Failed to connect to power supply: {e}")
            self.power_supply = None


        # Create the TimeTagger measurements
        self.running = True
        self.measurements_dirty = False
        self.tagger = tagger
        self.IntType = "Rolling"
        self.last_channels = [9, -5, -14, 18]
        self.active_channels = []
        self.last_coincidenceWindow = 0
        self.updateMeasurements()

        # Use a timer to redraw the plots every 100ms
        self.draw()
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw)
        self.timer.start(200)
        self.clock_divider = 2000  # divider 156.25MHz down to 78.125 KHz
        self.tagger.setEventDivider(18,self.clock_divider)
        
        # Flag for saving histogram when histBlock is full
        self.save_requested = False
        self.save_filename = None


    def fromFile(self):
        # self.ent = False
        with open("./channel_params.yaml", "r", encoding="utf8") as stream:
            try:
                ui_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.ui.channelA.setValue(int(ui_data["Channels"]["ChA"]["channel"]))
        self.ui.triggerA.setValue(float(ui_data["Channels"]["ChA"]["trigger"]))
        self.ui.delayA.setValue(int(ui_data["Channels"]["ChA"]["delay"]))
        self.ui.deadTimeA.setValue(int(ui_data["Channels"]["ChA"]["dead_time"]))

        self.ui.channelB.setValue(int(ui_data["Channels"]["ChB"]["channel"]))
        self.ui.triggerB.setValue(float(ui_data["Channels"]["ChB"]["trigger"]))
        self.ui.delayB.setValue(int(ui_data["Channels"]["ChB"]["delay"]))
        self.ui.deadTimeB.setValue(int(ui_data["Channels"]["ChB"]["dead_time"]))

        self.ui.channelC.setValue(int(ui_data["Channels"]["ChC"]["channel"]))
        self.ui.triggerC.setValue(float(ui_data["Channels"]["ChC"]["trigger"]))
        self.ui.delayC.setValue(int(ui_data["Channels"]["ChC"]["delay"]))
        self.ui.deadTimeC.setValue(int(ui_data["Channels"]["ChC"]["dead_time"]))

        self.ui.channelD.setValue(int(ui_data["Channels"]["ChD"]["channel"]))
        self.ui.triggerD.setValue(float(ui_data["Channels"]["ChD"]["trigger"]))
        self.ui.delayD.setValue(int(ui_data["Channels"]["ChD"]["delay"]))
        self.ui.deadTimeD.setValue(int(ui_data["Channels"]["ChD"]["dead_time"]))

        self.updateMeasurements()

    def toFile(self):

        settings_dict = {
                    "Channels": {
                        "ChA": {
                            "channel": int(self.ui.channelA.value()),
                            "trigger": float(self.ui.triggerA.value()),
                            "delay": int(self.ui.delayA.value()),
                            "dead_time": int(self.ui.deadTimeA.value())
                        },
                        "ChB": {
                            "channel": int(self.ui.channelB.value()),
                            "trigger": float(self.ui.triggerB.value()),
                            "delay": int(self.ui.delayB.value()),
                            "dead_time": int(self.ui.deadTimeB.value())
                        },
                        "ChC": {
                            "channel": int(self.ui.channelC.value()),
                            "trigger": float(self.ui.triggerC.value()),
                            "delay": int(self.ui.delayC.value()),
                            "dead_time": int(self.ui.deadTimeC.value())
                        },
                        "ChD": {
                            "channel": int(self.ui.channelD.value()),
                            "trigger": float(self.ui.triggerD.value()),
                            "delay": int(self.ui.delayD.value()),
                            "dead_time": int(self.ui.deadTimeD.value())
                        }
                    }
                }

        with open("channel_params.yaml", "w", encoding="utf8") as stream:
            try:
                yaml.safe_dump(settings_dict, stream)
            except yaml.YAMLError as exc:
                print(exc)


    def open_sim928_control(self):
        """Open the SIM928 control dialog"""
        dialog = SIM928ControlDialog(self)
        dialog.exec_()

    def open_keysight33622A_control(self):
        """Open the Keysight 33622A control dialog"""
        dialog = Keysight33622AControlDialog(self)
        dialog.exec_()

    def reInit(self):
        # Create the TimeTagger measurements
        self.running = True
        self.measurements_dirty = False
        self.tagger = tagger
        self.IntType = "Rolling"
        self.last_channels = [9, -5, -14, 18]
        self.last_coincidenceWindow = 0
        self.updateMeasurements()

        # Use a timer to redraw the plots every 100ms
        self.draw()
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw)
        self.timer.start(200)
        self.tagger.setEventDivider(18, self.clock_divider)


    def getCouterNormalizationFactor(self):
        bin_index = self.counter.getIndex()
        # normalize 'clicks / bin' to 'kclicks / second'
        return 1e12 / bin_index[1] / 1e3

    def updateMeasurements(self):
        '''Create/Update all TimeTagger measurement objects'''

        # If any configuration is changed while the measurements are stopped, recreate them on the start button
        if not self.running:
            self.measurements_dirty = True
            return

        # Set the input delay, trigger level, and test signal of both channels
        channels = [self.ui.channelA.value(), self.ui.channelB.value(), self.ui.channelC.value(),
                    self.ui.channelD.value()]

        self.active_channels = []

        if channels[0] != 0:
            self.tagger.setInputDelay(channels[0], self.ui.delayA.value())
            self.tagger.setTriggerLevel(channels[0], self.ui.triggerA.value())
            self.tagger.setDeadtime(channels[0], int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setDeadtime(channels[0]*-1, int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setTestSignal(channels[0], self.ui.testsignalA.isChecked())
            self.active_channels.append(channels[0])



        if channels[1] != 0:
            self.tagger.setInputDelay(channels[1], self.ui.delayB.value())
            self.tagger.setTriggerLevel(channels[1], self.ui.triggerB.value())
            self.tagger.setDeadtime(channels[1], int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setDeadtime(channels[1]*-1, int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setTestSignal(channels[1], self.ui.testsignalB.isChecked())
            self.active_channels.append(channels[1])

        if channels[2] != 0:
            self.tagger.setInputDelay(channels[2], self.ui.delayC.value())
            self.tagger.setTriggerLevel(channels[2], self.ui.triggerC.value())
            self.tagger.setDeadtime(channels[2], int(self.ui.deadTimeC.value() * 1000))
            self.tagger.setDeadtime(channels[2]*-1, int(self.ui.deadTimeC.value() * 1000))
            self.active_channels.append(channels[2])

        if channels[3] != 0:
            self.tagger.setInputDelay(channels[3], self.ui.delayD.value())
            self.tagger.setTriggerLevel(channels[3], self.ui.triggerD.value())
            self.tagger.setDeadtime(channels[3], int(self.ui.deadTimeD.value() * 1000))
            self.tagger.setDeadtime(channels[3]*-1, int(self.ui.deadTimeD.value() * 1000))
            self.active_channels.append(channels[3])

        self.correlationAxis.set_yscale('log')
        self.seconds = 1
        print("histblock depth: ", int(self.ui.IntTime.value()*5))
        self.histBlock = numpy.zeros((int(self.ui.IntTime.value()*5),self.ui.correlationBins.value() - self.masked_hist_bins))

        self.buffer = numpy.zeros((1,self.ui.correlationBins.value()))[self.masked_hist_bins:]
        self.buffer_old = numpy.zeros((1, self.ui.correlationBins.value()))[self.masked_hist_bins:]


        self.BlockIndex = 0

        # Only recreate the counter if its parameter has changed,
        # else we'll clear the count trace too often
        coincidenceWindow = self.ui.coincidenceWindow.value()
        if self.last_channels != self.active_channels or self.last_coincidenceWindow != coincidenceWindow:
            self.last_channels = self.active_channels
            self.last_coincidenceWindow = coincidenceWindow

            # Create the virtual coincidence channel
            self.coincidences = Coincidences(
                self.tagger,
                [self.active_channels[1:]],
                coincidenceWindow
            )

            # Measure the count rate of both input channels and the coincidence channel
            # Use 200 * 50ms binning
            self.counter = Counter(
                self.tagger,
                self.active_channels + list(self.coincidences.getChannels()),
                binwidth=int(50e9),
                n_values=200
            )

        print(self.active_channels)


        on_start = 30
        on_stop = 270
        off_start = 450
        off_stop = 950

        # for us right now (oct 9 2024), self.active_channels[2] (3rd row) is 5, which is the snspd
        self.filtered = GatedChannel(self.tagger, self.active_channels[2], self.active_channels[0], -self.active_channels[0])
        self.delay_1_start = DelayedChannel(self.tagger, self.active_channels[0], int(on_start*1e9))
        self.delay_1_stop = DelayedChannel(self.tagger, self.active_channels[0], int(on_stop*1e9))

        self.delay_2_start = DelayedChannel(self.tagger, self.active_channels[0], int(off_start*1e9))

        self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(off_stop*1e9))
        # self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(800e9))


        self.ratio_on = (on_stop - on_start) / 1000
        self.ratio_off = (off_stop - off_start) / 1000

        self.ratio_on = self.ratio_on * self.fudge_factor
        self.ratio_off = self.ratio_off / self.fudge_factor


        # thermal source on
        self.filtered_on = GatedChannel(self.tagger, self.active_channels[2], self.delay_1_start.getChannel(), self.delay_1_stop.getChannel())

        # thermal source off
        self.filtered_off = GatedChannel(self.tagger, self.active_channels[2], self.delay_2_start.getChannel(), self.delay_2_stop.getChannel())


        # Measure the correlation between A and B
        self.correlation = Correlation(
            self.tagger,
            #self.a_combined.getChannel(),
            #self.b_combined.getChannel(),
            # self.active_channels[1],
            # self.filtered.getChannel(),
            self.filtered_on.getChannel(),
            CHANNEL_UNUSED,

            self.ui.correlationBinwidth.value(),
            self.ui.correlationBins.value())

        self.tagger.sync()

        # Create the measurement plots
        self.counterAxis.clear() # this is a matplotlib figure
        self.plt_counter = self.counterAxis.plot(
            self.counter.getIndex() * 1e-12,
            self.counter.getData().T * self.getCouterNormalizationFactor()
        )
        self.counterAxis.set_xlabel('time (s)')
        self.counterAxis.set_ylabel('count rate (kEvents/s)')
        self.counterAxis.set_title('Count rate')
        self.counterAxis.legend(['A', 'B', 'C', 'D','coincidences'])
        self.counterAxis.grid(True)

        self.correlationAxis.clear()
        index = self.correlation.getIndex()[self.masked_hist_bins:]
        #data = self.correlation.getDataNormalized()
        data = self.correlation.getData()[self.masked_hist_bins:]
        self.plt_correlation = self.correlationAxis.plot(
            index * 1e-3,
            data
        )



        self.correlationAxis.set_xlabel('time (ns)')
        self.correlationAxis.set_ylabel('Counts')
        self.correlationAxis.set_title('Histogram between A and B')
        self.correlationAxis.grid(True)

        # Generate nicer plots
        self.fig.tight_layout()

        self.measurements_dirty = False

        # Update the plot with real numbers
        self.draw()
        ####

    # disconnected
    def startClicked(self):
        
        '''Handler for the start action button'''
        self.running = True

        if self.measurements_dirty:
            # If any configuration is changed while the measurements are stopped,
            # recreate them on the start button
            self.updateMeasurements()
        else:
            # else manually start them
            self.counter.start()
            self.correlation.start()

    def stopClicked(self):
        '''Handler for the stop action button'''
        self.running = False
        self.counter.stop()
        self.correlation.stop()

    def clearClicked(self):
        '''Handler for the clear action button'''
        self.correlation.clear()

    
    def saveHistogram(self):
        """Set up saving histogram data when histBlock is full"""
        from PySide2.QtWidgets import QFileDialog
        
        # Get save location using file dialog
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save Histogram Data',
            directory='histogram_data.json',  # default name
            filter='JSON Files (*.json);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )
        
        # If user cancels, exit the function
        if not filename:
            print("Save operation cancelled.")
            return
        
        # Ensure we have a .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
        
        # Set the flag to start saving when histBlock is full
        self.save_requested = True
        self.save_filename = filename
        
        print(f"Histogram will be saved to {filename} when data collection is complete.")
        print(f"Integration depth: {int(self.ui.IntTime.value()*5)} blocks")
        print("Data collection in progress...")
    
    def _save_histogram_data(self):
        """Internal method to save the accumulated histogram data"""
        try:
            # Get the accumulated data (sum of histBlock)
            accumulated_data = numpy.sum(self.histBlock, axis=0)
            
            # Get the x-axis data (index)
            index = self.correlation.getIndex()[self.masked_hist_bins:]
            
            # Prepare data for JSON export
            data_dict = {
                'x_axis_ps': index.tolist(),  # Convert to list for JSON serialization
                'histogram_counts': accumulated_data.tolist(),
                'integration_time_value': self.ui.IntTime.value(),
                'integration_blocks': int(self.ui.IntTime.value()*5),
                'binwidth_ps': self.ui.correlationBinwidth.value(),
                'total_bins': self.ui.correlationBins.value(),
                'masked_bins': self.masked_hist_bins,
                'timestamp': str(numpy.datetime64('now'))
            }
            
            # Save to JSON file
            with open(self.save_filename, 'w') as file:
                json.dump(data_dict, file, indent=2)
            
            print(f"Histogram data successfully saved to: {self.save_filename}")
            print(f"Total counts in histogram: {numpy.sum(accumulated_data)}")
            
        except Exception as e:
            print(f"Error saving histogram data: {e}")
        finally:
            # Reset the save flag
            self.save_requested = False
            self.save_filename = None
    
    # def saveHistogram(self):

    #     pass
        #  wf = keysight33622A('10.7.0.187')
        #  wf.connect()
        #  V_pp = 0.090 #in V 
        #  Start = 0.055
        #  Stop = 0.090
        #  Step = 0.005
        #  offset = numpy.arange(Start,Stop+0.005,Step) # in V 
        #  offset = numpy.append(offset,0.093)
        #  I_det = []
        #  c = input('How Long Do You Want to Integrate For?: ')
        #  int_time = int(c)*(numpy.ones((len(offset)-1,),dtype=int))
        #  int_time = numpy.append(int_time,2*int(c))
        #  wv = input('What wavelength is this for?: ')
        

        # # Determining Bias at the Detector
        #  for i in offset:
        #      V_det = ((V_pp/2) + i)/100
        #      I_det.append(((V_det/50)*1000000).round(4))  #in uA

        # #Setting Up Filter Channel 
        #  wf.channels_on()
        #  wf.phase_zero()
        #  wf.phase_sync()
        #  wf.filter_channel(-45,3000)

        # # making array for plotting 
        #  I_b = numpy.asarray(I_det,dtype = 'float')

        #  for i in range(len(I_b)):
        #      wf.gating_channel(offset[i])
        #      wf.phase_sync()
        #      time.sleep(1)
           
        #      print("starting "+str(I_b[i])+' histogram')
        # # numpy.save("histogram_data.npy", self.persistentData
        # #int_time = input("how long do you want to integrate for?")
        #      time.sleep(int_time[i])
        #      data = self.correlation.getData()
        # #array = self.persistentData
        #      json_string = json.dumps(data.tolist())
        #      with open(wv+'_1um_R1C4_'+str(I_b[i])+'uA_GatedRelLat.json','w') as file:
        #         file.write(json_string)
        #      print("finished")

        #  wf.channels_off()
        #  wf.disconnect()
    

    def saveTags(self):
        #depreciated
        self.tagger.reset()


        channels = [self.ui.channelA.value(), self.ui.channelB.value(), self.ui.channelC.value(),
                    self.ui.channelD.value()]


        if channels[0] != 0:
            #self.tagger.setInputDelay(channels[0], self.ui.delayA.value())
            self.tagger.setTriggerLevel(channels[0], self.ui.triggerA.value())
            self.tagger.setDeadtime(channels[0], int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setDeadtime(channels[0]*-1, int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setTestSignal(channels[0], self.ui.testsignalA.isChecked())

        if channels[1] != 0:
            #self.tagger.setInputDelay(channels[1], self.ui.delayB.value())
            self.tagger.setTriggerLevel(channels[1], self.ui.triggerB.value())
            self.tagger.setDeadtime(channels[1], int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setDeadtime(channels[1]*-1, int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setTestSignal(channels[1], self.ui.testsignalB.isChecked())

        if channels[2] != 0:
            #self.tagger.setInputDelay(channels[2], self.ui.delayC.value())
            self.tagger.setTriggerLevel(channels[2], self.ui.triggerC.value())
            self.tagger.setDeadtime(channels[2], int(self.ui.deadTimeC.value() * 1000))
            self.tagger.setDeadtime(channels[2]*-1, int(self.ui.deadTimeC.value() * 1000))

        if channels[3] != 0:
            #self.tagger.setInputDelay(channels[3], self.ui.delayD.value())
            self.tagger.setTriggerLevel(channels[3], self.ui.triggerD.value())
            self.tagger.setDeadtime(channels[3], int(self.ui.deadTimeD.value() * 1000))
            self.tagger.setDeadtime(channels[3]*-1, int(self.ui.deadTimeD.value() * 1000))
        self.tagger.setEventDivider(18, self.clock_divider)
        # self.a_combined = AverageChannel(self.tagger, -2, (-2, -3, -4))
        # self.b_combined = AverageChannel(self.tagger, -6, (-6, -7, -8))

        file = str(self.ui.saveFileName.text()) + ".ttbin"
        print("saving ", file, " in working directory")
        file_writer = FileWriter(self.tagger, file, [channels[0], self.a_combined.getChannel(),self.b_combined.getChannel()])
        #file_writer = FileWriter(self.tagger, file, [channels[0],channels[1]])
        sleep(self.ui.saveTime.value())  # write for some time
        file_writer.stop()
        print("done!")
        self.reInit()
        self.updateMeasurements()

    def saveTagsSimple(self, nameAddition = ""):
        self.tagger.reset()
        channels = [self.ui.channelA.value(), self.ui.channelB.value(), self.ui.channelC.value(),
                    self.ui.channelD.value()]
        self.active_channels = []
        if channels[0] != 0:
            #self.tagger.setInputDelay(channels[0], self.ui.delayA.value())
            self.tagger.setTriggerLevel(channels[0], self.ui.triggerA.value())
            self.tagger.setDeadtime(channels[0], int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setDeadtime(channels[0]*-1, int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setTestSignal(channels[0], self.ui.testsignalA.isChecked())
            self.active_channels.append(channels[0])

        if channels[1] != 0:
            #self.tagger.setInputDelay(channels[1], self.ui.delayB.value())
            self.tagger.setTriggerLevel(channels[1], self.ui.triggerB.value())
            self.tagger.setDeadtime(channels[1], int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setDeadtime(channels[1]*-1, int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setTestSignal(channels[1], self.ui.testsignalB.isChecked())
            self.active_channels.append(channels[1])

        if channels[2] != 0:
            #self.tagger.setInputDelay(channels[2], self.ui.delayC.value())
            self.tagger.setTriggerLevel(channels[2], self.ui.triggerC.value())
            self.tagger.setDeadtime(channels[2], int(self.ui.deadTimeC.value() * 1000))
            self.tagger.setDeadtime(channels[2] * -1, int(self.ui.deadTimeB.value() * 1000))
            self.active_channels.append(channels[2])

        if channels[3] != 0:
            #self.tagger.setInputDelay(channels[3], self.ui.delayD.value())
            self.tagger.setTriggerLevel(channels[3], self.ui.triggerD.value())
            self.tagger.setDeadtime(channels[3], int(self.ui.deadTimeD.value() * 1000))
            self.tagger.setDeadtime(channels[3] * -1, int(self.ui.deadTimeB.value() * 1000))
            self.active_channels.append(channels[3])

        #self.a_combined = AverageChannel(self.tagger, -3, (-3, -4))
        #self.b_combined = AverageChannel(self.tagger, -6, (-6, -7, -8))
        self.tagger.setEventDivider(18, self.clock_divider)
        file = str(self.ui.saveFileName.text()) + str(nameAddition) + ".ttbin"
        print("saving ", file, " in working directory")
        print("starting save")
        file_writer = FileWriter(self.tagger, file, self.active_channels)
        #file_writer = FileWriter(self.tagger, file, [channels[0],channels[1]])
        sleep(self.ui.saveTime.value())  # write for some time
        file_writer.stop()
        print("ending save")
        self.reInit()
        self.updateMeasurements()


    def _set_source_voltage_robustly(self, voltage):
        """Attempts to set the voltage on the SIM928 source, handling disconnections."""
        max_retries = 1 # Try original port, then the other port once
        retries = 0
        set_voltage_success = False
        while retries <= max_retries and not set_voltage_success:
            try:
                if not self.source: # Check if source was initialized
                     print("Error: SIM928 source not available.")
                     return False # Cannot set voltage if source is not available

                print(f"Attempting to set voltage {voltage:.3f} V on {self.source_port}...")
                self.source.setVoltage(voltage)
                set_voltage_success = True
                print(f"Successfully set Voltage: {voltage:.3f} V")
                return True # Voltage set successfully

            except (serial.SerialException, termios.error) as e: # Catch both SerialException and termios.error
                print(f"Serial/OS error setting voltage on {self.source_port}: {e}")
                retries += 1
                if retries > max_retries:
                    print("Max retries reached for setting voltage.")
                    break # Exit the while loop

                print("Attempting to reconnect to alternative port...")
                try:
                    self.source.disconnect()
                except Exception as disconnect_e:
                    print(f"Note: Error during disconnect (may already be closed): {disconnect_e}")

                # Cycle port
                current_index = self.possible_ports.index(self.source_port)
                next_index = (current_index + 1) % len(self.possible_ports)
                self.source_port = self.possible_ports[next_index]
                print(f"Trying port: {self.source_port}")

                try:
                    # Recreate and connect
                    self.source = sim928(self.source_port, self.source_gpib_addr, self.source_slot)
                    self.source.connect()
                    self.source.turnOn()
                    print(f"Successfully reconnected to {self.source_port}.")
                    # Retry setting voltage in the next loop iteration
                except (serial.SerialException, termios.error) as e2: # Also catch termios error on reconnect
                    print(f"Reconnect failed on {self.source_port}: {e2}")
                    # If reconnect fails, break the retry loop for this bias point
                    break
                except Exception as general_e:
                    print(f"Unexpected error during reconnect: {general_e}")
                    break

            except Exception as general_e:
                 print(f"Unexpected error setting voltage: {general_e}")
                 # Decide how to handle unexpected errors, e.g., skip point
                 break # Exit retry loop

        # If loop finishes without success
        return False


    def _shutdown_instruments(self, params):
        """
        Shutdown instruments based on YAML configuration
        """
        try:
            shutdown_config = params.get('turn_off_after_pcr', {})
            print("Shutting down instruments...")
            
            # Turn off SIM928
            if shutdown_config.get('sim928', False):
                try:
                    if self.source is not None:
                        self.source.turnOff()
                        print("SIM928 turned off successfully")
                    else:
                        print("SIM928 not available for shutdown")
                except Exception as e:
                    print(f"Error turning off SIM928: {e}")
            
            # Turn off cryo_amp (channel 3 of power supply)
            if shutdown_config.get('cryo_amp', False):
                try:
                    if self.power_supply is not None:
                        self.power_supply.output_off(3)
                        print("Cryo amp (channel 3) turned off successfully")
                    else:
                        print("Power supply not available for cryo amp shutdown")
                except Exception as e:
                    print(f"Error turning off cryo amp: {e}")
            
            # Turn off thermal_source (channel 2 of function generator)
            if shutdown_config.get('thermal_source', False):
                try:
                    if self.function_gen is not None:
                        self.function_gen.set_output(2, 0)
                        print("Thermal source (channel 2) turned off successfully")
                    else:
                        print("Function generator not available for thermal source shutdown")
                except Exception as e:
                    print(f"Error turning off thermal source: {e}")
                    
        except Exception as e:
            print(f"Error in instrument shutdown: {e}")

    def PCR(self):
        import yaml
        import os.path
        
        params_file = "./PCR_multi_trigger_params.yml"
        
        # Always load parameters from YAML file
        if not os.path.exists(params_file):
            print(f"Error: Parameter file '{params_file}' not found.")
            return # Exit if file doesn't exist
            
        try:
            with open(params_file, 'r') as file:
                params = yaml.safe_load(file)
            print("Parameters loaded from file.")

            self.fudge_factor = params['fudge_factor']
            
            # Extract common parameters
            Start = params['voltage']['start']
            Stop = params['voltage']['stop']
            Step = params['voltage']['step']
            int_time_sec = params['integration_time']
            measurement_type = params.get('measurement_type', 'filtered_pcr').lower()
            
            print(f"Measurement type: {measurement_type}")
            
            # Extract measurement-specific parameters
            if measurement_type == 'filtered_pcr':
                trigger_levels = params['filtered_PCR']['trigger_levels']
                if not isinstance(trigger_levels, list) or not trigger_levels:
                    print("Error: 'trigger_levels' in filtered_PCR YAML must be a non-empty list.")
                    return
                num_trigger_levels = len(trigger_levels)
                print(f"Using {num_trigger_levels} trigger levels: {trigger_levels}")
            elif measurement_type == 'dcr':
                # Support both old single trigger_level and new trigger_levels list
                if 'trigger_levels' in params['DCR']:
                    trigger_levels = params['DCR']['trigger_levels']
                    if not isinstance(trigger_levels, list) or not trigger_levels:
                        print("Error: 'trigger_levels' in DCR YAML must be a non-empty list.")
                        return
                elif 'trigger_level' in params['DCR']:
                    # Backward compatibility with single trigger level
                    trigger_level = params['DCR']['trigger_level']
                    trigger_levels = [str(trigger_level)]
                    print("Using single DCR trigger level (backward compatibility mode)")
                else:
                    print("Error: DCR section must contain either 'trigger_levels' (list) or 'trigger_level' (single value).")
                    return
                
                num_trigger_levels = len(trigger_levels)
                print(f"Using {num_trigger_levels} DCR trigger levels: {trigger_levels}")
            else:
                print(f"Error: Unknown measurement type '{measurement_type}'. Must be 'filtered_pcr' or 'dcr'.")
                return

        except (yaml.YAMLError, KeyError, TypeError) as e:
            print(f"Error loading or parsing parameters from '{params_file}': {e}")
            return # Exit on error
        except Exception as e:
             print(f"An unexpected error occurred while loading parameters: {e}")
             return

        # Get save location first using file dialog
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save PCR Curve Data',
            directory='PCR_Curve_Data.csv',  # default name
            filter='CSV Files (*.csv);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )
        
        # If user cancels, exit the function
        if not filename:
            print("Save operation cancelled.")
            return
        
        # Ensure we have a .csv extension for the CSV file
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        # Create the base filename for the PNG (remove .csv and we'll add .png later)
        png_filename = filename[:-4] if filename.lower().endswith('.csv') else filename
        png_filename += '.png'
        
        # V_pp = 0.090  # in V # This seems unused, consider removing if not needed elsewhere
        offset = numpy.arange(Start, Stop + Step, Step) # Corrected range to include Stop properly
        # Ensure Stop is included if the step doesn't divide the range perfectly
        if not numpy.isclose(offset[-1], Stop):
             offset = numpy.append(offset, Stop)

        I_det = []
        int_time = int(float(int_time_sec)*1e12)
        print("Integration time (ps): ", int_time)
        
        # For DCR measurements, calculate number of bins (0.1 second each)
        bin_duration = 0.1  # 0.1 second per bin (used for all measurements)
        if measurement_type == 'dcr':
            num_bins = int(int_time_sec / bin_duration)  # Total number of bins
            bin_time_ps = int(bin_duration * 1e12)  # 0.1 second in picoseconds
            print(f"DCR measurement: {num_bins} bins of {bin_duration} seconds each")
        else:
            num_bins = 1
            bin_time_ps = int_time
        
        Counts = [[] for _ in range(num_trigger_levels)]  # Store counts for each trigger level
        Counts_off = [[] for _ in range(num_trigger_levels)] if measurement_type == 'filtered_pcr' else None # Store dark counts for filtered PCR
        
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, ax = plt.subplots() # Use ax for plotting
        x_vals = []
        print(f"Estimated completion time (minutes): {round((int_time_sec * num_trigger_levels + 0.2 * num_trigger_levels)/60 * len(offset), 2)}") # Adjusted estimate
        
        # Determining Bias at the Detector
        for v_offset in offset:
            # Assuming 1.02 MOhm series resistance for current calculation
            # Verify this calculation is correct for your setup
            I_det.append(((v_offset / 1.02e6) * 1e6).round(4))  # in uA
    
        I_b = numpy.asarray(I_det, dtype='float')
    
        for i in range(len(I_b)): # Iterate through bias currents/voltages
            # --- Set Voltage Robustly ---
            set_voltage_success = self._set_source_voltage_robustly(offset[i])
            # --- End Set Voltage Robustly ---

            # If setting voltage failed after retries, skip the rest of the loop for this bias
            if not set_voltage_success:
                print(f"Skipping measurements for bias voltage index {i} (Voltage: {offset[i]:.3f} V) due to connection issues.")
                # Ensure lists have consistent lengths if skipping, e.g., append NaN or skip appending later
                # For simplicity here, we just continue to the next bias voltage.
                # Depending on plotting logic, you might need placeholder values.

                # Append NaN or placeholder to keep plot arrays aligned
                x_vals.append(I_b[i]) # Keep x-axis value
                for j in range(num_trigger_levels):
                    if measurement_type == 'filtered_pcr':
                        Counts[j].append(numpy.nan) # Use NaN for missing data
                        if Counts_off is not None:
                            Counts_off[j].append(numpy.nan)
                    else:  # dcr measurement
                        # For DCR, append an array of NaN values to match the expected structure
                        Counts[j].append(numpy.full(num_bins, numpy.nan))

                continue # Skip to the next value of i in the outer loop

            current_bias_ua = I_b[i]
            x_vals.append(current_bias_ua) # Append current bias value for plotting
            # print(f"\nBias: {current_bias_ua} uA") # Simplified print

            # Set up measurement channels based on measurement type
            if measurement_type == 'filtered_pcr':
                cr_on = Counter(self.tagger, [self.filtered_on.getChannel()], binwidth=int_time, n_values=1)
                cr_off = Counter(self.tagger, [self.filtered_off.getChannel()], binwidth=int_time, n_values=1)
                cr_dcr = None
            else:  # dcr measurement
                cr_dcr = Counter(self.tagger, [self.active_channels[2]], binwidth=bin_time_ps, n_values=num_bins)
                cr_on = None
                cr_off = None

            for j, trigger_level in enumerate(trigger_levels): # Iterate through trigger levels
                trigger_level_float = float(trigger_level) # Ensure it's float
                self.tagger.setTriggerLevel(self.ui.channelC.value(), trigger_level_float)
                # Optional: Add a small delay after setting trigger level if needed
                # time.sleep(0.05)
                # Verify trigger level was set (optional)
                # actual_trigger = self.tagger.getTriggerLevel(self.ui.channelC.value())
                # print(f"  Trigger Level {j+1}/{num_trigger_levels}: Set={trigger_level_float:.3f} V")#, Actual={actual_trigger:.3f} V")
                print(f"  Measuring Trigger Level: {trigger_level_float:.3f} V")

                time.sleep(0.2) # Delay before measurement
    
                if measurement_type == 'filtered_pcr' and cr_on is not None and cr_off is not None:
                    # Filtered PCR measurement
                    # Start measurements
                    cr_on.startFor(int_time, clear=True)
                    cr_off.startFor(int_time, clear=True)
                    
                    # Wait for measurements to complete
                    cr_on.waitUntilFinished()
                    cr_off.waitUntilFinished()
        
                    clicks_on = cr_on.getData() 
                    clicks_off = cr_off.getData() 

                    count = (clicks_on[0][0]/ (self.ratio_on*int_time_sec)) - (clicks_off[0][0]/ (self.ratio_off*int_time_sec)) # Calculate counts for this trigger level
                    dark_count = (clicks_off[0][0]/ (self.ratio_off*int_time_sec))

                    Counts[j].append(count)
                    if Counts_off is not None:
                        Counts_off[j].append(dark_count) # Store dark counts for this trigger level
                    print(f"    Signal Counts: {count}, Dark Counts: {dark_count}")
                    
                elif measurement_type == 'dcr' and cr_dcr is not None:
                    # DCR measurement - direct count on active_channels[2] with multiple bins
                    cr_dcr.startFor(int_time, clear=True)
                    cr_dcr.waitUntilFinished()
                    
                    clicks_data = cr_dcr.getData()  # This returns a 2D array: [channels][bins]
                    bin_counts = clicks_data[0]  # Get data for first (and only) channel
                    
                    # Convert to counts per second for each bin
                    bin_counts_per_sec = bin_counts / bin_duration
                    
                    # Store the entire array of bin counts
                    Counts[j].append(bin_counts_per_sec)
                    
                    # Calculate average for printing
                    avg_count = numpy.mean(bin_counts_per_sec)
                    print(f"    DCR Counts (avg): {avg_count:.2f} Hz, {num_bins} bins")

            # --- Plotting Update ---
            ax.clear() # Clear previous plot data for redraw
            for j in range(num_trigger_levels):
                color = colors[j % len(colors)] # Cycle through colors
                
                if measurement_type == 'filtered_pcr':
                    trigger_label = f'TL {j+1}: {trigger_levels[j]}'
                    dark_label = f'Dark TL {j+1}'
                    
                    # Filter out NaN values for plotting lines/scatter
                    valid_indices = ~numpy.isnan(Counts[j])
                    valid_x = numpy.array(x_vals)[valid_indices]
                    valid_counts = numpy.array(Counts[j])[valid_indices]
                    if Counts_off is not None:
                        valid_counts_off = numpy.array(Counts_off[j])[valid_indices]
                    else:
                        valid_counts_off = numpy.array([])

                    # Plot signal counts (scatter and line) - only plot valid points
                    ax.scatter(valid_x, valid_counts, color=color, s=10, label=trigger_label if i == len(I_b) - 1 else None) # Label only on last iteration
                    ax.plot(valid_x, valid_counts, color=color)

                    # Plot dark counts (line, dashed) - only plot valid points
                    if len(valid_counts_off) > 0:
                        ax.plot(valid_x, valid_counts_off, color=color, linestyle='--', label=dark_label if i == len(I_b) - 1 else None) # Label only on last iteration
                    
                    ax.set_title("Gated PCR Curve")
                    
                else:  # dcr measurement
                    dcr_label = f'DCR TL: {trigger_levels[j]}'
                    
                    # For DCR, Counts[j] contains arrays, so we need to calculate averages for plotting
                    avg_counts = []
                    for count_array in Counts[j]:
                        if isinstance(count_array, numpy.ndarray):
                            avg_counts.append(numpy.mean(count_array))
                        else:
                            avg_counts.append(count_array if not numpy.isnan(count_array) else numpy.nan)
                    
                    # Filter out NaN values for plotting
                    valid_indices = ~numpy.isnan(avg_counts)
                    valid_x = numpy.array(x_vals)[valid_indices]
                    valid_avg_counts = numpy.array(avg_counts)[valid_indices]

                    # Plot DCR counts (using averages)
                    ax.scatter(valid_x, valid_avg_counts, color=color, s=10, label=dcr_label if i == len(I_b) - 1 else None)
                    ax.plot(valid_x, valid_avg_counts, color=color)

                    ax.set_title("DCR Curve")
            
            ax.set_xlabel("Bias Current (uA)")
            ax.set_ylabel("Counts")
            ax.grid(True) # Add grid
            plt.draw()
            plt.pause(0.1) # Shorter pause

        ax.legend(loc='best')

        # Save the final plot figure as PNG
        try:
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {png_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # Show the final plot (optional, can be blocking)
        # plt.show()
    
        print(f'Finished {measurement_type.upper()} Curve Measurement.')
    
        # --- CSV Writing Update ---
        try:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Dynamically generate header based on measurement type
                header = ['Bias_Current']
                if measurement_type == 'filtered_pcr':
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'Counts_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'DCounts_TL{j+1}({tl})')
                else:  # dcr - create columns for each bin
                    for j, tl in enumerate(trigger_levels):
                        for bin_idx in range(num_bins):
                            header.append(f'DCR_TL{j+1}({tl})_Bin{bin_idx+1}')
                        
                csvwriter.writerow(header)

                # Write data rows
                for row_idx in range(len(I_b)):
                    row = [I_b[row_idx]]
                    
                    if measurement_type == 'filtered_pcr':
                        # Append signal counts for this bias
                        for tl_idx in range(num_trigger_levels):
                            # Handle potential NaN values when writing to CSV (replace with empty string or specific value)
                            count_val = Counts[tl_idx][row_idx]
                            row.append(count_val if not numpy.isnan(count_val) else '')
                        
                        # Append dark counts only for filtered PCR
                        if Counts_off is not None:
                            for tl_idx in range(num_trigger_levels):
                                dcount_val = Counts_off[tl_idx][row_idx]
                                row.append(dcount_val if not numpy.isnan(dcount_val) else '')
                    
                    else:  # dcr - write all bins for each trigger level
                        for tl_idx in range(num_trigger_levels):
                            count_data = Counts[tl_idx][row_idx]
                            if isinstance(count_data, numpy.ndarray):
                                # Write each bin value
                                for bin_val in count_data:
                                    row.append(bin_val if not numpy.isnan(bin_val) else '')
                            else:
                                # Handle case where it's a single value (e.g., NaN for failed measurements)
                                for bin_idx in range(num_bins):
                                    row.append(count_data if not numpy.isnan(count_data) else '')
                    
                    csvwriter.writerow(row)

            print(f"CSV data saved as: {filename}")
        except Exception as e:
            print(f"Error writing CSV file: {e}")
        
        time.sleep(0.5) 
        # Shutdown instruments based on YAML configuration
        self._shutdown_instruments(params)

    def saveTrace(self):
        self.tagger.reset()
        channels = [self.ui.channelA.value(), self.ui.channelB.value(), self.ui.channelC.value(),
                    self.ui.channelD.value()]

        if channels[0] != 0:
            self.tagger.setInputDelay(channels[0], self.ui.delayA.value())
            self.tagger.setTriggerLevel(channels[0], self.ui.triggerA.value())
            self.tagger.setDeadtime(channels[0], int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setTestSignal(channels[0], self.ui.testsignalA.isChecked())

        if channels[1] != 0:
            self.tagger.setInputDelay(channels[1], self.ui.delayB.value())
            self.tagger.setTriggerLevel(channels[1], self.ui.triggerB.value())
            self.tagger.setDeadtime(channels[1], int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setTestSignal(channels[1], self.ui.testsignalB.isChecked())

        if channels[2] != 0:
            self.tagger.setInputDelay(channels[2], self.ui.delayC.value())
            self.tagger.setTriggerLevel(channels[2], self.ui.triggerC.value())
            self.tagger.setDeadtime(channels[2], int(self.ui.deadTimeC.value() * 1000))

        if channels[3] != 0:
            self.tagger.setInputDelay(channels[3], self.ui.delayD.value())
            self.tagger.setTriggerLevel(channels[3], self.ui.triggerD.value())
            self.tagger.setDeadtime(channels[3], int(self.ui.deadTimeD.value() * 1000))

        #self.a_combined = AverageChannel(self.tagger, -2, (-2, -3, -4, -5,-6))
        self.tagger.sync()



        start = float(input("start voltage: "))
        end = float(input("end voltage: "))
        res = int(input("input vertical resolution: "))
        ch = int(input("input channel number (starting from zero is A, B is 2, etc.)"))

        self.correlation = Histogram(
            self.tagger,
            self.active_channels[ch],
            # self.a_combined.getChannel(),
            self.active_channels[0],
            self.ui.correlationBinwidth.value(),
            self.ui.correlationBins.value()
        )


        self.scopeBlock = numpy.zeros((res, self.ui.correlationBins.value()))
        trigger_levels = [(i*(end - start)/res) + start for i in range(len(self.scopeBlock))]
        trigger_levels.reverse()

        self.correlation.stop()
        self.correlation.clear()
        sleep(1)
        for i in range(len(self.scopeBlock)):
            self.tagger.setTriggerLevel(channels[ch], trigger_levels[i])
            print("Voltage: ", round(self.tagger.getTriggerLevel(channels[ch]),4))
            self.tagger.sync()
            sleep(.1)
            #self.correlation.clear()
            self.correlation.start()
            sleep(0.1)
            self.correlation.stop()

            self.buffer = self.correlation.getData()
            self.scopeBlock[i] = self.buffer - self.buffer_old
            #print(numpy.sum(self.buffer - self.buffer_old))
            # buffer is used in next loop for subtraction
            self.buffer_old = self.buffer


        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(self.scopeBlock + 1, norm=LogNorm(),extent = [0, self.ui.correlationBins.value(), start, end])
        #ax.set_aspect('equal')
        ax.set_aspect('auto')
        plt.show()
        sleep(0.5)  # write for some time
        print("done!")

        #self.reInit()
        self.updateMeasurements()
        R = input("Save numpy array? (y/n): ")
        if R == 'y' or R == 'Y':
            name = input("Input save Name: ")
            numpy.save(name, self.scopeBlock + 1)



    def Hist2D(self):

        self.tagger.reset()

        channels = [self.ui.channelA.value(), self.ui.channelB.value(), self.ui.channelC.value(),
                    self.ui.channelD.value()]

        if channels[0] != 0:
            self.tagger.setInputDelay(channels[0], self.ui.delayA.value())
            self.tagger.setTriggerLevel(channels[0], self.ui.triggerA.value())
            self.tagger.setDeadtime(channels[0], int(self.ui.deadTimeA.value() * 1000))
            self.tagger.setTestSignal(channels[0], self.ui.testsignalA.isChecked())

        if channels[1] != 0:
            self.tagger.setInputDelay(channels[1], self.ui.delayB.value())
            self.tagger.setTriggerLevel(channels[1], self.ui.triggerB.value())
            self.tagger.setDeadtime(channels[1], int(self.ui.deadTimeB.value() * 1000))
            self.tagger.setTestSignal(channels[1], self.ui.testsignalB.isChecked())

        if channels[2] != 0:
            self.tagger.setInputDelay(channels[2], self.ui.delayC.value())
            self.tagger.setTriggerLevel(channels[2], self.ui.triggerC.value())
            self.tagger.setDeadtime(channels[2], int(self.ui.deadTimeC.value() * 1000))

        if channels[3] != 0:
            self.tagger.setInputDelay(channels[3], self.ui.delayD.value())
            self.tagger.setTriggerLevel(channels[3], self.ui.triggerD.value())
            self.tagger.setDeadtime(channels[3], int(self.ui.deadTimeD.value() * 1000))

        self.tagger.sync()

        self.hist2D = Histogram2D(
            self.tagger,
            self.active_channels[0],
            self.active_channels[1],
            self.active_channels[2],
            self.ui.correlationBinwidth.value(),
            self.ui.correlationBinwidth.value(),
            self.ui.correlationBins.value(),
            self.ui.correlationBins.value()
        )

        print(self.active_channels[0])
        print(self.active_channels[1])
        print(self.active_channels[2])


        self.hist2D.startFor(int(3e12)) #1 second

        while self.hist2D.isRunning():
            sleep(0.1)

        img = self.hist2D.getData()

        print(numpy.max(img))
        print(numpy.min(img))
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('2DHist')
        plt.imshow(img + 1, norm=LogNorm())
        # ax.set_aspect('equal')
        ax.set_aspect('equal')
        plt.show()




    def saveClicked(self):
        '''Handler for the save action button'''

        # Ask for a filename
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save to File',
            directory='CoincidenceExampleData.txt',  # default name
            filter='All Files (*);;Text Files (*.txt)',
            options=QFileDialog.DontUseNativeDialog
        )

        # And write all results to disk
        if filename:
            with open(filename, 'w') as f:
                f.write('Input channel A: %d\n' % self.ui.channelA.value())
                f.write('Input channel B: %d\n' % self.ui.channelB.value())
                f.write('Input channel C: %d\n' % self.ui.channelC.value())
                f.write('Input channel D: %d\n' % self.ui.channelD.value())
                f.write('Input delay A: %d ps\n' % self.ui.delayA.value())
                f.write('Input delay B: %d ps\n' % self.ui.delayB.value())
                f.write('Input delay C: %d ps\n' % self.ui.delayC.value())
                f.write('Input delay D: %d ps\n' % self.ui.delayD.value())
                f.write('Trigger level A: %.3f V\n' % self.ui.triggerA.value())
                f.write('Trigger level B: %.3f V\n' % self.ui.triggerB.value())
                f.write('Trigger level C: %.3f V\n' % self.ui.triggerC.value())
                f.write('Trigger level D: %.3f V\n' % self.ui.triggerD.value())
                f.write('Test signal A: %d\n' %
                        self.ui.testsignalA.isChecked())
                f.write('Test signal B: %d\n' %
                        self.ui.testsignalB.isChecked())

                f.write('Coincidence window: %d ps\n' %
                        self.ui.coincidenceWindow.value())
                f.write('Correlation bin width: %d ps\n' %
                        self.ui.correlationBinwidth.value())
                f.write('Correlation bins: %d\n\n' %
                        self.ui.correlationBins.value())

                f.write('Counter data:\n%s\n\n' %
                        self.counter.getData().__repr__())
                f.write('Correlation data:\n%s\n\n' %
                        self.correlation.getData().__repr__())

    def resizeEvent(self, event):
        '''Handler for the resize events to update the plots'''
        self.fig.tight_layout()
        self.canvas.draw()

    def draw(self):
        '''Handler for the timer event to update the plots'''
        if self.running:
            # Counter
            #data = self.counter.getData() * self.getCouterNormalizationFactor()
            histblock_depth = int(self.ui.IntTime.value()*5)
            
            if self.BlockIndex >= histblock_depth:
                # Check if saving was requested and histBlock is now full
                if self.save_requested:
                    print("Data collection complete. Saving histogram...")
                    self._save_histogram_data()
                
                self.BlockIndex = 0

            data = self.counter.getData() * self.getCouterNormalizationFactor()
            #print("length of data", len(data))
            #print("###########")
            for data_line, plt_counter in zip(data, self.plt_counter): # loop though coincidences, Ch1, Ch2
                plt_counter.set_ydata(data_line)
            self.counterAxis.relim()
            self.counterAxis.autoscale_view(True, True, True)


            index = self.correlation.getIndex()[self.masked_hist_bins:]

            q = self.correlation.getData()[self.masked_hist_bins:]
            self.histBlock[self.BlockIndex] = q
            #print(numpy.sum(q))

            if self.ui.IntType.currentText() == "Discrete":
                if self.BlockIndex == 0:
                    self.persistentData = numpy.sum(self.histBlock, axis=0)
                else:
                    if self.IntType == "Rolling":

                        # first time changing from Rolling to Discrete
                        self.persistentData = numpy.sum(self.histBlock, axis=0)
                        self.BlockIndex = 1
                        self.IntType = "Discrete"
                currentData = self.persistentData
            else:
                    currentData = numpy.sum(self.histBlock, axis=0)
            #print(numpy.sum(currentData))
            self.IntType = self.ui.IntType.currentText()


            #Histdata = self.correlation.getData()
            # display data averaged for one second
            self.plt_correlation[0].set_ydata(currentData)
            #self.plt_gauss[0].set_ydata(gauss)
            self.correlationAxis.relim()
            #if self.BlockIndex == 0:
            self.correlationAxis.autoscale_view(True, True, True)
                #self.correlation.clear()
            #self.correlationAxis.legend(['measured correlation', '$\mu$=%.1fps, $\sigma$=%.1fps' % (
            #    offset, stdd), 'coincidence window'])
            self.canvas.draw()
            self.correlation.clear()

            self.BlockIndex = self.BlockIndex + 1


# If this file is executed, initialize PySide2, create a TimeTagger object, and show the UI
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)


    # used to check if JPL swabian supports high res. It does not.
    tagger = createTimeTagger(resolution = Resolution.HighResC)
    # tagger = createTimeTagger()

    # If you want to include this window within a bigger UI,
    # just copy these two lines within any of your handlers.
    window = CoincidenceExample(tagger)
    window.show()

    app.exec_()

    freeTimeTagger(tagger)
