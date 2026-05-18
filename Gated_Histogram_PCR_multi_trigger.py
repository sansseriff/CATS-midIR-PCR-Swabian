# from snspd_measure.inst.teledyneT3AFG200 import teledyneT3AFG200

# PySide2 for the UI
from PySide2.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QLabel, QGroupBox, QMessageBox
from PySide2.QtCore import QTimer, QThread, Signal, QObject
from PySide2.QtGui import QPalette, QColor

from snsphd.viz import phd_grid_style

from snspd_measure.inst.sim900 import sim928
from snspd_measure.inst.hp8114A import hp8114A

# Import client instruments
from client_teledyneT3AFG200 import ClientTeledyneT3AFG200
from client_keysightE36312A import ClientKeysightE36312A

# matplotlib for the plots, including its Qt backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import cm
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
from TimeTagger import Coincidences, Histogram2D, Counter, Correlation, StartStop, createTimeTagger, freeTimeTagger, Histogram, FileWriter, FileReader, TT_CHANNEL_FALLING_EDGES, Resolution, DelayedChannel, GatedChannel, Countrate, CHANNEL_UNUSED, SynchronizedMeasurements
from time import sleep
import time

import json
import csv
import os.path

import serial # Import serial for exception handling
import termios # Import termios for catching specific OS error

# from awgClient import AWGClient


def compute_pcr_colors(num_lines, cmap_name='plasma', crop_start=0.1, crop_end=0.9):
    """Return a list of RGBA colors for PCR plots.

    Colors are sampled from a cropped segment of the specified colormap
    (default "plasma") using linear interpolation so that the first and
    last lines are not too dark or too bright.

    Parameters
    ----------
    num_lines : int
        Number of colors required (e.g. number of trigger levels).
    cmap_name : str
        Name of the Matplotlib colormap to use.
    crop_start : float
        Start of the colormap segment (0–1) to use.
    crop_end : float
        End of the colormap segment (0–1) to use.
    """
    if num_lines <= 0:
        return []

    crop_start = max(0.0, min(1.0, crop_start))
    crop_end = max(0.0, min(1.0, crop_end))
    if crop_end < crop_start:
        crop_start, crop_end = crop_end, crop_start

    cmap = cm.get_cmap(cmap_name)

    if num_lines == 1:
        t_values = [0.5]
    else:
        t_values = numpy.linspace(crop_start, crop_end, num_lines)

    return [cmap(t) for t in t_values]


class PCRWorker(QObject):
    """Background worker that runs the long PCR scan in a QThread.

    It cooperatively checks for cancellation only between bias points,
    ensuring that the current bias finishes all trigger levels before
    stopping. It returns the collected data to the GUI for saving.
    """

    # Can emit either log strings or structured dicts with
    # partial data for live plotting.
    progress = Signal(object)
    finished_ok = Signal(dict)
    error = Signal(str)
    cancel_ack = Signal()

    def __init__(self, tagger, params, filename, png_filename,
                 ratio_on, ratio_on_dcr, ratio_off,
                 filtered_on_channel, filtered_on_dcr_channel, filtered_off_channel,
                 active_snspd_channel,
                 set_bias_fn,
                 channelC,
                 parent=None):
        super(PCRWorker, self).__init__(parent)
        self.tagger = tagger
        self.params = params
        self.filename = filename
        self.png_filename = png_filename
        self.ratio_on = ratio_on
        self.ratio_on_dcr = ratio_on_dcr
        self.ratio_off = ratio_off
        self.filtered_on_channel = filtered_on_channel
        self.filtered_on_dcr_channel = filtered_on_dcr_channel
        self.filtered_off_channel = filtered_off_channel
        self.active_snspd_channel = active_snspd_channel

        # Callable that sets the SIM928 bias voltage for a given offset,
        # mirroring _set_source_voltage_robustly from the GUI thread.
        self.set_bias_fn = set_bias_fn
        # Channel used for trigger level sweeps (was ui.channelC in original PCR).
        self.channelC = channelC

        self._cancel_requested = False
        self._cancel_ack_emitted = False

    def request_cancel(self):
        """Mark that a cancel was requested.

        The worker will only actually stop after finishing the current
        bias point (i.e. after the inner trigger-level loop).
        """
        self._cancel_requested = True

    def _check_cancel_after_bias(self):
        """Check for cancellation between bias points.

        Emits cancel_ack once, then instructs caller to stop looping
        by returning True.
        """
        if self._cancel_requested:
            if not self._cancel_ack_emitted:
                self._cancel_ack_emitted = True
                self.cancel_ack.emit()
                self.progress.emit("Cancel requested – finishing current bias and saving data…")
            return True
        return False

    @staticmethod
    def _derive_averages(measurement_type, num_trigger_levels, num_bias_pts,
                         fudge_factor, int_time_sec, ratio_on, ratio_on_dcr, ratio_off,
                         Acc_clicks_on, Acc_clicks_on_dcr, Acc_clicks_off,
                         Acc_dcr, Acc_n, num_bins):
        """Derive list-of-lists Counts/Counts_on_dcr/Counts_off/Clicks from accumulators.

        Computes running-average rates from accumulated raw click totals
        across cycles. Returns
        (Counts, Counts_on_dcr, Counts_off, Clicks_on, Clicks_on_dcr, Clicks_off)
        in the same list-of-lists format the GUI and CSV writer expect.

        For filtered_pcr the averaged rate is computed as:
            total_integration = int_time_sec * n_cycles_contributing
            signal = on_clicks / (ratio_on_eff * total_integration)
                   - off_clicks / (ratio_off_eff * total_integration)
            dark   = off_clicks / (ratio_off_eff * total_integration)
        This is mathematically equivalent to averaging the per-cycle rates.

        Clicks_on / Clicks_off report the *accumulated* raw click totals
        (summed across all contributing cycles).
        """
        import numpy

        ratio_on_eff = ratio_on * fudge_factor
        ratio_on_dcr_eff = ratio_on_dcr * fudge_factor
        ratio_off_eff = ratio_off / fudge_factor

        if measurement_type == 'filtered_pcr':
            Counts     = [[] for _ in range(num_trigger_levels)]
            Counts_on_dcr = [[] for _ in range(num_trigger_levels)]
            Counts_off = [[] for _ in range(num_trigger_levels)]
            Clicks_on  = [[] for _ in range(num_trigger_levels)]
            Clicks_on_dcr = [[] for _ in range(num_trigger_levels)]
            Clicks_off = [[] for _ in range(num_trigger_levels)]

            for j in range(num_trigger_levels):
                for i in range(num_bias_pts):
                    n = Acc_n[j, i]
                    if n == 0:
                        Counts[j].append(numpy.nan)
                        Counts_on_dcr[j].append(numpy.nan)
                        Counts_off[j].append(numpy.nan)
                        Clicks_on[j].append(numpy.nan)
                        Clicks_on_dcr[j].append(numpy.nan)
                        Clicks_off[j].append(numpy.nan)
                    else:
                        total_on  = Acc_clicks_on[j, i]
                        total_on_dcr = Acc_clicks_on_dcr[j, i]
                        total_off = Acc_clicks_off[j, i]
                        total_int = int_time_sec * n

                        signal = (total_on / (ratio_on_eff * total_int)) - (total_off / (ratio_off_eff * total_int))
                        signal_on_dcr = total_on_dcr / (ratio_on_dcr_eff * total_int)
                        dark   = total_off / (ratio_off_eff * total_int)

                        Counts[j].append(signal)
                        Counts_on_dcr[j].append(signal_on_dcr)
                        Counts_off[j].append(dark)
                        Clicks_on[j].append(total_on)
                        Clicks_on_dcr[j].append(total_on_dcr)
                        Clicks_off[j].append(total_off)

            return Counts, Counts_on_dcr, Counts_off, Clicks_on, Clicks_on_dcr, Clicks_off
        else:
            # DCR: average the accumulated bin-rate arrays
            Counts = [[] for _ in range(num_trigger_levels)]
            for j in range(num_trigger_levels):
                for i in range(num_bias_pts):
                    n = Acc_n[j, i]
                    if n == 0:
                        Counts[j].append(numpy.full(num_bins, numpy.nan))
                    else:
                        Counts[j].append(Acc_dcr[j, i, :] / n)
            return Counts, None, None, None, None, None

    def run(self):
        """Main worker entry point executed in the background thread."""
        try:
            import numpy
            import matplotlib.pyplot as plt
            import csv
            import time

            params = self.params
            fudge_factor = float(params.get('fudge_factor', 1.0))

            # Extract common parameters
            Start = params['voltage']['start']
            Stop = params['voltage']['stop']
            Step = params['voltage']['step']
            int_time_sec = params['integration_time']
            measurement_type = params.get('measurement_type', 'filtered_pcr').lower()
            num_cycles = int(params.get('cycles', 1))
            if num_cycles < 1:
                num_cycles = 1

            # Extract measurement-specific parameters
            if measurement_type == 'filtered_pcr':
                trigger_levels = params['filtered_PCR']['trigger_levels']
                num_trigger_levels = len(trigger_levels)
            elif measurement_type == 'dcr':
                if 'trigger_levels' in params['DCR']:
                    trigger_levels = params['DCR']['trigger_levels']
                else:
                    trigger_level = params['DCR']['trigger_level']
                    trigger_levels = [str(trigger_level)]
                num_trigger_levels = len(trigger_levels)
            else:
                self.error.emit(f"Unknown measurement type '{measurement_type}'")
                return

            # Voltage offsets
            offset = numpy.arange(Start, Stop + Step, Step)
            if not numpy.isclose(offset[-1], Stop):
                offset = numpy.append(offset, Stop)

            I_det = []
            for v_offset in offset:
                I_det.append(((v_offset / 1.02e6) * 1e6).round(4))
            I_b = numpy.asarray(I_det, dtype='float')

            int_time = int(float(int_time_sec) * 1e12)
            bin_duration = 0.1
            if measurement_type == 'dcr':
                num_bins = int(int_time_sec / bin_duration)
                bin_time_ps = int(bin_duration * 1e12)
            else:
                num_bins = 1
                bin_time_ps = int_time

            num_bias_pts = len(I_b)

            # ---- Accumulators: fixed-size arrays [trigger_level][bias_index] ----
            # These accumulate raw totals across cycles; derived rates are
            # recomputed from them so the average improves each cycle.

            Acc_clicks_on = None
            Acc_clicks_on_dcr = None
            Acc_clicks_off = None
            Acc_dcr = None

            if measurement_type == 'filtered_pcr':
                # Accumulated raw click totals across all cycles
                Acc_clicks_on  = numpy.zeros((num_trigger_levels, num_bias_pts))
                Acc_clicks_on_dcr = numpy.zeros((num_trigger_levels, num_bias_pts))
                Acc_clicks_off = numpy.zeros((num_trigger_levels, num_bias_pts))
                # Track how many valid cycles contributed to each (tl, bias) cell
                Acc_n = numpy.zeros((num_trigger_levels, num_bias_pts), dtype=int)
            else:
                # DCR: accumulate bin-count-per-sec arrays
                Acc_dcr = numpy.zeros((num_trigger_levels, num_bias_pts, num_bins))
                Acc_n = numpy.zeros((num_trigger_levels, num_bias_pts), dtype=int)

            # x_vals is just the bias current array (populated once)
            x_vals = list(I_b)

            # ---- Simple ETA estimation (now accounts for cycles) ----
            total_biases = num_bias_pts
            total_triggers = total_biases * num_trigger_levels * num_cycles
            per_trigger_est = float(int_time_sec) + 0.3
            total_estimated = max(1.0, total_triggers * per_trigger_est)
            start_time = time.time()
            eta_time = start_time + total_estimated

            self.progress.emit(
                f"Starting PCR measurement ({num_cycles} cycle{'s' if num_cycles > 1 else ''})… "
                f"Estimated duration: {total_estimated/60:.1f} min "
                f"(ETA {time.strftime('%H:%M:%S', time.localtime(eta_time))})"
            )

            cycles_completed = 0
            cancel_break = False

            # ============================================================
            # Outer loop over CYCLES
            # ============================================================
            for cycle in range(num_cycles):
                if cancel_break:
                    break

                self.progress.emit(f"── Cycle {cycle+1}/{num_cycles} ──")

                # Ramp bias back to start for each new cycle
                self.set_bias_fn(offset[0])

                # Loop over bias points
                for i in range(num_bias_pts):
                    set_voltage_success = self.set_bias_fn(offset[i])
                    if not set_voltage_success:
                        # Skip this bias point for this cycle (accumulators
                        # keep whatever they had from previous cycles).
                        self.progress.emit(
                            f"  Bias {i+1}/{num_bias_pts}: voltage set FAILED, skipping"
                        )
                        continue

                    if self._cancel_requested and not self._cancel_ack_emitted:
                        self.cancel_ack.emit()
                        self._cancel_ack_emitted = True
                        self.progress.emit("Cancel requested – finishing current bias…")

                    current_bias_ua = I_b[i]
                    self.progress.emit(
                        f"  Cycle {cycle+1}/{num_cycles}, Bias {i+1}/{num_bias_pts}, "
                        f"I = {current_bias_ua:.3f} µA"
                    )

                    # Configure counters for this bias
                    if measurement_type == 'filtered_pcr':
                        cr_on = Counter(self.tagger, [self.filtered_on_channel], binwidth=int_time, n_values=1)
                        cr_on_dcr = Counter(self.tagger, [self.filtered_on_dcr_channel], binwidth=int_time, n_values=1)
                        cr_off = Counter(self.tagger, [self.filtered_off_channel], binwidth=int_time, n_values=1)
                        cr_dcr = None
                    else:
                        cr_dcr = Counter(self.tagger, [self.active_snspd_channel], binwidth=bin_time_ps, n_values=num_bins)
                        cr_on = None
                        cr_on_dcr = None
                        cr_off = None

                    # Inner loop over trigger levels – always complete
                    for j, trigger_level in enumerate(trigger_levels):
                        trigger_level_float = float(trigger_level)
                        self.tagger.setTriggerLevel(self.channelC, trigger_level_float)
                        self.progress.emit(
                            f"    Trigger {j+1}/{num_trigger_levels}: {trigger_level_float:.3f} V"
                        )

                        time.sleep(0.2)

                        if measurement_type == 'filtered_pcr' and cr_on is not None and cr_on_dcr is not None and cr_off is not None:
                            assert Acc_clicks_on is not None
                            assert Acc_clicks_on_dcr is not None
                            assert Acc_clicks_off is not None
                            cr_on.startFor(int_time, clear=True)
                            cr_off.startFor(int_time, clear=True)
                            cr_on_dcr.startFor(int_time, clear=True)

                            cr_on.waitUntilFinished()
                            cr_off.waitUntilFinished()
                            cr_on_dcr.waitUntilFinished()

                            clicks_on = cr_on.getData()
                            clicks_off = cr_off.getData()
                            clicks_on_dcr = cr_on_dcr.getData()

                            clicks_on_total = clicks_on[0][0]
                            clicks_on_dcr_total = clicks_on_dcr[0][0]
                            clicks_off_total = clicks_off[0][0]


                            # Accumulate raw clicks
                            Acc_clicks_on[j, i]  += clicks_on_total
                            Acc_clicks_on_dcr[j, i] += clicks_on_dcr_total
                            Acc_clicks_off[j, i] += clicks_off_total
                            Acc_n[j, i] += 1

                            # Show instantaneous values for this single integration
                            ratio_on_eff = self.ratio_on * fudge_factor
                            ratio_on_dcr_eff = self.ratio_on_dcr * fudge_factor
                            ratio_off_eff = self.ratio_off / fudge_factor
                            inst_signal = (clicks_on_total / (ratio_on_eff * int_time_sec)) - (clicks_off_total / (ratio_off_eff * int_time_sec))
                            inst_on_dcr = clicks_on_dcr_total / (ratio_on_dcr_eff * int_time_sec)
                            inst_dark   = clicks_off_total / (ratio_off_eff * int_time_sec)
                            self.progress.emit(
                                f"      Signal(inst): {inst_signal:.3f}, On-DCR(inst): {inst_on_dcr:.3f}, Dark(inst): {inst_dark:.3f}"
                            )

                        elif measurement_type == 'dcr' and cr_dcr is not None:
                            assert Acc_dcr is not None
                            cr_dcr.startFor(int_time, clear=True)
                            cr_dcr.waitUntilFinished()

                            clicks_data = cr_dcr.getData()
                            bin_counts = clicks_data[0]
                            bin_counts_per_sec = bin_counts / bin_duration
                            Acc_dcr[j, i, :] += bin_counts_per_sec
                            Acc_n[j, i] += 1
                            avg_count = numpy.mean(bin_counts_per_sec)
                            self.progress.emit(f"      DCR avg(inst): {avg_count:.2f} Hz")

                        # Update ETA/progress after each trigger
                        completed_triggers = (
                            cycle * num_bias_pts * num_trigger_levels
                            + i * num_trigger_levels
                            + (j + 1)
                        )
                        elapsed = max(0.0, time.time() - start_time)
                        fraction = min(1.0, completed_triggers / float(total_triggers)) if total_triggers > 0 else 1.0
                        remaining_est = max(0.0, total_estimated * (1.0 - fraction))
                        self.progress.emit(
                            f"Elapsed {elapsed/60:.1f} min, est. remaining {remaining_est/60:.1f} min"
                        )

                    # ---- Derive running-average arrays for live plot ----
                    if measurement_type == 'filtered_pcr':
                        assert Acc_clicks_on is not None
                        assert Acc_clicks_on_dcr is not None
                        assert Acc_clicks_off is not None
                    else:
                        assert Acc_dcr is not None

                    Counts, Counts_on_dcr, Counts_off, Clicks_on, Clicks_on_dcr, Clicks_off = self._derive_averages(
                        measurement_type, num_trigger_levels, num_bias_pts,
                        fudge_factor, int_time_sec,
                        self.ratio_on, self.ratio_on_dcr, self.ratio_off,
                        Acc_clicks_on if measurement_type == 'filtered_pcr' else None,
                        Acc_clicks_on_dcr if measurement_type == 'filtered_pcr' else None,
                        Acc_clicks_off if measurement_type == 'filtered_pcr' else None,
                        Acc_dcr if measurement_type != 'filtered_pcr' else None,
                        Acc_n, num_bins,
                    )

                    # Emit live-plot snapshot after each bias point
                    self.progress.emit({
                        'type': 'update',
                        'I_b': I_b,
                        'x_vals': list(x_vals),
                        'Counts': Counts,
                        'Counts_on_dcr': Counts_on_dcr,
                        'Counts_off': Counts_off,
                        'Clicks_on': Clicks_on,
                        'Clicks_on_dcr': Clicks_on_dcr,
                        'Clicks_off': Clicks_off,
                        'measurement_type': measurement_type,
                        'trigger_levels': trigger_levels,
                        'cycle': cycle + 1,
                        'num_cycles': num_cycles,
                        'current_bias_ua': current_bias_ua,
                        'bias_index': i,
                        'num_bias_pts': num_bias_pts,
                    })

                    # Respect cancel between bias points
                    if self._check_cancel_after_bias():
                        cancel_break = True
                        break

                cycles_completed = cycle + 1

            # ---- Final derived averages for the result ----
            if measurement_type == 'filtered_pcr':
                assert Acc_clicks_on is not None
                assert Acc_clicks_on_dcr is not None
                assert Acc_clicks_off is not None
            else:
                assert Acc_dcr is not None

            Counts, Counts_on_dcr, Counts_off, Clicks_on, Clicks_on_dcr, Clicks_off = self._derive_averages(
                measurement_type, num_trigger_levels, num_bias_pts,
                fudge_factor, int_time_sec,
                self.ratio_on, self.ratio_on_dcr, self.ratio_off,
                Acc_clicks_on if measurement_type == 'filtered_pcr' else None,
                Acc_clicks_on_dcr if measurement_type == 'filtered_pcr' else None,
                Acc_clicks_off if measurement_type == 'filtered_pcr' else None,
                Acc_dcr if measurement_type != 'filtered_pcr' else None,
                Acc_n, num_bins,
            )

            # Package data and let the GUI do CSV + shutdown
            result = {
                'I_b': I_b,
                'x_vals': x_vals,
                'Counts': Counts,
                'Counts_on_dcr': Counts_on_dcr,
                'Counts_off': Counts_off,
                'Clicks_on': Clicks_on,
                'Clicks_on_dcr': Clicks_on_dcr,
                'Clicks_off': Clicks_off,
                'trigger_levels': trigger_levels,
                'measurement_type': measurement_type,
                'num_bins': num_bins,
                'bin_duration': bin_duration,
                'int_time_sec': int_time_sec,
                'num_cycles': num_cycles,
                'cycles_completed': cycles_completed,
                'params': params,
                'filename': self.filename,
                'cancelled': self._cancel_requested,
            }
            self.finished_ok.emit(result)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"PCR worker error: {e}\n{tb}")


class PCRProgressDialog(QDialog):
    """Simple progress dialog with a Cancel button for PCR scans."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super(PCRProgressDialog, self).__init__(parent)
        self.setWindowTitle("PCR Measurement")

        layout = QVBoxLayout(self)
        self.label = QLabel("Running PCR curve measurement…")
        layout.addWidget(self.label)

        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def set_status(self, text: str):
        self.label.setText(text)

    def set_finishing_up(self):
        self.label.setText("Finishing up current bias point…")
        self.cancel_button.setEnabled(False)


class ShutdownConfirmationDialog(QDialog):
    """Dialog asking user whether to shut down instruments after PCR cancel.
    
    Has a 30-second timeout. If no choice is made, falls back to YAML config.
    """

    def __init__(self, parent=None):
        super(ShutdownConfirmationDialog, self).__init__(parent)
        self.setWindowTitle("Shutdown Instruments?")
        self.setModal(True)
        
        self.user_choice = None  # Will be 'yes', 'no', or None (timeout)
        self.timeout_seconds = 30
        self.remaining_seconds = self.timeout_seconds
        
        self.setupUI()
        
        # Timer to update countdown and handle timeout
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)  # 1 second
        self.countdown_timer.timeout.connect(self._on_countdown_tick)
        self.countdown_timer.start()
        
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        self.question_label = QLabel(
            "Turn off cryoamp, thermal source, and sim928?"
        )
        layout.addWidget(self.question_label)
        
        self.countdown_label = QLabel(
            f"(Defaulting to YAML config in {self.remaining_seconds} seconds)"
        )
        layout.addWidget(self.countdown_label)
        
        button_layout = QHBoxLayout()
        
        self.yes_button = QPushButton("Yes")
        self.yes_button.clicked.connect(self._on_yes_clicked)
        button_layout.addWidget(self.yes_button)
        
        self.no_button = QPushButton("No")
        self.no_button.clicked.connect(self._on_no_clicked)
        button_layout.addWidget(self.no_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def _on_countdown_tick(self):
        self.remaining_seconds -= 1
        if self.remaining_seconds <= 0:
            self.countdown_timer.stop()
            self.user_choice = None  # Timeout - use YAML config
            self.accept()
        else:
            self.countdown_label.setText(
                f"(Defaulting to YAML config in {self.remaining_seconds} seconds)"
            )
    
    def _on_yes_clicked(self):
        self.countdown_timer.stop()
        self.user_choice = 'yes'
        self.accept()
        
    def _on_no_clicked(self):
        self.countdown_timer.stop()
        self.user_choice = 'no'
        self.accept()


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

class TeledyneT3AFG200ControlDialog(QDialog):
    """Modal dialog for controlling the Teledyne T3AFG200 function generator"""

    def __init__(self, parent=None):
        super(TeledyneT3AFG200ControlDialog, self).__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Teledyne T3AFG200 Function Generator Control")
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
        """Set the high level on the Teledyne T3AFG200 function generator"""
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


class PowerRampDialog(QDialog):
    """Interactive dialog for the Power Ramp measurement.

    The operator manually adjusts QCL current between each measurement
    point while the detector bias is held fixed.  The dialog collects
    signal/dark count rates at each QCL current and live-plots the result.

    Flow
    ----
    1. Ask for bias voltage → set SIM928.
    2. Ask for trigger level → set on tagger.
    3. Loop:
        a. Ask for QCL current (mA) – or finish.
        b. Integrate and compute signal / dark.
        c. Append to arrays, update live plot.
    4. On "Finish" → save CSV, show final plot.
    """

    def __init__(self, parent_window, parent=None):
        super(PowerRampDialog, self).__init__(parent)
        self.parent_window = parent_window
        self.setWindowTitle("Power Ramp Measurement")
        self.setModal(True)
        self.resize(420, 320)

        # Data accumulators
        self.qcl_currents = []
        self.signal_rates = []
        self.dark_rates = []
        self.clicks_on_list = []
        self.clicks_off_list = []

        # Will be set after the first prompts
        self.bias_voltage = None
        self.trigger_level_value = None
        self.filename = None

        self._build_ui()

    # ---- UI construction ------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Press Start to begin.")
        layout.addWidget(self.status_label)

        # --- Bias voltage input ---
        bias_group = QGroupBox("1. Detector Bias Voltage")
        bias_lay = QHBoxLayout()
        bias_lay.addWidget(QLabel("Bias V:"))
        self.bias_spinbox = QDoubleSpinBox()
        self.bias_spinbox.setRange(0.0, 15.0)
        self.bias_spinbox.setDecimals(4)
        self.bias_spinbox.setSingleStep(0.001)
        self.bias_spinbox.setSuffix(" V")
        self.bias_spinbox.setValue(0.0)
        bias_lay.addWidget(self.bias_spinbox)
        bias_group.setLayout(bias_lay)
        layout.addWidget(bias_group)

        # --- Trigger level input ---
        trig_group = QGroupBox("2. Trigger Level")
        trig_lay = QHBoxLayout()
        trig_lay.addWidget(QLabel("Trigger:"))
        self.trigger_spinbox = QDoubleSpinBox()
        self.trigger_spinbox.setRange(-2.5, 2.5)
        self.trigger_spinbox.setDecimals(4)
        self.trigger_spinbox.setSingleStep(0.001)
        self.trigger_spinbox.setSuffix(" V")
        self.trigger_spinbox.setValue(0.014)
        trig_lay.addWidget(self.trigger_spinbox)
        trig_group.setLayout(trig_lay)
        layout.addWidget(trig_group)

        # --- QCL current input (reused each iteration) ---
        qcl_group = QGroupBox("3. QCL Current (enter & measure)")
        qcl_lay = QHBoxLayout()
        qcl_lay.addWidget(QLabel("QCL I:"))
        self.qcl_spinbox = QDoubleSpinBox()
        self.qcl_spinbox.setRange(0.0, 9999.0)
        self.qcl_spinbox.setDecimals(2)
        self.qcl_spinbox.setSingleStep(1.0)
        self.qcl_spinbox.setSuffix(" mA")
        self.qcl_spinbox.setValue(0.0)
        qcl_lay.addWidget(self.qcl_spinbox)
        self.measure_button = QPushButton("Measure")
        self.measure_button.clicked.connect(self._on_measure_clicked)
        self.measure_button.setEnabled(False)
        qcl_lay.addWidget(self.measure_button)
        qcl_group.setLayout(qcl_lay)
        layout.addWidget(qcl_group)

        # --- Results label ---
        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        # --- Bottom buttons ---
        btn_lay = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        btn_lay.addWidget(self.start_button)

        self.finish_button = QPushButton("Finish && Save")
        self.finish_button.clicked.connect(self._on_finish_clicked)
        self.finish_button.setEnabled(False)
        btn_lay.addWidget(self.finish_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        btn_lay.addWidget(self.cancel_button)

        layout.addLayout(btn_lay)

    # ---- Handlers --------------------------------------------------------

    def _on_start_clicked(self):
        """Set bias, set trigger, and enable the QCL measurement loop."""
        # Ask for CSV filename
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save Power Ramp Data',
            directory='Power_Ramp_Data.csv',
            filter='CSV Files (*.csv);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog,
        )
        if not filename:
            return
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        self.filename = filename

        # Set bias voltage on SIM928
        self.bias_voltage = self.bias_spinbox.value()
        success = self.parent_window._set_source_voltage_robustly(self.bias_voltage)
        if not success:
            QMessageBox.warning(self, "Error", "Failed to set SIM928 bias voltage.")
            return

        # Set trigger level on channel C
        self.trigger_level_value = self.trigger_spinbox.value()
        channelC = self.parent_window.ui.channelC.value()
        self.parent_window.tagger.setTriggerLevel(channelC, self.trigger_level_value)

        self.status_label.setText(
            f"Bias = {self.bias_voltage:.4f} V, "
            f"Trigger = {self.trigger_level_value:.4f} V\n"
            "Enter QCL current and press Measure."
        )

        # Lock the setup inputs, enable measurement
        self.bias_spinbox.setEnabled(False)
        self.trigger_spinbox.setEnabled(False)
        self.start_button.setEnabled(False)
        self.measure_button.setEnabled(True)
        self.finish_button.setEnabled(True)

    def _on_measure_clicked(self):
        """Take one filtered-PCR integration at the current QCL current."""
        import time as _time

        pw = self.parent_window
        params = pw.params
        int_time_sec = float(params.get('integration_time', 10))
        fudge_factor = float(params.get('fudge_factor', 1.0))
        int_time_ps = int(int_time_sec * 1e12)

        qcl_current = self.qcl_spinbox.value()

        self.status_label.setText(f"Integrating for {int_time_sec} s …")
        QApplication.processEvents()

        try:
            filtered_on_ch = pw.filtered_on.getChannel()
            filtered_off_ch = pw.filtered_off.getChannel()

            cr_on = Counter(pw.tagger, [filtered_on_ch], binwidth=int_time_ps, n_values=1)
            cr_off = Counter(pw.tagger, [filtered_off_ch], binwidth=int_time_ps, n_values=1)

            _time.sleep(0.2)

            cr_on.startFor(int_time_ps, clear=True)
            cr_off.startFor(int_time_ps, clear=True)
            cr_on.waitUntilFinished()
            cr_off.waitUntilFinished()

            clicks_on_total = cr_on.getData()[0][0]
            clicks_off_total = cr_off.getData()[0][0]

            ratio_on_eff = pw.ratio_on * fudge_factor
            ratio_off_eff = pw.ratio_off / fudge_factor

            signal = (clicks_on_total / (ratio_on_eff * int_time_sec)) \
                   - (clicks_off_total / (ratio_off_eff * int_time_sec))
            dark = clicks_off_total / (ratio_off_eff * int_time_sec)

            # Store
            self.qcl_currents.append(qcl_current)
            self.signal_rates.append(signal)
            self.dark_rates.append(dark)
            self.clicks_on_list.append(clicks_on_total)
            self.clicks_off_list.append(clicks_off_total)

            n = len(self.qcl_currents)
            self.result_label.setText(
                f"Point {n}: QCL = {qcl_current:.2f} mA → "
                f"Signal = {signal:.3f}, Dark = {dark:.3f}"
            )
            self.status_label.setText("Ready for next QCL current.")

            # Update the live plot on the main window
            self._update_live_plot()

        except Exception as e:
            self.status_label.setText(f"Measurement error: {e}")
            print(f"Power ramp measurement error: {e}")

    def _update_live_plot(self):
        """Redraw the correlationAxis in the main window with power ramp data."""
        import numpy as np

        ax = self.parent_window.correlationAxis
        ax.clear()

        x = np.array(self.qcl_currents)
        sig = np.array(self.signal_rates)
        drk = np.array(self.dark_rates)

        colors = compute_pcr_colors(2)
        ax.plot(x, sig, color=colors[0], marker='o', markersize=5,
                linestyle='-', label='Signal')
        ax.plot(x, drk, color=colors[1], marker='s', markersize=4,
                linestyle='--', label='Dark')

        ax.set_xlabel('QCL Current (mA)')
        ax.set_ylabel('Count Rate (Hz)')
        ax.set_title(f'Power Ramp  (bias = {self.bias_voltage:.4f} V)')
        ax.grid(True)
        ax.legend(loc='best')
        self.parent_window.fig.tight_layout()
        self.parent_window.canvas.draw_idle()

    def _on_finish_clicked(self):
        """Save collected data to CSV and show a final matplotlib window."""
        import numpy as np
        import csv as _csv

        if not self.qcl_currents:
            QMessageBox.information(self, "No data", "No measurements to save.")
            return

        # --- Save CSV ---
        try:
            with open(self.filename, 'w', newline='') as f:
                writer = _csv.writer(f)
                writer.writerow(['# Power Ramp Measurement'])
                writer.writerow(['bias_voltage_V', self.bias_voltage])
                writer.writerow(['trigger_level_V', self.trigger_level_value])
                writer.writerow(['integration_time_s',
                                 self.parent_window.params.get('integration_time', '')])
                writer.writerow(['fudge_factor',
                                 self.parent_window.params.get('fudge_factor', '')])
                writer.writerow([])
                writer.writerow(['QCL_Current_mA', 'Signal_Hz', 'Dark_Hz',
                                 'ClicksOn', 'ClicksOff'])
                for i in range(len(self.qcl_currents)):
                    writer.writerow([
                        self.qcl_currents[i],
                        self.signal_rates[i],
                        self.dark_rates[i],
                        self.clicks_on_list[i],
                        self.clicks_off_list[i],
                    ])
            print(f"Power ramp CSV saved: {self.filename}")
        except Exception as e:
            print(f"Error saving power ramp CSV: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save CSV: {e}")

        # --- Pop-up final plot ---
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            x = np.array(self.qcl_currents)
            colors = compute_pcr_colors(2)
            ax.plot(x, self.signal_rates, color=colors[0], marker='o',
                    markersize=5, linestyle='-', label='Signal')
            ax.plot(x, self.dark_rates, color=colors[1], marker='s',
                    markersize=4, linestyle='--', label='Dark')
            ax.set_xlabel('QCL Current (mA)')
            ax.set_ylabel('Count Rate (Hz)')
            ax.set_title(f'Power Ramp  (bias = {self.bias_voltage:.4f} V)')
            ax.grid(True)
            ax.legend(loc='best', fancybox=False)
            fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error showing final power ramp plot: {e}")

        self.accept()


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
        self.ui.clearButton.clicked.connect(self.open_teledyneT3AFG200_control)
        self.ui.saveButton.clicked.connect(self.saveHistogram)
        self.ui.powerRampButton.clicked.connect(self.power_ramp)
        self.interarrival_button = QPushButton("Interarrival Corr")
        self.interarrival_button.setCheckable(True)
        self.interarrival_button.toggled.connect(self.toggle_interarrival_view)
        self.ui.horizontalLayout_2.addWidget(self.interarrival_button)
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

        # Create the matplotlib figure with its subplots for the counter and PCR
        # Use a 1:3 height ratio so the PCR plot is larger.
        self.fig = Figure()
        gs = GridSpec(2, 1, height_ratios=[1, 3], figure=self.fig)
        self.counterAxis = self.fig.add_subplot(gs[0, 0])
        self.correlationAxis = self.fig.add_subplot(gs[1, 0])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ui.plotLayout.addWidget(self.toolbar)
        self.ui.plotLayout.addWidget(self.canvas)

        self.masked_hist_bins = 2
        self.plot_mode = 'histogram'
        self.special_corr = None
        self.special_corr_start = None
        self.special_corr_delay_ps = 100
        self.plt_correlation = []


        # phd_style(jupyterStyle=True, data_width=1)
        phd_grid_style(grid=True)

        self.load_params()

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

        # Initialize function generator (Teledyne T3AFG200) and Keysight PSU
        try:
            self.function_gen = ClientTeledyneT3AFG200()
            self.function_gen.connect()
            print("Function generator (Teledyne T3AFG200) connected successfully")
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
        self.draw_timer_interval_ms = 50
        self.histogram_block_duration_ms = 200
        self.histogram_sync = None
        self.histogram_start_counter = None
        self.histogram_block_running = False
        self.save_requested = False
        self.save_filename = None
        self.histStartCounts = None
        self.persistentStartCounts = 0.0
        self.updateMeasurements()

        # Use a timer to redraw the plots every 100ms
        self.draw()
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw)
        self.timer.start(self.draw_timer_interval_ms)
        self.clock_divider = 2000  # divider 156.25MHz down to 78.125 KHz
        self.tagger.setEventDivider(18,self.clock_divider)

        


    def load_params(self):
        """Load parameters from PCR_multi_trigger_params.yml"""
        params_file = "./PCR_multi_trigger_params.yml"
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as file:
                    self.params = yaml.safe_load(file)
                print(f"Parameters loaded from {params_file}")
            except Exception as e:
                print(f"Error loading parameters: {e}")
                self.params = {}
        else:
            print(f"Warning: {params_file} not found.")
            self.params = {}

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

    def open_teledyneT3AFG200_control(self):
        """Open the Teledyne T3AFG200 control dialog"""
        dialog = TeledyneT3AFG200ControlDialog(self)
        dialog.exec_()

    def power_ramp(self):
        """Open the Power Ramp measurement dialog.

        Reloads params and updates gating/measurements before starting,
        so that ratio_on / ratio_off are current.
        """
        self.load_params()
        self.updateMeasurements()
        dialog = PowerRampDialog(parent_window=self, parent=self)
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
        self.timer.start(self.draw_timer_interval_ms)
        self.tagger.setEventDivider(18, self.clock_divider)


    def getCouterNormalizationFactor(self):
        bin_index = self.counter.getIndex()
        # normalize 'clicks / bin' to 'kclicks / second'
        return 1e12 / bin_index[1] / 1e3

    def toggle_interarrival_view(self, checked):
        """Toggle the lower plot between histogram and interarrival correlation."""
        self.plot_mode = 'special_correlation' if checked else 'histogram'
        self.interarrival_button.setText("Main Histogram" if checked else "Interarrival Corr")
        self._initialize_correlation_plot()
        self.draw()

    def _apply_correlation_axis_format(self, title, ylabel):
        """Apply common formatting to the lower correlation axis."""
        if self.plot_mode == 'special_correlation' or self.ui.LogScaleCheck.isChecked():
            self.correlationAxis.set_yscale('log')
        else:
            self.correlationAxis.set_yscale('linear')
        self.correlationAxis.set_xlabel('time (ns)')
        self.correlationAxis.set_ylabel(ylabel)
        self.correlationAxis.set_title(title)
        self.correlationAxis.grid(True)

    def _initialize_correlation_plot(self):
        """Initialize the lower plot for the currently selected view mode."""
        self.correlationAxis.clear()

        if self.plot_mode == 'special_correlation' and self.special_corr is not None:
            index, data = self._get_special_corr_plot_data()
            display_data = self._get_special_corr_display_data(data)
            self.plt_correlation = self.correlationAxis.plot(index * 1e-3, display_data, drawstyle='steps-mid')
            self._apply_correlation_axis_format(
                'Interarrival Histogram of filtered_on',
                'Counts',
            )
            self._autoscale_special_correlation_axis(index, data)
        elif hasattr(self, 'correlation') and self.correlation is not None:
            index = self.correlation.getIndex()[self.masked_hist_bins:]
            data = self._normalize_histogram_counts_to_rate(
                self.correlation.getData()[self.masked_hist_bins:],
                start_counts=0.0,
            )
            self.plt_correlation = self.correlationAxis.plot(index * 1e-3, data)
            self._apply_correlation_axis_format(
                'Histogram between A and B',
                'Instantaneous Count Rate (Hz)',
            )
        else:
            self.plt_correlation = self.correlationAxis.plot([], [])
            self._apply_correlation_axis_format('Correlation', 'Counts')

        self.fig.tight_layout()

    def _get_special_corr_plot_data(self):
        """Return StartStop data on a dense grid that respects the current bin width."""
        if self.special_corr is None:
            return numpy.asarray([], dtype=float), numpy.asarray([], dtype=float)

        raw_data = numpy.asarray(self.special_corr.getData(), dtype=float)
        if raw_data.size == 0:
            return numpy.asarray([], dtype=float), numpy.asarray([], dtype=float)

        raw_data = numpy.atleast_2d(raw_data)
        if raw_data.shape[1] < 2:
            return numpy.asarray([], dtype=float), numpy.asarray([], dtype=float)

        binwidth_ps = max(1, int(self.ui.correlationBinwidth.value()))
        n_bins = max(1, int(self.ui.correlationBins.value()))
        raw_times_ps = numpy.asarray(raw_data[:, 0], dtype=float) + float(self.special_corr_delay_ps)
        raw_counts = numpy.asarray(raw_data[:, 1], dtype=float)

        valid = numpy.isfinite(raw_times_ps) & numpy.isfinite(raw_counts) & (raw_counts > 0)
        if not numpy.any(valid):
            return numpy.asarray([], dtype=float), numpy.asarray([], dtype=float)

        raw_times_ps = raw_times_ps[valid]
        raw_counts = raw_counts[valid]

        bin_indices = numpy.rint(raw_times_ps / float(binwidth_ps)).astype(int)
        valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
        if not numpy.any(valid_bins):
            dense_times_ps = numpy.arange(n_bins, dtype=float) * float(binwidth_ps)
            dense_counts = numpy.zeros(n_bins, dtype=float)
            return dense_times_ps, dense_counts

        bin_indices = bin_indices[valid_bins]
        raw_counts = raw_counts[valid_bins]

        dense_counts = numpy.zeros(n_bins, dtype=float)
        numpy.add.at(dense_counts, bin_indices, raw_counts)
        dense_times_ps = numpy.arange(n_bins, dtype=float) * float(binwidth_ps)

        return dense_times_ps, dense_counts

    def _get_special_corr_display_data(self, data):
        """Return special-correlation y-data formatted for plotting on log axes."""
        display_data = numpy.asarray(data, dtype=float).copy()
        display_data[display_data <= 0] = numpy.nan
        return display_data

    def _autoscale_special_correlation_axis(self, index, data):
        """Autoscale the special-correlation axis, including the empty-data case."""
        binwidth_ns = max(float(self.ui.correlationBinwidth.value()) * 1e-3, 1e-6)
        n_bins = max(1, int(self.ui.correlationBins.value()))
        self.correlationAxis.set_xlim(0.0, n_bins * binwidth_ns)

        if index.size == 0 or data.size == 0 or not numpy.any(numpy.isfinite(data)):
            self.correlationAxis.set_ylim(0.0, 1.0)
            return

        finite_data = numpy.asarray(data, dtype=float)
        finite_data = finite_data[numpy.isfinite(finite_data)]
        if finite_data.size == 0:
            self.correlationAxis.set_ylim(0.0, 1.0)
            return

        max_y = float(numpy.max(finite_data))
        positive = finite_data[finite_data > 0]
        if self.plot_mode == 'special_correlation' or self.ui.LogScaleCheck.isChecked():
            if positive.size == 0:
                self.correlationAxis.set_ylim(0.5, 1.0)
            else:
                min_y = float(numpy.min(positive))
                self.correlationAxis.set_ylim(max(min_y * 0.8, 1e-12), max(max_y * 1.2, min_y * 2.0))
        else:
            self.correlationAxis.set_ylim(0.0, max(1.0, max_y * 1.05))

    def _get_histogram_block_duration_ps(self):
        """Return the live histogram block duration in ps."""
        return int(self.histogram_block_duration_ms * 1e9)

    def _get_histogram_block_count(self):
        """Return the number of histogram blocks kept in memory."""
        integration_time_s = float(self.ui.IntTime.value())
        block_duration_s = self.histogram_block_duration_ms / 1000.0
        if block_duration_s <= 0:
            return 1
        return max(1, int(round(integration_time_s / block_duration_s)))

    def _start_histogram_block_acquisition(self):
        """Start one synchronized histogram block if no block is active."""
        if self.histogram_sync is None or self.histogram_block_running:
            return

        self.histogram_sync.startFor(self._get_histogram_block_duration_ps(), clear=True)
        self.histogram_block_running = True

    def _normalize_histogram_counts_to_rate(self, counts, start_counts):
        """Convert histogram counts to instantaneous count rate in Hz.

        For a multiple-start / multiple-stop histogram,

            counts(bin) ~= N_start * rate(delay) * bin_width

        so the instantaneous rate is obtained from

            rate(delay) = counts(bin) / (N_start * bin_width)

        where `N_start` is the accumulated number of detected start events over
        the displayed integration window.
        """
        counts_array = numpy.asarray(counts, dtype=float)
        binwidth_ps = float(self.ui.correlationBinwidth.value())
        binwidth_s = binwidth_ps * 1e-12

        if counts_array.size == 0:
            return counts_array

        if start_counts <= 0 or binwidth_s <= 0:
            return numpy.zeros_like(counts_array, dtype=float)

        return counts_array / (float(start_counts) * binwidth_s)

    def updateMeasurements(self):
        '''Create/Update all TimeTagger measurement objects'''

        # Always reload YAML parameters so gating settings update immediately.
        self.load_params()

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

        self.correlationAxis.set_yscale('log' if self.ui.LogScaleCheck.isChecked() else 'linear')
        self.seconds = 1
        histblock_depth = self._get_histogram_block_count()
        print("histblock depth: ", histblock_depth)
        self.histBlock = numpy.zeros((histblock_depth, self.ui.correlationBins.value() - self.masked_hist_bins))
        self.histStartCounts = numpy.zeros(histblock_depth, dtype='float')
        self.persistentStartCounts = 0.0

        self.buffer = numpy.zeros((1,self.ui.correlationBins.value()))[self.masked_hist_bins:]
        self.buffer_old = numpy.zeros((1, self.ui.correlationBins.value()))[self.masked_hist_bins:]


        self.BlockIndex = 0

        print(self.active_channels)


        # Load gating delays from params based on mode
        mode = self.params.get('mode', 'thermal_source')
        delays = self.params.get('gating_delays', {}).get(mode, {
            'on_start': 30, 'on_stop': 270,
            'on_dcr_start': 270, 'on_dcr_stop': 450,
            'off_start': 450, 'off_stop': 950
        })
        
        on_start = delays.get('on_start', 30)
        on_stop = delays.get('on_stop', 270)
        off_start = delays.get('off_start', 450)
        off_stop = delays.get('off_stop', 950)
        on_dcr_start = delays.get('on_dcr_start', on_stop)
        on_dcr_stop = delays.get('on_dcr_stop', off_start)
        
        print(
            f"Mode: {mode}, Delays (ms): on=[{on_start}, {on_stop}], "
            f"on_dcr=[{on_dcr_start}, {on_dcr_stop}], off=[{off_start}, {off_stop}]"
        )

        # for us right now (oct 9 2024), self.active_channels[2] (3rd row) is 5, which is the snspd
        self.filtered = GatedChannel(self.tagger, self.active_channels[2], self.active_channels[0], -self.active_channels[0])
        self.delay_1_start = DelayedChannel(self.tagger, self.active_channels[0], int(on_start*1e9))
        self.delay_1_stop = DelayedChannel(self.tagger, self.active_channels[0], int(on_stop*1e9))

        self.delay_1_5_start = DelayedChannel(self.tagger, self.active_channels[0], int(on_dcr_start*1e9))
        self.delay_1_5_stop = DelayedChannel(self.tagger, self.active_channels[0], int(on_dcr_stop*1e9))


        self.delay_2_start = DelayedChannel(self.tagger, self.active_channels[0], int(off_start*1e9))
        self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(off_stop*1e9))
        # self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(800e9))


        self.ratio_on = (on_stop - on_start) / 1000
        self.ratio_on_dcr = (on_dcr_stop - on_dcr_start) / 1000
        self.ratio_off = (off_stop - off_start) / 1000

        # Scale by pulse repetition rate: in QCL mode there are multiple
        # pulses per second; in thermal_source mode there is always 1.
        if mode == 'qcl':
            pulse_rep_rate = float(self.params.get('pulse_rep_rate', 1))
        else:
            pulse_rep_rate = 1.0
        self.ratio_on *= pulse_rep_rate
        self.ratio_on_dcr *= pulse_rep_rate
        self.ratio_off *= pulse_rep_rate
        print(
            f"pulse_rep_rate: {pulse_rep_rate}, ratio_on: {self.ratio_on}, "
            f"ratio_on_dcr: {self.ratio_on_dcr}, ratio_off: {self.ratio_off}"
        )


        # thermal source on
        self.filtered_on = GatedChannel(self.tagger, self.active_channels[2], self.delay_1_start.getChannel(), self.delay_1_stop.getChannel())

        self.filtered_on_dcr = GatedChannel(self.tagger, self.active_channels[2], self.delay_1_5_start.getChannel(), self.delay_1_5_stop.getChannel())

        # thermal source off
        self.filtered_off = GatedChannel(self.tagger, self.active_channels[2], self.delay_2_start.getChannel(), self.delay_2_stop.getChannel())

        # Create a slightly delayed copy of filtered_on so StartStop does not
        # pair each event with itself at t = 0.
        self.special_corr_start = DelayedChannel(
            self.tagger,
            self.filtered_on.getChannel(),
            self.special_corr_delay_ps,
        )



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
                self.active_channels + list([self.filtered_on.getChannel()]),
                binwidth=int(50e9),
                n_values=200
            )


        # Measure the correlation between A and B
        # self.correlation = Correlation(
        #     self.tagger,
        #     #self.a_combined.getChannel(),
        #     #self.b_combined.getChannel(),
        #     # self.active_channels[1],
        #     # self.filtered.getChannel(),
        #     self.filtered_on.getChannel(),
        #     CHANNEL_UNUSED,

        #     self.ui.correlationBinwidth.value(),
        #     self.ui.correlationBins.value())
        
        self.histogram_sync = SynchronizedMeasurements(self.tagger)
        histogram_tagger = self.histogram_sync.getTagger()
        self.correlation = Histogram(
            histogram_tagger,
            self.active_channels[2],
            self.active_channels[0],
            self.ui.correlationBinwidth.value(),
            self.ui.correlationBins.value())
        self.special_corr = StartStop(
            self.tagger,
            self.filtered_on.getChannel(),
            self.special_corr_start.getChannel(),
            self.ui.correlationBinwidth.value(),
        )
        self.histogram_start_counter = Counter(
            histogram_tagger,
            [self.active_channels[0]],
            binwidth=self._get_histogram_block_duration_ps(),
            n_values=1,
        )
        self.histogram_block_running = False
        self._start_histogram_block_acquisition()

        self.tagger.sync()

        # Create the measurement plots
        self.counterAxis.clear()  # this is a matplotlib figure

        # Use the shared colormap helper to color each count-rate trace
        count_rate_colors = compute_pcr_colors(
            len(self.active_channels) + 1, crop_start=0.2, crop_end=0.7
        )

        # Plot all count-rate traces; this returns a list of Line2D objects
        self.plt_counter = self.counterAxis.plot(
            self.counter.getIndex() * 1e-12,
            self.counter.getData().T * self.getCouterNormalizationFactor(),
        )

        # Assign a distinct color to each line
        for line, color in zip(self.plt_counter, count_rate_colors):
            line.set_color(color)
        self.counterAxis.set_xlabel('time (s)')
        self.counterAxis.set_ylabel('Rate (kHz)')
        # self.counterAxis.set_title('Count rate')
        # self.counterAxis.legend(['A', 'B', 'C', 'D','coincidences'])
        self.counterAxis.grid(True)

        self._initialize_correlation_plot()

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
            if self.special_corr is not None:
                try:
                    self.special_corr.start()
                except Exception:
                    pass
            self._start_histogram_block_acquisition()

    def stopClicked(self):
        '''Handler for the stop action button'''
        self.running = False
        self.counter.stop()
        if self.special_corr is not None:
            try:
                self.special_corr.stop()
            except Exception:
                pass
        if self.histogram_sync is not None:
            self.histogram_sync.stop()
        else:
            self.correlation.stop()
        self.histogram_block_running = False

    def clearClicked(self):
        '''Handler for the clear action button'''
        self.correlation.clear()
        if self.special_corr is not None:
            try:
                self.special_corr.clear()
            except Exception:
                pass
        if self.histogram_start_counter is not None:
            self.histogram_start_counter.clear()

    def _prompt_histogram_save_filename(self):
        """Prompt the user for a JSON filename for histogram export."""
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save Histogram Data',
            directory='histogram_data.json',
            filter='JSON Files (*.json);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )

        if not filename:
            print("Save operation cancelled.")
            return None

        if not filename.lower().endswith('.json'):
            filename += '.json'

        return filename

    
    def saveHistogram(self):
        """Save the currently active lower-plot data as JSON."""
        filename = self._prompt_histogram_save_filename()
        if filename is None:
            return

        if self.plot_mode == 'special_correlation':
            self._save_special_correlation_data(filename)
            return

        self.save_requested = True
        self.save_filename = filename

        print(f"Histogram will be saved to {filename} when data collection is complete.")
        print(f"Integration depth: {self._get_histogram_block_count()} blocks")
        print("Data collection in progress...")
    
    def _save_histogram_data(self, filename=None):
        """Internal method to save the accumulated standard histogram data."""
        try:
            target_filename = filename if filename is not None else self.save_filename
            if not target_filename:
                print("Error saving histogram data: no target filename set")
                return

            # Get the accumulated data (sum of histBlock)
            accumulated_data = numpy.sum(self.histBlock, axis=0)
            total_start_counts = float(numpy.sum(self.histStartCounts)) if self.histStartCounts is not None else 0.0
            normalized_rate_hz = self._normalize_histogram_counts_to_rate(
                accumulated_data,
                total_start_counts,
            )
            
            # Get the x-axis data (index)
            index = self.correlation.getIndex()[self.masked_hist_bins:]
            
            # Prepare data for JSON export
            data_dict = {
                'data_columns': [
                    'x_axis_ps',
                    'histogram_counts',
                    'instantaneous_count_rate_hz',
                ],
                'x_axis_ps': index.tolist(),  # Convert to list for JSON serialization
                'histogram_counts': accumulated_data.tolist(),
                'instantaneous_count_rate_hz': normalized_rate_hz.tolist(),
                'integration_time_value': self.ui.IntTime.value(),
                'integration_blocks': self._get_histogram_block_count(),
                'histogram_block_duration_ms': self.histogram_block_duration_ms,
                'histogram_start_counts_per_block': self.histStartCounts.tolist() if self.histStartCounts is not None else [],
                'histogram_total_start_counts': total_start_counts,
                'binwidth_ps': self.ui.correlationBinwidth.value(),
                'total_bins': self.ui.correlationBins.value(),
                'masked_bins': self.masked_hist_bins,
                'y_axis_units': 'Hz',
                'plot_mode': 'histogram',
                'measurement_class': 'Histogram',
                'y_axis_normalization': {
                    'description': 'Per-start instantaneous count rate used by the live GUI histogram',
                    'formula': 'rate_hz = histogram_counts / (histogram_total_start_counts * binwidth_seconds)',
                    'start_channel': self.active_channels[0] if self.active_channels else None,
                },
                'timestamp': str(numpy.datetime64('now'))
            }
            
            # Save to JSON file
            with open(target_filename, 'w') as file:
                json.dump(data_dict, file, indent=2)
            
            print(f"Histogram data successfully saved to: {target_filename}")
            print(f"Total counts in histogram: {numpy.sum(accumulated_data)}")
            
        except Exception as e:
            print(f"Error saving histogram data: {e}")
        finally:
            # Reset the save flag
            self.save_requested = False
            self.save_filename = None

    def _save_special_correlation_data(self, filename):
        """Save the StartStop special-correlation data immediately as JSON."""
        try:
            if self.special_corr is None:
                print("Error saving special correlation data: measurement not initialized")
                return

            raw_data = numpy.asarray(self.special_corr.getData(), dtype=float)
            raw_data = numpy.atleast_2d(raw_data) if raw_data.size else numpy.empty((0, 2), dtype=float)
            if raw_data.shape[1] < 2:
                raw_data = numpy.empty((0, 2), dtype=float)

            raw_times_minus_delay_ps = raw_data[:, 0].tolist()
            raw_times_ps = (raw_data[:, 0] + float(self.special_corr_delay_ps)).tolist()
            raw_counts = raw_data[:, 1].tolist()

            display_x_ps, display_counts = self._get_special_corr_plot_data()
            display_counts_semilog = self._get_special_corr_display_data(display_counts)

            data_dict = {
                'plot_mode': 'special_correlation',
                'measurement_class': 'StartStop',
                'data_columns': [
                    'display_x_axis_ps',
                    'display_counts',
                    'display_counts_semilog',
                ],
                'raw_sparse_columns': [
                    'time_ps_minus_delay_correction',
                    'time_ps',
                    'counts',
                ],
                'raw_sparse_time_ps_minus_delay_correction': raw_times_minus_delay_ps,
                'raw_sparse_time_ps': raw_times_ps,
                'raw_sparse_counts': raw_counts,
                'display_x_axis_ps': display_x_ps.tolist(),
                'display_counts': display_counts.tolist(),
                'display_counts_semilog': display_counts_semilog.tolist(),
                'binwidth_ps': int(self.ui.correlationBinwidth.value()),
                'display_total_bins': int(self.ui.correlationBins.value()),
                'special_corr_delay_ps': float(self.special_corr_delay_ps),
                'y_axis_units': 'counts',
                'y_axis_scale_for_display': 'log',
                'active_channels': list(self.active_channels),
                'filtered_on_channel': self.filtered_on.getChannel() if hasattr(self, 'filtered_on') else None,
                'special_corr_start_channel': self.special_corr_start.getChannel() if self.special_corr_start is not None else None,
                'timestamp': str(numpy.datetime64('now')),
            }

            with open(filename, 'w') as file:
                json.dump(data_dict, file, indent=2)

            print(f"Special correlation data successfully saved to: {filename}")
            print(f"Non-empty StartStop bins saved: {len(raw_counts)}")
        except Exception as e:
            print(f"Error saving special correlation data: {e}")
    
    # def saveHistogram(self):

    #     pass
        #  wf = teledyneT3AFG200('10.7.0.187')
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


            if shutdown_config.get('qcl', False):
                try:
                
                    self.qcl = hp8114A("/dev/ttyUSB1", 10)
                    self.qcl.connect()
                    time.sleep(0.1)
                    self.qcl.setOutput(0) 
                    time.sleep(0.1)
                    self.qcl.disconnect()

                except Exception as e:
                    print(f"Error turning off QCL: {e}")
                    
        except Exception as e:
            print(f"Error in instrument shutdown: {e}")

    def _shutdown_all_instruments(self):
        """
        Shutdown all instruments unconditionally (used when user confirms shutdown after cancel).
        """
        try:
            print("Shutting down all instruments...")
            
            # Turn off SIM928
            try:
                if self.source is not None:
                    self.source.turnOff()
                    print("SIM928 turned off successfully")
                else:
                    print("SIM928 not available for shutdown")
            except Exception as e:
                print(f"Error turning off SIM928: {e}")
            
            # Turn off cryo_amp (channel 3 of power supply)
            try:
                if self.power_supply is not None:
                    self.power_supply.output_off(3)
                    print("Cryo amp (channel 3) turned off successfully")
                else:
                    print("Power supply not available for cryo amp shutdown")
            except Exception as e:
                print(f"Error turning off cryo amp: {e}")
            
            # Turn off thermal_source (channel 2 of function generator)
            try:
                if self.function_gen is not None:
                    self.function_gen.set_output(2, 0)
                    print("Thermal source (channel 2) turned off successfully")
                else:
                    print("Function generator not available for thermal source shutdown")
            except Exception as e:
                print(f"Error turning off thermal source: {e}")


            # turn off hp8114A pulse generator (QCL)
            try:
                self.qcl = hp8114A("/dev/ttyUSB1", 10)
                self.qcl.connect()
                time.sleep(0.1)
                self.qcl.setOutput(0) 
                time.sleep(0.1)
                self.qcl.disconnect()
            except Exception as e:
                print(f"Error turning off QCL: {e}")
                    
        except Exception as e:
            print(f"Error in instrument shutdown: {e}")

    def PCR(self):
        """Start the PCR measurement in a background thread with cancel support."""

        # Ask user for CSV filename
        filename, _ = QFileDialog().getSaveFileName(
            parent=self,
            caption='Save PCR Curve Data',
            directory='PCR_Curve_Data.csv',
            filter='CSV Files (*.csv);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )

        if not filename:
            print("Save operation cancelled.")
            return

        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        png_filename = filename[:-4] + '.png'

        # Reload params and update hardware settings/gating before starting
        self.load_params()
        self.updateMeasurements()
        
        params = self.params
        if not params:
            print("Error: No parameters loaded. Check PCR_multi_trigger_params.yml")
            return

        try:
            fudge_factor = params['fudge_factor']
            # The worker applies fudge_factor when computing Counts/Counts_off.
            ratio_on = self.ratio_on
            ratio_on_dcr = self.ratio_on_dcr
            ratio_off = self.ratio_off

            measurement_type = params.get('measurement_type', 'filtered_pcr').lower()
            print(f"Measurement type: {measurement_type}")

            if measurement_type == 'filtered_pcr':
                trigger_levels = params['filtered_PCR']['trigger_levels']
                if not isinstance(trigger_levels, list) or not trigger_levels:
                    print("Error: 'trigger_levels' in filtered_PCR YAML must be a non-empty list.")
                    return
            elif measurement_type == 'dcr':
                if 'trigger_levels' in params['DCR']:
                    trigger_levels = params['DCR']['trigger_levels']
                    if not isinstance(trigger_levels, list) or not trigger_levels:
                        print("Error: 'trigger_levels' in DCR YAML must be a non-empty list.")
                        return
                elif 'trigger_level' in params['DCR']:
                    trigger_level = params['DCR']['trigger_level']
                    trigger_levels = [str(trigger_level)]
                    print("Using single DCR trigger level (backward compatibility mode)")
                else:
                    print("Error: DCR section must contain either 'trigger_levels' (list) or 'trigger_level' (single value).")
                    return
            else:
                print(f"Error: Unknown measurement type '{measurement_type}'. Must be 'filtered_pcr' or 'dcr'.")
                return
        except (yaml.YAMLError, KeyError, TypeError) as e:
            print(f"Error loading or parsing parameters: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while loading parameters: {e}")
            return

        # Prepare worker and thread
        try:
            filtered_on_channel = self.filtered_on.getChannel()
            filtered_on_dcr_channel = self.filtered_on_dcr.getChannel()
            filtered_off_channel = self.filtered_off.getChannel()
            active_snspd_channel = self.active_channels[2]
        except Exception as e:
            print(f"Error preparing PCR worker channels: {e}")
            return

        self._pcr_thread = QThread(self)
        self._pcr_worker = PCRWorker(
            tagger=self.tagger,
            params=params,
            filename=filename,
            png_filename=png_filename,
            ratio_on=ratio_on,
            ratio_on_dcr=ratio_on_dcr,
            ratio_off=ratio_off,
            filtered_on_channel=filtered_on_channel,
            filtered_on_dcr_channel=filtered_on_dcr_channel,
            filtered_off_channel=filtered_off_channel,
            active_snspd_channel=active_snspd_channel,
            set_bias_fn=lambda v: self._set_source_voltage_robustly(v),
            channelC=self.ui.channelC.value(),
        )
        self._pcr_worker.moveToThread(self._pcr_thread)

        # Connect signals
        self._pcr_thread.started.connect(self._pcr_worker.run)
        self._pcr_worker.finished_ok.connect(self._on_pcr_finished)
        self._pcr_worker.error.connect(self._on_pcr_error)
        self._pcr_worker.finished_ok.connect(self._pcr_thread.quit)
        self._pcr_worker.finished_ok.connect(self._pcr_worker.deleteLater)
        self._pcr_thread.finished.connect(self._pcr_thread.deleteLater)
        self._pcr_worker.progress.connect(self._on_pcr_progress)
        self._pcr_worker.cancel_ack.connect(self._on_pcr_cancel_ack)

        # Progress dialog with cancel button
        self._pcr_dialog = PCRProgressDialog(self)
        self._pcr_dialog.cancel_requested.connect(self._on_pcr_cancel_clicked)

        self._pcr_thread.start()
        self._pcr_dialog.show()

    def _on_pcr_cancel_clicked(self):
        """User pressed cancel in the PCR progress dialog."""
        if hasattr(self, '_pcr_worker') and self._pcr_worker is not None:
            self._pcr_worker.request_cancel()
        # Give immediate visual feedback in the dialog while
        # the worker finishes the current bias point.
        if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
            self._pcr_dialog.set_finishing_up()

    def _on_pcr_cancel_ack(self):
        """Worker acknowledged cancel; now finishing current bias."""
        if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
            self._pcr_dialog.set_finishing_up()

    def _on_pcr_progress(self, text):
        """Update progress dialog text and live PCR plot from worker."""
        # Simple string: just log and update dialog label
        if isinstance(text, str):
            print(text)
            if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
                self._pcr_dialog.set_status(text)
            return

        # Structured dict with live data snapshot
        if not isinstance(text, dict):
            return

        if text.get('type') != 'update':
            return

        import numpy as np

        I_b = text['I_b']
        x_vals = text['x_vals']
        Counts = text['Counts']
        Counts_on_dcr = text.get('Counts_on_dcr', None)
        Counts_off = text['Counts_off']
        measurement_type = text['measurement_type']
        trigger_levels = text['trigger_levels']

        num_trigger_levels = len(trigger_levels)
        ax = self.correlationAxis
        ax.clear()

        colors = compute_pcr_colors(num_trigger_levels)

        for j in range(num_trigger_levels):
            color = colors[j % len(colors)]

            if measurement_type == 'filtered_pcr':
                y_array = np.array(Counts[j], dtype=float)
                valid = ~np.isnan(y_array)
                if not np.any(valid):
                    continue
                valid_x = np.array(x_vals)[valid]
                valid_y = y_array[valid]

                label = f'TL {j+1}: {trigger_levels[j]}'
                ax.plot(valid_x, valid_y, color=color, marker='o', markersize=4, linestyle='-', label=label)

                if Counts_on_dcr is not None:
                    y_on_dcr = np.array(Counts_on_dcr[j], dtype=float)
                    valid_on_dcr = y_on_dcr[valid]
                    ax.plot(valid_x, valid_on_dcr, color=color, marker='s', markersize=3, linestyle='-.', label=f'TL {j+1} on_dcr', alpha=0.1)

                if Counts_off is not None:
                    y_dark = np.array(Counts_off[j], dtype=float)
                    valid_dark = y_dark[valid]
                    ax.plot(valid_x, valid_dark, color=color, markersize=4, linestyle='--', label=f'TL {j+1} dark')

                cycle_info = text.get('cycle', '')
                num_cycles_info = text.get('num_cycles', 1)
                if num_cycles_info > 1:
                    ax.set_title(f'Gated PCR Curve (cycle {cycle_info}/{num_cycles_info})')
                else:
                    ax.set_title('Gated PCR Curve')

            else:  # dcr
                avg_counts = []
                for count_array in Counts[j]:
                    if isinstance(count_array, np.ndarray):
                        avg_counts.append(np.mean(count_array))
                    else:
                        avg_counts.append(count_array)

                avg_counts = np.array(avg_counts, dtype=float)
                valid = ~np.isnan(avg_counts)
                if not np.any(valid):
                    continue
                valid_x = np.array(x_vals)[valid]
                valid_y = avg_counts[valid]

                label = f'DCR TL: {trigger_levels[j]}'
                ax.plot(valid_x, valid_y, color=color, linestyle='-', label=label)

                cycle_info = text.get('cycle', '')
                num_cycles_info = text.get('num_cycles', 1)
                if num_cycles_info > 1:
                    ax.set_title(f'DCR Curve (cycle {cycle_info}/{num_cycles_info})')
                else:
                    ax.set_title('DCR Curve')

        # Draw a vertical line at the current bias point
        current_bias = text.get('current_bias_ua', None)
        if current_bias is not None:
            ax.axvline(x=current_bias, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

        ax.set_xlabel('Bias Current (uA)')
        ax.set_ylabel('Counts')
        ax.grid(True)
        ax.legend(loc='best')
        self.fig.tight_layout()
        self.canvas.draw_idle()

        if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
            cycle = text.get('cycle', 1)
            num_cycles = text.get('num_cycles', 1)
            bias_idx = text.get('bias_index', 0)
            num_bias = text.get('num_bias_pts', 0)
            bias_str = f'{current_bias:.3f} µA' if current_bias is not None else '?'
            if num_cycles > 1:
                self._pcr_dialog.set_status(
                    f'Cycle {cycle}/{num_cycles}  —  '
                    f'Bias {bias_idx + 1}/{num_bias} ({bias_str})'
                )
            else:
                self._pcr_dialog.set_status(
                    f'Bias {bias_idx + 1}/{num_bias} ({bias_str})'
                )

    def _on_pcr_error(self, message):
        """Handle errors from the PCR worker."""
        print(message)
        if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
            self._pcr_dialog.reject()
            self._pcr_dialog = None

    def _on_pcr_finished(self, result):
        """Worker finished successfully; save CSV and shutdown instruments."""
        I_b = result['I_b']
        x_vals = result['x_vals']
        Counts = result['Counts']
        Counts_on_dcr = result.get('Counts_on_dcr', None)
        Counts_off = result['Counts_off']
        Clicks_on = result.get('Clicks_on', None)
        Clicks_on_dcr = result.get('Clicks_on_dcr', None)
        Clicks_off = result.get('Clicks_off', None)
        trigger_levels = result['trigger_levels']
        measurement_type = result['measurement_type']
        num_bins = result['num_bins']
        params = result['params']
        filename = result['filename']
        num_cycles = result.get('num_cycles', 1)
        cycles_completed = result.get('cycles_completed', 1)

        num_trigger_levels = len(trigger_levels)

        try:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)

                # ---- Metadata block (always saved) ----
                try:
                    mode = str(params.get('mode', '')).strip()
                    gating_delays_all = params.get('gating_delays', {}) or {}
                    gating_active = gating_delays_all.get(mode, {}) if mode else {}
                    csvwriter.writerow(['# metadata', 'value'])
                    if mode:
                        csvwriter.writerow(['mode', mode])
                    if 'fudge_factor' in params:
                        csvwriter.writerow(['fudge_factor', params.get('fudge_factor', '')])
                    csvwriter.writerow(['cycles', num_cycles])
                    csvwriter.writerow(['cycles_completed', cycles_completed])
                    csvwriter.writerow(['integration_time', params.get('integration_time', '')])
                    if gating_active:
                        csvwriter.writerow(['on_start', gating_active.get('on_start', '')])
                        csvwriter.writerow(['on_stop', gating_active.get('on_stop', '')])
                        csvwriter.writerow(['on_dcr_start', gating_active.get('on_dcr_start', '')])
                        csvwriter.writerow(['on_dcr_stop', gating_active.get('on_dcr_stop', '')])
                        csvwriter.writerow(['off_start', gating_active.get('off_start', '')])
                        csvwriter.writerow(['off_stop', gating_active.get('off_stop', '')])
                    csvwriter.writerow([])
                except Exception:
                    # Metadata should never prevent saving data.
                    pass

                header = ['Bias_Current']
                if measurement_type == 'filtered_pcr':
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'Counts_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'CountsOnDcr_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'DCounts_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'ClicksOn_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'ClicksOnDcr_TL{j+1}({tl})')
                    for j, tl in enumerate(trigger_levels):
                        header.append(f'ClicksOff_TL{j+1}({tl})')
                else:
                    for j, tl in enumerate(trigger_levels):
                        for bin_idx in range(num_bins):
                            header.append(f'DCR_TL{j+1}({tl})_Bin{bin_idx+1}')

                csvwriter.writerow(header)

                # The worker may have truncated x_vals/Counts if cancelled
                num_rows = len(x_vals)
                for row_idx in range(num_rows):
                    row = [x_vals[row_idx]]

                    if measurement_type == 'filtered_pcr':
                        for tl_idx in range(num_trigger_levels):
                            count_val = Counts[tl_idx][row_idx]
                            row.append(count_val if not numpy.isnan(count_val) else '')

                        if Counts_on_dcr is not None:
                            for tl_idx in range(num_trigger_levels):
                                dcr_val = Counts_on_dcr[tl_idx][row_idx]
                                row.append(dcr_val if not numpy.isnan(dcr_val) else '')
                        else:
                            for _ in range(num_trigger_levels):
                                row.append('')

                        if Counts_off is not None:
                            for tl_idx in range(num_trigger_levels):
                                dcount_val = Counts_off[tl_idx][row_idx]
                                row.append(dcount_val if not numpy.isnan(dcount_val) else '')

                        # Raw clicks (if available)
                        if Clicks_on is not None:
                            for tl_idx in range(num_trigger_levels):
                                cval = Clicks_on[tl_idx][row_idx]
                                row.append(cval if not numpy.isnan(cval) else '')
                        else:
                            for _ in range(num_trigger_levels):
                                row.append('')

                        if Clicks_on_dcr is not None:
                            for tl_idx in range(num_trigger_levels):
                                cval = Clicks_on_dcr[tl_idx][row_idx]
                                row.append(cval if not numpy.isnan(cval) else '')
                        else:
                            for _ in range(num_trigger_levels):
                                row.append('')

                        if Clicks_off is not None:
                            for tl_idx in range(num_trigger_levels):
                                cval = Clicks_off[tl_idx][row_idx]
                                row.append(cval if not numpy.isnan(cval) else '')
                        else:
                            for _ in range(num_trigger_levels):
                                row.append('')
                    else:
                        for tl_idx in range(num_trigger_levels):
                            count_data = Counts[tl_idx][row_idx]
                            if isinstance(count_data, numpy.ndarray):
                                for bin_val in count_data:
                                    row.append(bin_val if not numpy.isnan(bin_val) else '')
                            else:
                                for _ in range(num_bins):
                                    row.append(count_data if not numpy.isnan(count_data) else '')

                    csvwriter.writerow(row)

            print(f"CSV data saved as: {filename}")
        except Exception as e:
            print(f"Error writing CSV file: {e}")

        # Pop up a separate Matplotlib window at the end of the measurement
        # to allow detailed zooming and manual PNG saving, matching the
        # original PCR() behavior.
        try:

            # phd_style(jupyterStyle=False)
            import numpy as np
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            num_trigger_levels = len(trigger_levels)
            colors = compute_pcr_colors(num_trigger_levels)

            for j in range(num_trigger_levels):
                color = colors[j % len(colors)] if colors else None

                if measurement_type == 'filtered_pcr':
                    y_array = np.array(Counts[j], dtype=float)
                    valid = ~np.isnan(y_array)
                    if not np.any(valid):
                        continue
                    valid_x = np.array(x_vals)[valid]
                    valid_y = y_array[valid]

                    label = f'TL {j+1}: {trigger_levels[j]}'
                    ax.plot(valid_x, valid_y, color=color, marker='o', markersize=4, linestyle='-', label=label)

                    if Counts_on_dcr is not None:
                        y_on_dcr = np.array(Counts_on_dcr[j], dtype=float)
                        valid_on_dcr = y_on_dcr[valid]
                        ax.plot(valid_x, valid_on_dcr, color=color, marker='s', markersize=3, linestyle='-.', label=f'TL {j+1} on_dcr', alpha=0.1)

                    if Counts_off is not None:
                        y_dark = np.array(Counts_off[j], dtype=float)
                        valid_dark = y_dark[valid]
                        ax.plot(valid_x, valid_dark, color=color, markersize=4, linestyle='--', label=f'TL {j+1} dark')

                else:  # dcr
                    avg_counts = []
                    for count_array in Counts[j]:
                        if isinstance(count_array, np.ndarray):
                            avg_counts.append(np.mean(count_array))
                        else:
                            avg_counts.append(count_array)

                    avg_counts = np.array(avg_counts, dtype=float)
                    valid = ~np.isnan(avg_counts)
                    if not np.any(valid):
                        continue
                    valid_x = np.array(x_vals)[valid]
                    valid_y = avg_counts[valid]

                    label = f'DCR TL: {trigger_levels[j]}'
                    ax.plot(valid_x, valid_y, color=color, marker='o', linestyle='-', label=label)

            ax.set_xlabel('Bias Current (uA)')
            ax.set_ylabel('Counts')
            ax.grid(True)
            ax.legend(loc='best', fancybox=False)
            fig.tight_layout()

            # Pause the live-update timer while this window is open so its
            # 50ms draw callbacks don't compete with the save dialog.
            self.timer.stop()

            def _on_pcr_fig_close(event):
                if self.running:
                    self.timer.start(self.draw_timer_interval_ms)

            fig.canvas.mpl_connect('close_event', _on_pcr_fig_close)

            # Blocking window the user can zoom/pan and save from.
            plt.show()
        except Exception as e:
            print(f"Error showing final PCR plot window: {e}")
            # Make sure the timer is always restarted even if something goes wrong.
            if not self.timer.isActive() and self.running:
                self.timer.start(self.draw_timer_interval_ms)

        time.sleep(0.5)

        # Show shutdown confirmation dialog with 30-second timeout
        shutdown_dialog = ShutdownConfirmationDialog(self)
        shutdown_dialog.exec_()
        
        user_choice = shutdown_dialog.user_choice
        if user_choice == 'yes':
            # User chose to shut down all instruments
            print("User chose to shut down instruments.")
            self._shutdown_all_instruments()
        elif user_choice == 'no':
            # User chose not to shut down
            print("User chose NOT to shut down instruments.")
        else:
            # Timeout - use YAML config
            print("Timeout reached. Using YAML configuration for shutdown.")
            self._shutdown_instruments(params)

        if hasattr(self, '_pcr_dialog') and self._pcr_dialog is not None:
            self._pcr_dialog.accept()
            self._pcr_dialog = None

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

    def remove_peak(self, data, index=None):
        """Truncate the correlation's center (0-delay) bin.

        The correlation histogram includes an artificial spike at ~0 delay because
        every event contributes to the center bin. For visualization, truncate the
        center bin height down to the height of the second-largest bin (i.e. the
        maximum of all *other* bins).

        Parameters
        ----------
        data : array-like
            1D histogram counts.
        index : array-like, optional
            1D x-axis positions (ps) corresponding to `data`. If provided, the
            center bin is chosen as the bin(s) closest to 0.
        """
        if data is None:
            return data

        arr = numpy.asarray(data)
        if arr.ndim != 1 or arr.size < 3:
            return arr

        # Work on a copy so we don't mutate persistent accumulation arrays.
        out = arr.copy()

        try:
            if index is not None:
                idx_arr = numpy.asarray(index)
                if idx_arr.shape[0] == out.shape[0]:
                    abs_idx = numpy.abs(idx_arr)
                    min_abs = numpy.nanmin(abs_idx)
                    center_idxs = numpy.where(abs_idx == min_abs)[0]
                else:
                    center_idxs = numpy.array([out.size // 2])
            else:
                center_idxs = numpy.array([out.size // 2])

            if center_idxs.size == 0:
                return out

            mask = numpy.ones(out.shape[0], dtype=bool)
            mask[center_idxs] = False
            if not numpy.any(mask):
                return out

            # "Second largest" relative to an oversized center peak.
            max_other = numpy.nanmax(out[mask])
            if not numpy.isfinite(max_other):
                return out

            # Only truncate if the center is larger.
            out[center_idxs] = numpy.where(out[center_idxs] > max_other, max_other, out[center_idxs])
            return out
        except Exception:
            # Never let display-only truncation break acquisition.
            return out

    def draw(self):
        '''Handler for the timer event to update the plots'''
        if self.running:
            # Counter
            #data = self.counter.getData() * self.getCouterNormalizationFactor()
            histblock_depth = self._get_histogram_block_count()
            
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

            current_hist_index = None
            current_hist_data = None

            if self.histogram_sync is not None and not self.histogram_sync.isRunning():
                self.histogram_block_running = False

                current_hist_index = self.correlation.getIndex()[self.masked_hist_bins:]
                start_counts = 0.0
                try:
                    if self.histogram_start_counter is not None:
                        start_data = self.histogram_start_counter.getData()
                        start_counts = float(start_data[0][0])
                except Exception:
                    start_counts = 0.0

                block_duration_s = self.histogram_block_duration_ms / 1000.0
                measured_start_rate_hz = 0.0
                if block_duration_s > 0:
                    measured_start_rate_hz = start_counts / block_duration_s
                # if self.active_channels:
                #     print(
                #         f"Measured start rate on channel {self.active_channels[0]}: "
                #         f"{measured_start_rate_hz:.3f} Hz "
                #         f"({start_counts:.0f} starts in {block_duration_s:.3f} s block)"
                #     )

                q = self.correlation.getData()[self.masked_hist_bins:]
                self.histBlock[self.BlockIndex] = q
                if self.histStartCounts is None:
                    self.histStartCounts = numpy.zeros(histblock_depth, dtype='float')
                self.histStartCounts[self.BlockIndex] = start_counts

                if self.ui.IntType.currentText() == "Discrete":
                    if self.BlockIndex == 0:
                        self.persistentData = numpy.sum(self.histBlock, axis=0)
                        self.persistentStartCounts = float(numpy.sum(self.histStartCounts))
                    else:
                        if self.IntType == "Rolling":
                            self.persistentData = numpy.sum(self.histBlock, axis=0)
                            self.persistentStartCounts = float(numpy.sum(self.histStartCounts))
                            self.BlockIndex = 1
                            self.IntType = "Discrete"
                    currentCounts = self.persistentData
                    currentStartCounts = self.persistentStartCounts
                else:
                    currentCounts = numpy.sum(self.histBlock, axis=0)
                    currentStartCounts = float(numpy.sum(self.histStartCounts))
                self.IntType = self.ui.IntType.currentText()

                current_hist_data = self._normalize_histogram_counts_to_rate(
                    currentCounts,
                    currentStartCounts,
                )
                current_hist_data = self.remove_peak(current_hist_data, index=current_hist_index)

                self.BlockIndex = self.BlockIndex + 1
                self._start_histogram_block_acquisition()

            if not self.plt_correlation:
                self._initialize_correlation_plot()

            if self.plot_mode == 'special_correlation' and self.special_corr is not None:
                special_index, special_data = self._get_special_corr_plot_data()
                self.plt_correlation[0].set_xdata(special_index * 1e-3)
                self.plt_correlation[0].set_ydata(self._get_special_corr_display_data(special_data))
                self._apply_correlation_axis_format(
                    'Interarrival Histogram of filtered_on',
                    'Counts',
                )
                self._autoscale_special_correlation_axis(special_index, special_data)
            elif current_hist_data is not None and current_hist_index is not None:
                self.plt_correlation[0].set_xdata(current_hist_index * 1e-3)
                self.plt_correlation[0].set_ydata(current_hist_data)
                self.correlationAxis.relim()
                self._apply_correlation_axis_format(
                    'Histogram between A and B',
                    'Instantaneous Count Rate (Hz)',
                )
                self.correlationAxis.autoscale_view(True, True, True)

            self.canvas.draw_idle()
            self.canvas.draw_idle()





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
