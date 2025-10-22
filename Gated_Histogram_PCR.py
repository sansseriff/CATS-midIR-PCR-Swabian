# from snspd_measure.inst.keysight33622A import keysight33622A

# PySide2 for the UI
from PySide2.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide2.QtCore import QTimer
from PySide2.QtGui import QPalette, QColor

from snspd_measure.inst.sim900 import sim928

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
from TimeTagger import Coincidences, Histogram2D, Counter, Correlation, createTimeTagger, freeTimeTagger, Histogram, FileWriter, FileReader, TT_CHANNEL_FALLING_EDGES, Resolution, DelayedChannel, GatedChannel, Countrate
from time import sleep

import json
import csv

       

# from awgClient import AWGClient

class CoincidenceExample(QMainWindow):
    ''' Small example of how to create a UI for the TimeTagger with the PySide2 framework'''

    def __init__(self, tagger):
        '''Constructor of the coincidence example window
        The TimeTagger object must be given as arguments to support running many windows at once.'''

        # Create the UI from the designer file and connect its action buttons
        super(CoincidenceExample, self).__init__()
        self.ui = Ui_CoincidenceExample()
        self.ui.setupUi(self)
        self.ui.PCRButton.clicked.connect(self.gated_PCR)
        # self.ui.triggerScanButton.clicked.connect(self.triggerJitterScan)
        # self.ui.clearButton.clicked.connect(self.dBScanAWG)
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
        self.histBlock = numpy.zeros((int(self.ui.IntTime.value()*5),self.ui.correlationBins.value()))

        self.buffer = numpy.zeros((1,self.ui.correlationBins.value()))
        self.buffer_old = numpy.zeros((1, self.ui.correlationBins.value()))


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

        #print("coincidences on ch", self.coincidences.getChannels())

        # self.delayed_start = DelayedChannel(self.tagger, self.active_channels[0], 20e6)

        # self.delayed_stop = DelayedChannel(self.tagger, self.active_channels[0], int(23e6))
        # print(self.delayed_stop)

        # for us right now (oct 9 2024), self.active_channels[2] (3rd row) is 5, which is the snspd
        self.filtered = GatedChannel(self.tagger, self.active_channels[2], self.active_channels[0], -self.active_channels[0])
        self.delay_1_start = DelayedChannel(self.tagger, self.active_channels[0], int(150e9))
        self.delay_1_stop = DelayedChannel(self.tagger, self.active_channels[0], int(300e9))

        # self.delay_1_start = DelayedChannel(self.tagger, self.active_channels[0], int(0e9))
        # self.delay_1_stop = DelayedChannel(self.tagger, self.active_channels[0], int(800e9))

        self.delay_2_start = DelayedChannel(self.tagger, self.active_channels[0], int(800e9))

        self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(950e9))
        # self.delay_2_stop = DelayedChannel(self.tagger, self.active_channels[0], int(800e9))


        # thermal source on
        self.filtered_on = GatedChannel(self.tagger, self.active_channels[2], self.delay_1_start.getChannel(), self.delay_1_stop.getChannel())

        # thermal source off
        self.filtered_off = GatedChannel(self.tagger, self.active_channels[2], self.delay_2_start.getChannel(), self.delay_2_stop.getChannel())


        # Measure the correlation between A and B
        self.correlation = Histogram(
            self.tagger,
            #self.a_combined.getChannel(),
            #self.b_combined.getChannel(),
            self.active_channels[1],
            self.filtered.getChannel(),

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
        index = self.correlation.getIndex()
        #data = self.correlation.getDataNormalized()
        data = self.correlation.getData()
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

    '''
    def saveHistogram(self):
        print("saving persistant histogram data")
        # numpy.save("histogram_data.npy", self.persistentData
        int_time = input("how long do you want to integrate for?")
        time.sleep(int(int_time))
        data = self.correlation.getData()
        #array = self.persistentData
        json_string = json.dumps(data.tolist())
        with open('Data_output.json','w') as file:
           file.write(json_string)
        print("finished")
    '''
    def saveHistogram(self):
         wf = keysight33622A('10.7.0.187')
         wf.connect()
         V_pp = 0.090 #in V 
         Start = 0.055
         Stop = 0.090
         Step = 0.005
         offset = numpy.arange(Start,Stop+0.005,Step) # in V 
         offset = numpy.append(offset,0.093)
         I_det = []
         c = input('How Long Do You Want to Integrate For?: ')
         int_time = int(c)*(numpy.ones((len(offset)-1,),dtype=int))
         int_time = numpy.append(int_time,2*int(c))
         wv = input('What wavelength is this for?: ')
        

        # Determining Bias at the Detector
         for i in offset:
             V_det = ((V_pp/2) + i)/100
             I_det.append(((V_det/50)*1000000).round(4))  #in uA

        #Setting Up Filter Channel 
         wf.channels_on()
         wf.phase_zero()
         wf.phase_sync()
         wf.filter_channel(-45,3000)

        # making array for plotting 
         I_b = numpy.asarray(I_det,dtype = 'float')

         for i in range(len(I_b)):
             wf.gating_channel(offset[i])
             wf.phase_sync()
             time.sleep(1)
           
             print("starting "+str(I_b[i])+' histogram')
        # numpy.save("histogram_data.npy", self.persistentData
        #int_time = input("how long do you want to integrate for?")
             time.sleep(int_time[i])
             data = self.correlation.getData()
        #array = self.persistentData
             json_string = json.dumps(data.tolist())
             with open(wv+'_1um_R1C4_'+str(I_b[i])+'uA_GatedRelLat.json','w') as file:
                file.write(json_string)
             print("finished")

         wf.channels_off()
         wf.disconnect()
    

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


    def gated_PCR(self):

        # wf = keysight33622A('10.7.0.187')
        # wf.connect()

        source = sim928('/dev/ttyUSB0', 2, 1)
        source.connect()
        source.turnOn()


        V_pp = 0.090 #in V 
        Start = float(input('Start(V): '))
        Stop = float(input("End(V): "))
        Step = float(input("Step(V): "))
        offset = numpy.arange(Start,Stop+0.005,Step) # in V 
        # f = numpy.arange(3000,12000,1000)
        I_det = []
        waiting = input("Enter Integration Time in seconds in form '#(secs)e12': ")
        int_time = float(waiting)
        Counts = []
        fig = plt.figure()
        x_vals = []
        y_vals = []
        darkcounts = []

        # Determining Bias at the Detector
        for i in offset:
            # V_det = ((V_pp/2) + i)/100

            I_det.append(((i/1.02e6)*1e6).round(4))  #in uA

        #Setting Up Filter Channel 
        # wf.channels_on()
        # wf.phase_zero()
        # wf.phase_sync()
        # wf.filter_channel(-45,3000)

        # making array for plotting 
        I_b = numpy.asarray(I_det,dtype = 'float')

        for i in range(len(I_b)):
        # for bias in range(offset):
            source.setVoltage(offset[i])
            print("current voltage: ", offset[i])


            #wf.filter_channel(10,f[i])
            # wf.gating_channel(offset[i])
            # wf.phase_sync()
            time.sleep(1)
            # cr = Countrate(self.tagger, [self.filtered.getChannel()])

            cr_on = Countrate(self.tagger, [self.filtered_on.getChannel()])
            cr_off = Countrate(self.tagger, [self.filtered_off.getChannel()])

            #cr = Countrate(self.tagger, [-5])
            #cr = Countrate(self.tagger, [9])
            cr_on.startFor(int(int_time),clear = True) #in picoseconds
            cr_off.startFor(int(int_time),clear = True) #in picoseconds
            cr_on.waitUntilFinished()
            cr_off.waitUntilFinished()

            clicks_on = cr_on.getCountsTotal()
            clicks_off = cr_off.getCountsTotal()


            Counts.append(clicks_on[0] - clicks_off[0])
            x_vals.append(I_b[i])
            y_vals.append(Counts[i])
            darkcounts.append(clicks_off[0])
            plt.scatter(x_vals,y_vals, color="darkred", s=10)
            plt.scatter(x_vals, darkcounts, color="black", s=10)
            plt.plot(x_vals,y_vals, color="darkred")
            plt.plot(x_vals, darkcounts, color="black")
            plt.title("Gated PCR Curve")
            plt.xlabel("Bias Current (uA)")
            plt.ylabel("Counts")
            plt.draw()
            plt.pause(1)

            if i < (len(I_b)-1):
                # don't clear at the end so the figure is visible and 
                # you can interact with it
                fig.clear()
    
        plt.show() 

        print('Finished PCR Curve')
        # wf.channels_off()
        # wf.disconnect()

        # Generate CSV file with 1st column as I_b and 2nd as counts 
        filename = input('Save Data Name (add .csv at end): ')

        with open(filename,'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Bias_Current','counts','darkcounts'])
            for i in range(len(I_b)):
                csvwriter.writerow([I_b[i], Counts[i],darkcounts[i]])
       
        
        
        
        ##self.count_rate_measurment.startFor(int(1*1e12), clear=True) self.count_rate_measurment = 
        
        #print(self.count_rate_measurment.getData())

        # print("working")
        #voltage_source = sim928("/dev/ttyUSB0", 2, 3)
        #voltage_source.connect()

        #start_voltage = input("input start voltage: ")
        #end_voltage = input("input end voltage: ")
        #step_voltage = input("input voltage step: ")
        #int_time = input("input integration time per step: ")
        
        #values = numpy.arange(start_voltage, end_voltage, step_voltage).tolist()
        #values = [round(i,3) for i in values]

        #self.count_rate_measurment = Countrate(self.tagger, self.filtered.getChannel())

        #count_rate = []

        # self.filtered.getChannel()

        #plt.ion()
        #figure, ax = plt.subplots(figsize=(10, 8))
        #line1, = ax.plot(index[:len(data_plot)],data_plot)



        #for voltage in values:
            #voltage_source.setVoltage(voltage)
            #sleep(0.2)
            #self.count_rate_measurment.startFor(int(int_time*1e12), clear=True)
            #sleep(int_time*1.1) # I'm not sure if startFor() is blocking...
            #data = self.count_rate_measurment.getData()
            #print("data shape: ", numpy.shape(data))
            #count_rate.append(data)


            #line1.set_xdata(values[:len(count_rate)])
            #line1.set_ydata(count_rate)

            #figure.canvas.draw()
            #figure.canvas.flush_events()
            #plt.xlim((min(values)*.9,max(values)*1.1))
            #plt.ylim((0,max(count_rate)*1.1))

        #name = input("save data name? ")
        #data_dict = {"voltages": values, "countrate": count_rate}
        #json_string = json.dumps(data_dict)
        #with open(f'{name}.json', 'w') as outfile:
            #json.dump(json_string, outfile)

        #plt.ioff()

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
            if self.BlockIndex >= int(self.ui.IntTime.value()*5):
                self.BlockIndex = 0

            data = self.counter.getData() * self.getCouterNormalizationFactor()
            #print("length of data", len(data))
            #print("###########")
            for data_line, plt_counter in zip(data, self.plt_counter): # loop though coincidences, Ch1, Ch2
                plt_counter.set_ydata(data_line)
            self.counterAxis.relim()
            self.counterAxis.autoscale_view(True, True, True)


            index = self.correlation.getIndex()

            q = self.correlation.getData()
            self.histBlock[self.BlockIndex] = q
            #print(numpy.sum(q))

            if self.ui.IntType.currentText() == "Discrete":
                if self.BlockIndex == 0:
                    self.persistentData = numpy.sum(self.histBlock, axis=0)
                else:
                    if self.IntType == "Rolling":

                        # first time changing from Rolling to Discrete
                        self.persistentData = numpy.sum(self.histBlock, axis=0)
                        self.BlockIndex == 1
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
