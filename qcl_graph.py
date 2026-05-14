


import json
import matplotlib.pyplot as plt
import numpy as np


with open("/home/cats/Documents/measurements/QCL_testing/4.3.2026/HIST__QCL_63_8.39_64.3_500_80__FRIDGE_260__SAVE_600_0.012__DETECTOR_0.08.json") as f:
    data = json.load(f)


plt.plot(np.array(data["x_axis_ps"])/1e6, data["instantaneous_count_rate_hz"])


plt.ylabel("Instantaneous Count Rate (Hz)")
plt.xlabel("Time (microseconds)")


plt.ylim(0, 80000)

plt.show()
print(data)