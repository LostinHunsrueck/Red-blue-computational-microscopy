# Red-blue-computational-microscopy
Computational Phase contrast microscopy using red-blue led light  usb camera input python openCV on normal microscope

=== CONTROLS ===
ESC  - Quit
p    - Cycle view (phase / overlay / raw)
n    - Toggle noise reduction
m    - Toggle scale bar
c    - Cycle contrast modes
w    - Toggle soft white background
e    - Cycle emboss modes
i    - Cycle improvement (off / micro / medium / strong)
v    - Cycle intensity cutoff (off / 20% / 40%)
s    - Save normal frame (PNG)
q    - Super-resolution still (3-frame average, PNG)
1    - Start/stop recording (25 FPS WMV)
2    - Start/stop recording (time-lapse 3 FPS → 25 FPS)
3    - Start/stop recording (time-lapse 10 FPS → 25 FPS)
================


## What You Need
* An industrial USB camera (UVC compatible)
* Red and blue illumination setup
* Python 3 installed on your computer

import cv2
import numpy as np
from datetime import datetime
import os
import sys

## How to Run
1. Download this repository (click the green **Code** button -> **Download ZIP**).
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Open the `.py` file in IDLE (or your terminal) and run it.
