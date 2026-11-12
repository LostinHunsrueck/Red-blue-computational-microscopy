# Red-blue-computational-microscopy
Computational Phase contrast microscopy using red-blue led light  usb camera input python openCV on normal microscope

## Controls

| Key | Action |
| :--- | :--- |
| <kbd>ESC</kbd> | Quit |
| <kbd>p</kbd> | Cycle view (`phase` / `overlay` / `raw`) |
| <kbd>n</kbd> | Toggle noise reduction |
| <kbd>m</kbd> | Toggle scale bar |
| <kbd>c</kbd> | Cycle contrast modes |
| <kbd>w</kbd> | Toggle soft white background |
| <kbd>e</kbd> | Cycle emboss modes |
| <kbd>i</kbd> | Cycle improvement (`off` / `micro` / `medium` / `strong`) |
| <kbd>v</kbd> | Cycle intensity cutoff (`off` / `20%` / `40%`) |
| <kbd>s</kbd> | Save normal frame (`PNG`) |
| <kbd>q</kbd> | Super-resolution still (3-frame average, `PNG`) |
| <kbd>1</kbd> | Start/stop recording (25 FPS `WMV`) |
| <kbd>2</kbd> | Start/stop recording (time-lapse 3 FPS -> 25 FPS) |
| <kbd>3</kbd> | Start/stop recording (time-lapse 10 FPS -> 25 FPS) |


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
