import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

import sys
import numpy as np
from rosbag_loader.loader import load_bag_topics

bag_data = load_bag_topics('../rosbags/circle_forward_2025-12-17-17-37-38.bag', verbose=False)
intensities = []
for f in bag_data.radar_velocity:
    if f.intensities is not None:
        intensities.extend(f.intensities)
intensities = np.array(intensities)
print("Min:", np.min(intensities))
print("Mean:", np.mean(intensities))
print("Median:", np.median(intensities))
print("90th:", np.percentile(intensities, 90))
print("95th:", np.percentile(intensities, 95))
print("99th:", np.percentile(intensities, 99))
print("Max:", np.max(intensities))
