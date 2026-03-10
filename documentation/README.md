# Data Analysis Playground

Python workspace for rosbag analysis, data visualization, and experimentation.

## Structure

```
analysis/
├── notebooks/          # Jupyter notebooks for interactive analysis
├── scripts/            # Python scripts for batch processing
├── data/              # Processed/exported data (gitignored)
└── requirements.txt   # Python dependencies
```

## Setup

```bash
# Inside the Docker container
cd /workspace/analysis
pip install -r requirements.txt

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access at: `http://localhost:8888`

## Quick Examples

```python
# Load rosbag
import rosbag
bag = rosbag.Bag('/workspace/rosbags/2025-12-17-16-02-22.bag')

# Analyze radar topics
for topic, msg, t in bag.read_messages(topics=['/ti_mmwave/radar_scan']):
    # Process radar data
    pass
```
