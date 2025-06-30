# CalSol Cell Characterization Dashboard
Tools for characterizing li-ion cells

## Overview

This repository contains tools for battery cell testing and characterization:

- **RC3563 Reader**: Interface for Hioki RC3563 resistance meter, collecting resistance and voltage measurements
- **Battery Analyzer**: Arduino-based tool to measure battery capacity, resistance, and performance characteristics

## Dependencies

### Common Dependencies
```bash
pip install matplotlib numpy scipy pandas
```

### RC3563 Reader Dependencies
```bash
pip install kivy pyserial pandas
```

### Battery Analyzer Dependencies
```bash
pip install pyserial matplotlib numpy scipy
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CalSol/cell-characterization-dashboard.git
   cd cell-characterization-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### RC3563 Reader (rc3563reader.py)

Tool for interfacing with Hioki RC3563 resistance meter:

```bash
python rc3563reader.py
```

1. Enter the serial port (e.g., `/dev/ttyUSB0` on Linux or `COM3` on Windows)
2. Click "Connect" to establish connection with the meter
3. Readings will be displayed and automatically recorded
4. Click "Export to CSV" to save the collected data

### Battery Analyzer (capacity.py)

Tool for measuring battery capacity and internal resistance:

```bash
python capacity.py [--port PORT] [--baud BAUD] [--duration SECONDS] [--output PREFIX] [--no-plot] [--report]
```

Options:
- `--port`: Specify serial port (auto-detects if not specified)
- `--baud`: Set baud rate (default: 9600)
- `--duration`: Set data collection duration in seconds
- `--output`: Set output file prefix
- `--no-plot`: Skip generating plots
- `--report`: Generate a comprehensive text report

Example:
```bash
python capacity.py --port /dev/ttyACM0 --duration 3600 --report
```

## Output Files

The tools generate several output files:

- **CSV data files**: Raw measurement data
- **PNG plot files**: Visualizations of battery characteristics
- **TXT report files**: Comprehensive analysis results

## Arduino Setup

For the Battery Analyzer, an Arduino running the battery tester code is required. Ensure:

1. The Arduino is properly connected to your computer
2. The battery tester firmware is uploaded to the Arduino
3. The test battery is connected according to the circuit diagram (see documentation)

## Troubleshooting

If you encounter connection issues:
1. Check that the device is properly connected
2. Verify you have the correct port specified
3. Ensure you have proper permissions for the serial port
4. On Linux, you may need to add your user to the `dialout` group:
   ```bash
   sudo usermod -a -G dialout $USER
   ```