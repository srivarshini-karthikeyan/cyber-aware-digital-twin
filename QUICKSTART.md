# 🚀 Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python -c "import tensorflow; import pandas; import matplotlib; print('✅ All dependencies installed')"
```

## Running the System

### Option 1: Run Full Pipeline (Recommended)

```bash
python src/dashboard.py
```

This will:
- Process SWaT data (if provided)
- Train GenAI model (if data available)
- Generate and analyze all attack scenarios
- Create visualizations
- Generate reports

### Option 2: Run Specific Attack Scenario

```bash
python src/dashboard.py --attack-type sensor_spoofing
```

Available attack types:
- `sensor_spoofing`
- `slow_manipulation`
- `frozen_sensor`
- `delayed_response`

### Option 3: Run Demo Script

```bash
python example_demo.py
```

This runs pre-configured demos without requiring SWaT data.

## Using SWaT Dataset

1. **Download SWaT dataset** (if available)
2. **Place CSV files** in `data/raw/` directory
3. **Run with data file**:
```bash
python src/dashboard.py --data-file data/raw/swat_data.csv
```

## Output Files

After running, check:
- **Plots**: `outputs/plots/`
- **Reports**: `outputs/reports/`
- **Results**: `outputs/results/`

## Configuration

Edit `config.yaml` to customize:
- Subsystem thresholds
- GenAI model parameters
- Attack generation settings
- Visualization options

## Troubleshooting

### Issue: TensorFlow not found
```bash
pip install tensorflow==2.13.0
```

### Issue: Model not trained
- The system can run without training (uses dummy data)
- For real training, provide SWaT dataset

### Issue: Import errors
```bash
# Make sure you're in project root
cd /path/to/iitkanpur
python src/dashboard.py
```

## Next Steps

1. Read `docs/architecture.md` for system design
2. Read `docs/attack_scenarios.md` for attack details
3. Read `docs/final_report.md` for complete documentation
4. Explore `src/` directory for code

---

**Status**: ✅ Ready to Run | 🎯 Production Ready
