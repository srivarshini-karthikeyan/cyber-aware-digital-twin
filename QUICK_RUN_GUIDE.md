# 🚀 Quick Run Guide - See Everything Working!

## Step 1: Activate Your Virtual Environment

```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

## Step 2: Install Dependencies (if not done)

```bash
pip install -r requirements.txt
```

## Step 3: Run the Quick Demo (Recommended First!)

This shows all features in action:

```bash
python run_demo.py
```

**What you'll see:**
- ✅ System initialization
- ✅ Ensemble model training
- ✅ Real-time stream processing (50 samples)
- ✅ Attack detection
- ✅ Trust assessment
- ✅ Statistics and metrics

## Step 4: Run Full Validation Experiment

For comprehensive ground-truth validation:

```bash
python run_validation.py
```

**What you'll see:**
- ✅ Complete validation with 200 test samples
- ✅ Overall performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Per-attack performance breakdown
- ✅ Confusion matrix
- ✅ Saved results in `outputs/research/`

## Step 5: Run Original Demo Scripts

```bash
# Original attack scenario demo
python example_demo.py

# Original dashboard
python src/dashboard.py
```

## Step 6: Check Outputs

```bash
# View generated plots
ls outputs/plots/

# View research outputs
ls outputs/research/

# View reports
ls outputs/reports/
```

## Step 7: Explore Research Features

### Real-Time Processing
```python
from src.research_dashboard import ResearchGradeDashboard

dashboard = ResearchGradeDashboard()
sensor_data = {'level': 500.0, 'valve': 1, 'pump': 0}
result = dashboard.process_real_time_stream(sensor_data)
print(result)
```

### Generate Attack Scenarios
```python
from src.attack_generator import AttackGenerator

gen = AttackGenerator()
scenario = gen.generate_sensor_spoofing_attack(duration=60)
print(scenario['attack_id'])
```

### View Trust Assessment
```python
from src.enhanced_trust import EnhancedTrustAssessment

trust = EnhancedTrustAssessment()
# ... process data ...
summary = trust.get_trust_summary()
print(summary)
```

## Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in project root
cd /path/to/iitkanpur
python run_demo.py
```

### Issue: TensorFlow not found
```bash
pip install tensorflow==2.13.0
```

### Issue: sklearn errors
```bash
pip install scikit-learn==1.3.0
```

### Issue: Model not trained
- The demo scripts auto-train with dummy data
- For real training, provide SWaT dataset in `data/raw/`

## Expected Output

When you run `run_demo.py`, you should see:

```
======================================================================
🛡️  RESEARCH-GRADE CYBER-AWARE DIGITAL TWIN - DEMO
======================================================================

📦 Initializing Research-Grade Dashboard...
✅ Dashboard initialized!

🧠 Training Ensemble Models (with dummy data)...
✅ Models trained!

🔄 Simulating Real-Time Stream Processing...
   Processing 50 samples...
   Sample 0: Anomaly=False, Confidence=0.15, Trust=0.95
   Sample 10: Anomaly=True, Confidence=0.82, Trust=0.45
   ...

✅ Stream processing complete!

📊 Statistics:
   Total samples processed: 50
   Anomalies detected: 30
   Average detection latency: 2.5s

🔒 Trust Assessment Summary:
   Current trust: 0.85
   Trust state: medium
   Attack periods: 1
   Recovery periods: 1

...
```

## Next Steps

1. ✅ Run `run_demo.py` - See everything working
2. ✅ Run `run_validation.py` - Full validation experiment
3. ✅ Check `outputs/` - View generated plots and reports
4. ✅ Read `RESEARCH_GRADE_UPGRADE.md` - Understand all features
5. ✅ Read `docs/final_report.md` - Complete documentation

## Need Help?

- Check `README.md` for overview
- Check `RESEARCH_GRADE_UPGRADE.md` for feature details
- Check `QUICKSTART.md` for installation help
- Check `docs/` for detailed documentation

---

**Happy Running! 🚀**
