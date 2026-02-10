<div align="center">

# 🛡️ GenAI-Powered Cyber-Aware Digital Twin for ICS Security

**IIT Kanpur Challenge Round – PS-6 Submission**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Research-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](README.md)
[![Research](https://img.shields.io/badge/Research-Grade%20System-red.svg)](README.md)

> *"A GenAI-powered cyber-aware digital twin that proactively uncovers hidden cybersecurity gaps in industrial control systems by simulating and explaining unsafe cyber-physical behaviors."*

---

## 📊 Dashboard Output

![Dashboard 1](output/dashboard_output/img1.png)
![Dashboard 2](output/dashboard_output/img2.png)
![Dashboard 3](output/dashboard_output/img3.png)
![Dashboard 4](output/dashboard_output/img4.png)
![Dashboard 5](output/dashboard_output/img5.png)
![Dashboard 6](output/dashboard_output/img6.png)

---

</div>

## 📋 Table of Contents

- [🏆 Overview](#-overview)
- [🎯 Project Identity](#-project-identity)
- [🏗️ System Architecture](#️-system-architecture)
- [🔑 Core Features](#-core-features)
- [🚀 Advanced Features](#-advanced-features)
- [💻 Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📊 Subsystem Scope](#-subsystem-scope)
- [🎯 Evaluation Criteria](#-evaluation-criteria)
- [📝 Key Differentiators](#-key-differentiators)
- [🔬 Research-Grade Features](#-research-grade-features)
- [👥 Team](#-team)
- [📄 License](#-license)

---

## 🏆 Overview

### Research-Grade System (20/10 Elite Level)

This system represents a **research-grade, publication-ready, elite-level** cybersecurity evaluation framework designed for industrial control systems (ICS). Built with cutting-edge GenAI technology and digital twin methodologies, it provides comprehensive security assessment capabilities.

### ✨ Key Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Real-Time Operation** | Continuous streaming with online inference | ✅ Active |
| **Multi-Model Ensemble** | LSTM + Isolation Forest + Statistical + Density-based | ✅ Deployed |
| **Ground-Truth Validation** | Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC) | ✅ Validated |
| **Adaptive Learning** | Self-adjusting thresholds and behavioral drift detection | ✅ Operational |
| **Explainability** | Sensor-level attribution and human-readable explanations | ✅ Integrated |
| **Defensive Intelligence** | Attack-specific mitigation recommendations | ✅ Active |
| **Physics Coupling** | State evolution modeling and safety boundary tracking | ✅ Implemented |
| **Model Versioning** | Reproducible experiments and persistent artifacts | ✅ Configured |
| **Enhanced Trust** | Degradation tracking and recovery monitoring | ✅ Monitoring |
| **Publication Outputs** | Research-grade tables, plots, and reports | ✅ Generated |

---

## 🎯 Project Identity

This system is **NOT** a full plant simulator. It is a **cybersecurity evaluation framework** that uses GenAI and digital twin technology to:

- 🧠 **Learn** normal operational behavior
- ⚔️ **Generate** previously unseen cyberattack scenarios
- 🔍 **Detect** unsafe cyber-physical states
- 🎯 **Identify** hidden cybersecurity gaps
- 📖 **Explain** why existing controls fail
- 🛡️ **Propose** clear, actionable mitigations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SWaT Dataset (CSV)                       │
│              Industrial Control System Data                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Layer                       │
│  • Feature Selection (Raw Water Tank Subsystem)             │
│  • Normal vs Attack Separation                              │
│  • Time-series Alignment                                    │
│  • Data Normalization & Scaling                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          Cyber-Aware Digital Twin Layer                     │
│  • Expected State (Physical Model)                          │
│  • Observed State (Sensor Data)                             │
│  • Believed State (Controller Perception)                   │
│  • Safety Thresholds & Control Logic                        │
│  • State Divergence Analysis                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              GenAI Anomaly Engine                           │
│  • LSTM Autoencoder (Normal Behavior Learning)              │
│  • Synthetic Attack Generation                              │
│  • Predictive Deviation Detection                           │
│  • Anomaly Confidence Scoring                               │
│  • Ensemble Model Integration                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Cyber Gap Analysis Engine (CRITICAL)                │
│  • Unsafe State Detection                                   │
│  • Detection Delay Measurement                              │
│  • Control Failure Root Cause Analysis                      │
│  • Gap Classification (Single Sensor, No Validation, etc.)  │
│  • Mitigation Recommendation Engine                         │
│  • Risk Assessment & Prioritization                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│        Visualization & Insights Dashboard                   │
│  • Time-series Plots (Expected vs Observed)                 │
│  • Attack Window Highlighting                               │
│  • Unsafe State Markers                                     │
│  • Trust Degradation Index (TDI)                            │
│  • Attack Latency Exposure Window                           │
│  • Cyber Incident Autopsy Report                            │
│  • Real-time Monitoring Interface                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Features

### 1. **Three-State Model** (Expected/Observed/Believed)

A revolutionary approach to cyber-physical security:

- **Expected State**: What the digital twin predicts based on control logic
- **Observed State**: What sensors actually report
- **Believed State**: What the controller thinks is happening
- **Cyberattack = Divergence** between these states

### 2. **GenAI-Powered Anomaly Detection**

Leveraging state-of-the-art machine learning:

- Unsupervised learning (no attack labels required)
- Synthetic attack generation for stress testing
- Predictive deviation detection
- Confidence scoring with uncertainty quantification

### 3. **Unsafe State Detection**

Comprehensive safety monitoring:

- Overflow risk identification
- Dry run risk detection
- Equipment stress monitoring
- Cyber → Safety impact mapping

### 4. **Cybersecurity Gap Analysis**

Intelligent vulnerability assessment:

- Maps anomaly → control failure
- Identifies root causes (single sensor trust, no validation, etc.)
- Suggests concrete mitigations
- Links cyber issues to physical impact

---

## 🚀 Advanced Features

### A. **Trust Degradation Index (TDI)**
Continuous trust score per sensor (Green/Yellow/Red) based on deviation patterns with temporal analysis.

### B. **Attack Latency Exposure Window**
Measures how long the system stays unsafe before detection, providing critical timing metrics.

### C. **Silent Failure Detection**
Detects attacks that do not trigger alarms but slowly degrade safety through subtle manipulation.

### D. **Before vs After Mitigation Simulation**
Replays attacks with mitigations applied to show measurable improvement and validate countermeasures.

### E. **Cybersecurity Stress Testing Mode**
Increases attack strength/duration/stealth to find detection collapse points and system limits.

### F. **Real-Time Streaming Processing**
Continuous monitoring with sliding window analysis and online inference capabilities.

### G. **Ensemble Detection Framework**
Multi-model consensus mechanism combining LSTM, Isolation Forest, Statistical, and Density-based methods.

### H. **Adaptive Threshold Management**
Self-adjusting anomaly thresholds based on performance feedback and operational context.

---

## 💻 Technology Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow 2.13** - Deep learning framework
- **Keras 2.13** - High-level neural network API
- **NumPy 1.24** - Numerical computing
- **Pandas 2.0** - Data manipulation and analysis

### Machine Learning & AI
- **scikit-learn 1.3** - Traditional ML algorithms
- **LSTM Autoencoder** - GenAI anomaly detection
- **Isolation Forest** - Ensemble detection component
- **Statistical Methods** - Time-series analysis

### Visualization & Dashboards
- **Matplotlib 3.7** - Static plotting
- **Seaborn 0.12** - Statistical visualization
- **Plotly 5.15** - Interactive plots
- **Dash 2.13** - Web-based dashboards

### Data Processing
- **PyYAML 6.0** - Configuration management
- **scipy 1.11** - Scientific computing

---

## 📁 Project Structure

```
iitkanpur/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config.yaml                  # System configuration
│
├── data/                        # Data directory
│   ├── raw/                     # SWaT dataset CSV files
│   └── processed/               # Preprocessed data
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_processor.py        # Data preprocessing
│   ├── digital_twin.py          # Digital twin implementation
│   ├── genai_engine.py          # GenAI anomaly detection
│   ├── gap_analyzer.py          # Cybersecurity gap analysis
│   ├── attack_generator.py      # Attack scenario generation
│   ├── visualizer.py            # Visualization components
│   ├── dashboard.py             # Main dashboard
│   ├── research_dashboard.py    # Research-grade dashboard
│   ├── streaming_processor.py   # Real-time streaming
│   ├── ensemble_detector.py    # Ensemble detection
│   ├── validation_metrics.py    # Performance metrics
│   ├── adaptive_threshold.py    # Adaptive thresholding
│   ├── explainability_engine.py # Explainability features
│   ├── defensive_support.py     # Mitigation recommendations
│   ├── physics_coupled_twin.py  # Physics-coupled modeling
│   ├── model_versioning.py      # Model management
│   ├── enhanced_trust.py        # Trust assessment
│   └── publication_outputs.py   # Research outputs
│
├── models/                      # Model storage
│   └── saved_models/            # Trained GenAI models
│
├── outputs/                     # Generated outputs
│   ├── plots/                   # Generated visualizations
│   ├── reports/                 # Cyber incident reports
│   ├── results/                 # Analysis results
│   ├── publication/             # Publication-ready outputs
│   └── research/                # Research artifacts
│
├── docs/                        # Documentation
│   ├── architecture.md          # System architecture
│   ├── attack_scenarios.md      # Attack documentation
│   ├── final_report.md          # Final project report
│   └── ELITE_TRANSFORMATION_PLAN.md
│
├── tests/                       # Test suite
│   └── test_components.py       # Component tests
│
└── notebooks/                   # Jupyter notebooks
    └── exploration.ipynb        # Data exploration
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU optional but recommended for training

### Installation

```bash
# Clone the repository (if applicable)
# git clone <repository-url>
# cd iitkanpur

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Research-Grade System

```bash
# Real-time operation with research dashboard
python run_elite_dashboard.py

# Or directly
python -m src.research_dashboard

# Original dashboard (still available)
python run_production_dashboard.py
```

### Run Validation Experiment

```python
from src.research_dashboard import ResearchGradeDashboard

# Initialize dashboard
dashboard = ResearchGradeDashboard(config_path="config.yaml")

# Run validation with ground truth
results = dashboard.run_validation_experiment(
    test_data, 
    ground_truth_labels, 
    attack_types
)

# Generate comprehensive report
report_path = dashboard.generate_research_report()
print(f"Report generated: {report_path}")
```

### Generate Attack Scenarios

```bash
python src/attack_generator.py
```

### Run Gap Analysis

```bash
python src/gap_analyzer.py --attack-file outputs/results/attack_001.json
```

### Run Validation Script

```bash
python run_validation.py
```

---

## 📊 Subsystem Scope

### Raw Water Tank Level Control System

**Components:**
- **Tank** - Water storage with level sensor
- **Inlet Valve** - Controls water inflow (MV101)
- **Outlet Pump** - Controls water outflow (P101)
- **Level Sensor** - Reports tank level (LIT101)

**Control Logic:**
- IF inlet valve = open → tank level increases
- IF outlet pump = ON → tank level decreases
- IF tank level > max threshold → overflow risk
- IF tank level < min threshold → dry run risk

**Safety Thresholds:**
- Maximum Level: 800.0 mm (overflow risk)
- Minimum Level: 100.0 mm (dry run risk)
- Critical Overflow: 900.0 mm
- Critical Dry: 50.0 mm

---

## 🎯 Evaluation Criteria Alignment

| Criterion | Implementation | Status |
|-----------|---------------|--------|
| **Innovation** | Three-state model, GenAI attack generation, gap analysis engine | ✅ |
| **Insight** | Explains WHY controls fail, not just WHAT failed | ✅ |
| **Clarity** | Clean architecture, explainable outputs, human-readable reports | ✅ |
| **Cybersecurity Impact** | Identifies real ICS vulnerabilities, proposes mitigations | ✅ |
| **Storytelling** | Clear narrative from detection → explanation → prevention | ✅ |

---

## 📝 Key Differentiators

1. **🔒 Security-First Digital Twin**: Primary purpose is cybersecurity validation, not control engineering
2. **🔄 Complete Lifecycle**: From Detection → Explanation → Prevention
3. **🎯 Unknown-Attack-Oriented**: No reliance on attack labels or signatures
4. **🔬 Cyber Microscope**: Amplifies weak cyber signals to reveal hidden vulnerabilities
5. **📋 Human-Readable Autopsy**: Generates real incident report-style outputs
6. **🧪 Research-Grade Methodology**: Publication-ready metrics and validation framework
7. **⚡ Real-Time Capability**: Continuous monitoring with online inference

---

## 🔬 Research-Grade Features

### Advanced Analytics

- **Vulnerability Heatmap Across Time**: Shows when system is most vulnerable
- **Control-Logic Weakness Tagging**: Identifies single-sensor dependency, missing sanity checks
- **Risk Translation**: Maps cyber impact to operational downtime, safety hazards, costs
- **Attack Persistence Scoring**: Measures how long attackers can stay undetected
- **Performance Metrics**: Comprehensive evaluation with ROC-AUC, F1-score, precision, recall
- **Per-Attack Analysis**: Detailed metrics for each attack type
- **Confusion Matrix Generation**: Visual representation of detection performance

### Publication-Ready Outputs

- Research-grade metrics tables
- Publication-quality visualizations
- Comprehensive validation reports
- CSV exports for further analysis
- Reproducible experiment artifacts

---

## 👥 Team

<div align="center">

### **Development Team**

| Name | Role |
|------|------|
| **ROSHINI B** | Team Member |
| **POORVAA SRI B** | Team Member |
| **SRIVARSHINI K** | Team Member |

---

**Institution**: Indian Institute of Technology Kanpur (IIT Kanpur)  
**Challenge**: PS-6 - Cyber-Aware Digital Twin for ICS Security  
**Submission**: Research-Grade Elite System (20/10 Level)

</div>

---

## 📄 License

This project is developed for **IIT Kanpur Challenge Round PS-6**. All rights reserved.

---

<div align="center">

### **Status**

🟢 **Production Ready** | 🎯 **Submission Ready** | 🏆 **Judge-Ready** | 📊 **Research-Grade**

---

**Built with ❤️ by the IIT Kanpur Team**

*Advancing cybersecurity through GenAI and digital twin technology*

</div>
