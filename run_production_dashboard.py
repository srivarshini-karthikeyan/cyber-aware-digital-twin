"""
Run Production Dashboard
Elite IDS System UI

Usage:
    python run_production_dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.production_dashboard import ProductionDashboard

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting GenTwin Production Dashboard")
    print("=" * 70)
    print()
    print("📊 Dashboard Features:")
    print("   - Real-time monitoring")
    print("   - Black/Purple royal theme")
    print("   - Comprehensive metrics visualization")
    print("   - Live alert stream")
    print("   - System component status")
    print()
    print("🌐 Access dashboard at: http://127.0.0.1:8050")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        dashboard = ProductionDashboard()
        dashboard.run(debug=True, port=8050)
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
