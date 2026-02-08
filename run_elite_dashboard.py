"""
Run Elite Production Dashboard
Advanced IDS System UI

Usage:
    python run_elite_dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.production_dashboard import ProductionDashboard


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ELITE PRODUCTION DASHBOARD")
    print("=" * 70)
    print()
    print("📊 Features:")
    print("   ✓ Black/Purple Royal Theme")
    print("   ✓ 5+ Dynamic Visualizations")
    print("   ✓ Real-time Live Data")
    print("   ✓ All Validation Metrics")
    print("   ✓ Collapsible Sidebar")
    print("   ✓ Advanced UI Components")
    print()
    print("🌐 Access at: http://127.0.0.1:8050")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
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
