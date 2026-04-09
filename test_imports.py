import sys, pathlib
sys.path.insert(0, str(pathlib.Path(".").resolve()))
import streamlit
import plotly
from dashboard.insight_engine import generate_insights, generate_verdict
print("All imports OK")
ins = generate_insights(
    city="Bengaluru", scenario_id="exp1_baseline",
    final_usi=0.746, min_usi=0.63,
    final_trust=0.42, final_gini=0.0052, final_sr=0.81
)
print(f"{len(ins)} insights generated")
for i in ins:
    print(f"  [{i['level']}] {i['icon']} {i['title']}")

verdict = generate_verdict("Bengaluru", "exp5a_shock_nopolicy", 0.55)
print(f"\nVerdict: {verdict}")
