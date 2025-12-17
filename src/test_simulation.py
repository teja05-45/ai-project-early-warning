import pandas as pd
from simulation import WhatIfSimulator

df = pd.read_csv("data/raw/project_delivery_risk.csv")

sample_row = df.iloc[0]

simulator = WhatIfSimulator()

result = simulator.simulate(
    sample_row,
    actions={
        "planned_story_points": -0.2,   # reduce scope by 20%
        "dependency_blockers": -0.5,    # reduce blockers by 50%
        "avg_work_hours": -0.25         # reduce overtime
    }
)

print(result)
