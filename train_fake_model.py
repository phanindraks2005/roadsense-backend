"""
Generates a working (but not real-world-trained) speed_bump_model.pkl
so the RoadSense backend can run. This is synthetic training data —
fine for a demo, but the predictions are not based on real road data.
If you ever get the real model file from your teammate, just drop it
in this same folder and replace this one.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)

FEATURE_COLUMNS = [
    "speed_before",
    "minimum_speed",
    "speed_drop",
    "speed_drop_pct",
    "deceleration",
    "recovery_time",
    "jerk",
]

n = 2000
rows = []
labels = []

for i in range(n):
    is_bump = np.random.rand() < 0.5

    speed_before = np.random.uniform(10, 60)

    if is_bump:
        # Sharp drop and quick recovery -> looks like a real speed bump
        drop_pct = np.random.uniform(25, 70)
        minimum_speed = speed_before * (1 - drop_pct / 100)
        recovery_time = np.random.uniform(1, 4)
        deceleration = (speed_before - minimum_speed) / np.random.uniform(0.5, 2)
        jerk = np.random.uniform(5, 20)
    else:
        # Gradual slowdown (traffic, turning, braking for a signal) -> not a bump
        drop_pct = np.random.uniform(5, 35)
        minimum_speed = speed_before * (1 - drop_pct / 100)
        recovery_time = np.random.uniform(3, 10)
        deceleration = (speed_before - minimum_speed) / np.random.uniform(2, 6)
        jerk = np.random.uniform(0, 6)

    speed_drop = speed_before - minimum_speed

    rows.append({
        "speed_before": speed_before,
        "minimum_speed": minimum_speed,
        "speed_drop": speed_drop,
        "speed_drop_pct": drop_pct,
        "deceleration": deceleration,
        "recovery_time": recovery_time,
        "jerk": jerk,
    })
    labels.append(1 if is_bump else 0)

df = pd.DataFrame(rows)[FEATURE_COLUMNS]
y = np.array(labels)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(df, y)

joblib.dump(model, "speed_bump_model.pkl")
print("Saved speed_bump_model.pkl")
print("Train accuracy:", model.score(df, y))
