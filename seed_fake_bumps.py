"""
Drops a few sample speed bumps straight into speedbump.db, near
Bengaluru (12.9716, 77.5946) — the same default location your Flutter
app's map centers on. Run this once after starting the backend (or
before) so your demo map shows bump markers immediately, without
needing to actually drive GPS data through the detection pipeline.

Run from the same folder as main.py:
    python seed_fake_bumps.py
"""
import sqlite3

DATABASE = "speedbump.db"

# Small offsets around 12.9716, 77.5946 so markers appear nearby on the map.
fake_bumps = [
    {"lat": 12.9726, "lng": 77.5956, "speed_before": 42.0, "minimum_speed": 14.0,
     "speed_drop_pct": 66.7, "confidence": 0.91, "confirmed": 1, "detection_count": 3},
    {"lat": 12.9706, "lng": 77.5936, "speed_before": 35.0, "minimum_speed": 12.0,
     "speed_drop_pct": 65.7, "confidence": 0.84, "confirmed": 1, "detection_count": 2},
    {"lat": 12.9740, "lng": 77.5970, "speed_before": 50.0, "minimum_speed": 20.0,
     "speed_drop_pct": 60.0, "confidence": 0.77, "confirmed": 0, "detection_count": 1},
]

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS bumps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        speed_before REAL,
        minimum_speed REAL,
        speed_drop_pct REAL,
        confidence REAL,
        confirmed INTEGER DEFAULT 0,
        detection_count INTEGER DEFAULT 1
    )
""")

for b in fake_bumps:
    cursor.execute("""
        INSERT INTO bumps
        (timestamp, lat, lng, speed_before, minimum_speed, speed_drop_pct, confidence, confirmed, detection_count)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        b["lat"], b["lng"], b["speed_before"], b["minimum_speed"],
        b["speed_drop_pct"], b["confidence"], b["confirmed"], b["detection_count"],
    ))

connection.commit()
connection.close()

print(f"Inserted {len(fake_bumps)} fake bumps into {DATABASE}")
