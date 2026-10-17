from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import joblib
import pandas as pd
import numpy as np
import math


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Speed Bump Detection API",
    description="Backend for GPS-based speed bump detection",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = "speedbump.db"
MODEL_PATH = "speed_bump_model.pkl"

DROP_THRESHOLD = 20          # minimum speed drop percentage
RECOVERY_SECONDS = 5         # recovery must happen within 5 seconds
ML_CONFIDENCE = 0.70         # minimum ML confidence
CROWD_RADIUS_METERS = 20     # distance for crowd validation
MIN_CROWD_DETECTIONS = 2     # minimum detections needed for confirmation


FEATURE_COLUMNS = [
    "speed_before",
    "minimum_speed",
    "speed_drop",
    "speed_drop_pct",
    "deceleration",
    "recovery_time",
    "jerk"
]


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # GPS readings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trip_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            speed REAL NOT NULL,
            trip_id TEXT DEFAULT 'default'
        )
    """)

    # Detected bumps
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

    # Add columns if an older database already exists
    try:
        cursor.execute(
            "ALTER TABLE trip_data ADD COLUMN trip_id TEXT DEFAULT 'default'"
        )
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute(
            "ALTER TABLE bumps ADD COLUMN confirmed INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute(
            "ALTER TABLE bumps ADD COLUMN detection_count INTEGER DEFAULT 1"
        )
    except sqlite3.OperationalError:
        pass

    connection.commit()
    connection.close()


init_database()


# ============================================================
# AUTO-SEED DEMO BUMPS (free hosting wipes the database on restart)
# ============================================================

def seed_demo_bumps_if_empty():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM bumps")
    count = cursor.fetchone()[0]

    if count == 0:
        demo = [
            (12.9726, 77.5956, 42.0, 14.0, 66.7, 0.91, 1, 3),
            (12.9706, 77.5936, 35.0, 12.0, 65.7, 0.84, 1, 2),
            (12.9740, 77.5970, 50.0, 20.0, 60.0, 0.77, 0, 1),
        ]
        for lat, lng, sb, ms, pct, conf, confirmed, dc in demo:
            cursor.execute("""
                INSERT INTO bumps
                (timestamp, lat, lng, speed_before, minimum_speed,
                 speed_drop_pct, confidence, confirmed, detection_count)
                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
            """, (lat, lng, sb, ms, pct, conf, confirmed, dc))
        connection.commit()

    connection.close()


seed_demo_bumps_if_empty()


# ============================================================
# DATA MODEL
# ============================================================

class TripData(BaseModel):

    lat: float
    lng: float
    speed: float
    timestamp: str

    # Optional identifier for future crowd validation
    trip_id: str = "default"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Speed Bump Detection API is running",
        "status": "online"
    }


# ============================================================
# RECEIVE GPS DATA
# ============================================================

@app.post("/trip-data")
def receive_trip_data(data: TripData):

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO trip_data
        (
            timestamp,
            lat,
            lng,
            speed,
            trip_id
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        data.timestamp,
        data.lat,
        data.lng,
        data.speed,
        data.trip_id
    ))

    connection.commit()
    connection.close()

    return {
        "message": "Trip data received successfully",
        "lat": data.lat,
        "lng": data.lng,
        "speed": data.speed,
        "trip_id": data.trip_id
    }


# ============================================================
# GET ALL TRIP DATA
# ============================================================

@app.get("/trip-data")
def get_trip_data():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            lat,
            lng,
            speed,
            trip_id
        FROM trip_data
        ORDER BY id
    """)

    rows = cursor.fetchall()

    connection.close()

    data = []

    for row in rows:

        data.append({
            "id": row[0],
            "timestamp": row[1],
            "lat": row[2],
            "lng": row[3],
            "speed": row[4],
            "trip_id": row[5]
        })

    return {
        "total_readings": len(data),
        "data": data
    }


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371000  # Earth radius in meters

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# CROWD VALIDATION
# ============================================================

def validate_crowd(lat, lng):

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            lat,
            lng
        FROM bumps
    """)

    rows = cursor.fetchall()

    connection.close()

    nearby_ids = []

    for row in rows:

        bump_id = row[0]
        bump_lat = row[1]
        bump_lng = row[2]

        distance = haversine_distance(
            lat,
            lng,
            bump_lat,
            bump_lng
        )

        if distance <= CROWD_RADIUS_METERS:
            nearby_ids.append(bump_id)

    return nearby_ids


# ============================================================
# DETECT SPEED BUMPS
# ============================================================

@app.post("/detect-bumps")
def detect_bumps():

    connection = sqlite3.connect(DATABASE)

    df = pd.read_sql_query("""
        SELECT
            timestamp,
            lat,
            lng,
            speed,
            trip_id
        FROM trip_data
        ORDER BY timestamp
    """, connection)

    connection.close()

    if len(df) < 7:

        return {
            "message": "Not enough GPS data",
            "bumps_detected": 0,
            "bumps": []
        }

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------
    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

    df = (
          df
        .sort_values("timestamp")
        .reset_index(drop=True)
     )

    # --------------------------------------------------------
    # FEATURE CALCULATIONS
    # --------------------------------------------------------

    df["time_diff"] = (
        df["timestamp"].diff().dt.total_seconds()
    )

    df["speed_change"] = df["speed"].diff()

    df["acceleration"] = (
        df["speed_change"]
        /
        df["time_diff"]
    )

    df["jerk"] = (
        df["acceleration"].diff()
        /
        df["time_diff"]
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    detected_bumps = []

    # --------------------------------------------------------
    # THRESHOLD PRE-FILTER
    # --------------------------------------------------------

    for i in range(1, len(df)):

        speed_before = float(
            df.loc[i - 1, "speed"]
        )

        current_speed = float(
            df.loc[i, "speed"]
        )

        # Ignore stationary readings
        if speed_before <= 0:
            continue

        # ----------------------------------------------------
        # SPEED DROP
        # ----------------------------------------------------

        speed_drop_pct = (
            (speed_before - current_speed)
            /
            speed_before
        ) * 100

        # 20% threshold
        if speed_drop_pct < DROP_THRESHOLD:
            continue

        # ----------------------------------------------------
        # RECOVERY CHECK
        # ----------------------------------------------------

        current_time = df.loc[i, "timestamp"]

        recovery_limit = (
            current_time
            +
            pd.Timedelta(
                seconds=RECOVERY_SECONDS
            )
        )

        future = df[
            (df["timestamp"] > current_time)
            &
            (df["timestamp"] <= recovery_limit)
        ]

        if len(future) == 0:
            continue

        recovery_speed = (
            speed_before * 0.90
        )

        recovered = future[
            future["speed"] >= recovery_speed
        ]

        if len(recovered) == 0:
            continue

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        minimum_speed = min(
            current_speed,
            float(future["speed"].min())
        )

        speed_drop = (
            speed_before
            -
            minimum_speed
        )

        time_to_minimum = max(
            1,
            (
                future["timestamp"].iloc[0]
                -
                current_time
            ).total_seconds()
        )

        deceleration = (
            speed_drop
            /
            time_to_minimum
        )

        recovery_time = (
            recovered["timestamp"].iloc[0]
            -
            current_time
        ).total_seconds()

        jerk = abs(
            float(df.loc[i, "jerk"])
        )

        features = pd.DataFrame([{

            "speed_before": speed_before,

            "minimum_speed": minimum_speed,

            "speed_drop": speed_drop,

            "speed_drop_pct": speed_drop_pct,

            "deceleration": deceleration,

            "recovery_time": recovery_time,

            "jerk": jerk

        }])[FEATURE_COLUMNS]

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        prediction = model.predict(
            features
        )[0]

        confidence = model.predict_proba(
            features
        )[0][1]

        # ----------------------------------------------------
        # ML CONFIRMATION
        # ----------------------------------------------------

        if (
            prediction == 1
            and
            confidence >= ML_CONFIDENCE
        ):

            detected_bumps.append({

                "timestamp":
                    current_time.isoformat(),

                "lat":
                    float(df.loc[i, "lat"]),

                "lng":
                    float(df.loc[i, "lng"]),

                "speed_before":
                    speed_before,

                "minimum_speed":
                    minimum_speed,

                "speed_drop_pct":
                    speed_drop_pct,

                "confidence":
                    float(confidence)

            })

    # ========================================================
    # SAVE DETECTED BUMPS
    # ========================================================

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # Clear detections from previous analysis
    cursor.execute("DELETE FROM bumps")

    for bump in detected_bumps:

        cursor.execute("""
            INSERT INTO bumps
            (
                timestamp,
                lat,
                lng,
                speed_before,
                minimum_speed,
                speed_drop_pct,
                confidence,
                confirmed,
                detection_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            bump["timestamp"],

            bump["lat"],

            bump["lng"],

            bump["speed_before"],

            bump["minimum_speed"],

            bump["speed_drop_pct"],

            bump["confidence"],

            0,

            1
        ))

    connection.commit()
    connection.close()

    # ========================================================
    # CROWD VALIDATION
    # ========================================================

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            lat,
            lng
        FROM bumps
    """)

    bump_rows = cursor.fetchall()

    for row in bump_rows:

        bump_id = row[0]
        bump_lat = row[1]
        bump_lng = row[2]

        # Count nearby detections
        count = 0

        for other in bump_rows:

            other_id = other[0]

            if other_id == bump_id:
                continue

            distance = haversine_distance(
                bump_lat,
                bump_lng,
                other[1],
                other[2]
            )

            if distance <= CROWD_RADIUS_METERS:
                count += 1

        detection_count = count + 1

        confirmed = (
            1
            if detection_count >= MIN_CROWD_DETECTIONS
            else 0
        )

        cursor.execute("""
            UPDATE bumps
            SET
                detection_count = ?,
                confirmed = ?
            WHERE id = ?
        """, (
            detection_count,
            confirmed,
            bump_id
        ))

    connection.commit()
    connection.close()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "message":
            "Speed bump detection completed",

        "bumps_detected":
            len(detected_bumps),

        "bumps":
            detected_bumps
    }


# ============================================================
# GET DETECTED BUMPS
# ============================================================

@app.get("/bumps")
def get_bumps():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            lat,
            lng,
            speed_before,
            minimum_speed,
            speed_drop_pct,
            confidence,
            confirmed,
            detection_count
        FROM bumps
        ORDER BY id
    """)

    rows = cursor.fetchall()

    connection.close()

    bumps = []

    for row in rows:

        bumps.append({

            "id":
                row[0],

            "timestamp":
                row[1],

            "lat":
                row[2],

            "lng":
                row[3],

            "speed_before":
                row[4],

            "minimum_speed":
                row[5],

            "speed_drop_pct":
                row[6],

            "confidence":
                row[7],

            "confirmed":
                bool(row[8]),

            "detection_count":
                row[9]

        })

    return {

        "total_bumps":
            len(bumps),

        "bumps":
            bumps

    }


# ============================================================
# GET ONLY CONFIRMED BUMPS
# ============================================================

@app.get("/confirmed-bumps")
def get_confirmed_bumps():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            lat,
            lng,
            confidence,
            detection_count
        FROM bumps
        WHERE confirmed = 1
        ORDER BY id
    """)

    rows = cursor.fetchall()

    connection.close()

    bumps = []

    for row in rows:

        bumps.append({

            "id":
                row[0],

            "timestamp":
                row[1],

            "lat":
                row[2],

            "lng":
                row[3],

            "confidence":
                row[4],

            "detection_count":
                row[5]

        })

    return {

        "total_confirmed_bumps":
            len(bumps),

        "bumps":
            bumps

    }


# ============================================================
# DELETE ALL TRIP DATA
# ============================================================

@app.delete("/trip-data")
def delete_trip_data():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM trip_data"
    )

    cursor.execute(
        "DELETE FROM bumps"
    )

    connection.commit()
    connection.close()

    return {
        "message":
            "Trip data and bump detections cleared successfully"
    }
