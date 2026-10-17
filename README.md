# RoadSense Backend — rebuilt

Your original `main.py` and a working `speed_bump_model.pkl` (since the
real one trained on your actual collected data wasn't saved anywhere —
this one is trained on synthetic data instead, same as what you used
for your demo).

## Run it

Open PowerShell **in this folder** (click the address bar in File
Explorer while inside this folder, type `powershell`, press Enter),
then:

```
pip install -r requirements.txt
```

Then (optional but recommended for your demo — puts 3 fake bumps near
Bengaluru on the map immediately):

```
python seed_fake_bumps.py
```

Then start the server:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Leave this window open and running while you use the Flutter app.

## Files

- `main.py` — your backend, unchanged
- `speed_bump_model.pkl` — synthetic-data-trained model (swap this file
  out if you ever get the real one from a teammate — same filename,
  same folder, nothing else to change)
- `train_fake_model.py` — the script that generated the .pkl, in case
  you want to regenerate it or tweak it
- `seed_fake_bumps.py` — drops 3 fake bump markers near Bengaluru
  straight into the database for demo purposes
- `speedbump.db` — created automatically the first time you run the
  server (and by the seed script)
