
import onnxruntime as rt
import numpy as np
import json
import re
from datetime import datetime, timedelta

# ── Load models and config ──────────────────────────
sess_temp     = rt.InferenceSession("student_temp.onnx")
sess_humidity = rt.InferenceSession("student_humidity.onnx")
sess_rain     = rt.InferenceSession("student_rain.onnx")

with open("adobe_edge_config.json") as f:
    config = json.load(f)

LOCATIONS    = {k: tuple(v) for k, v in config["locations"].items()}
feature_cols = config["feature_cols"]

conversation_memory = {"last_location": "Central Delhi (CP)"}

def extract_location(text):
    text_lower = text.lower()
    shortcuts = {
        "cp": "Central Delhi (CP)", "central delhi": "Central Delhi (CP)",
        "delhi": "Central Delhi (CP)", "dwarka": "Dwarka", "rohini": "Rohini",
        "noida": "Noida", "gurgaon": "Gurgaon", "gurugram": "Gurgaon",
        "faridabad": "Faridabad", "ghaziabad": "Ghaziabad",
    }
    for keyword, full_name in shortcuts.items():
        if keyword in text_lower:
            conversation_memory["last_location"] = full_name
            return full_name
    return conversation_memory["last_location"]

def extract_date(text):
    text_lower = text.lower()
    today = datetime.now().date()
    if "today"              in text_lower: return today.strftime("%Y-%m-%d")
    if "day before yesterday" in text_lower: return (today - timedelta(days=2)).strftime("%Y-%m-%d")
    if "yesterday"          in text_lower: return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    if "day after tomorrow" in text_lower: return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    if "tomorrow"           in text_lower: return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    months = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
        "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
        "aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,
        "nov":11,"november":11,"dec":12,"december":12
    }
    pattern = r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(months.keys()) + r")|(" + "|".join(months.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?"
    match = re.search(pattern, text_lower)
    if match:
        if match.group(1) and match.group(2):
            day, month = int(match.group(1)), months[match.group(2)]
        else:
            day, month = int(match.group(4)), months[match.group(3)]
        candidate = today.replace(month=month, day=day)
        if candidate > today:
            candidate = candidate.replace(year=today.year - 1)
        return candidate.strftime("%Y-%m-%d")
    return today.strftime("%Y-%m-%d")

def predict_edge(date_str, location):
    lat, lon = LOCATIONS[location]
    date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
    doy       = date_obj.timetuple().tm_yday

    # Build feature vector using seasonal averages
    avg = {col: 0.0 for col in feature_cols}
    avg["month"]       = float(date_obj.month)
    avg["day"]         = float(date_obj.day)
    avg["day_of_year"] = float(doy)
    avg["latitude"]    = float(lat)
    avg["longitude"]   = float(lon)

    # Seasonal estimates for Delhi NCR
    month = date_obj.month
    if month in [12,1,2]:   avg["temperature_2m_mean"] = 15.0; avg["relative_humidity_2m_mean"] = 60.0
    elif month in [3,4,5]:  avg["temperature_2m_mean"] = 30.0; avg["relative_humidity_2m_mean"] = 35.0
    elif month in [6,7,8,9]:avg["temperature_2m_mean"] = 32.0; avg["relative_humidity_2m_mean"] = 75.0
    else:                   avg["temperature_2m_mean"] = 25.0; avg["relative_humidity_2m_mean"] = 55.0

    avg["temp_mean_yesterday"]     = avg["temperature_2m_mean"]
    avg["humidity_yesterday"]      = avg["relative_humidity_2m_mean"]
    avg["temperature_2m_max"]      = avg["temperature_2m_mean"] + 5
    avg["temperature_2m_min"]      = avg["temperature_2m_mean"] - 5
    avg["apparent_temperature_mean"] = avg["temperature_2m_mean"] - 2
    avg["relative_humidity_2m_max"] = avg["relative_humidity_2m_mean"] + 10
    avg["relative_humidity_2m_min"] = avg["relative_humidity_2m_mean"] - 10

    X = np.array([[avg.get(col, 0.0) for col in feature_cols]], dtype=np.float32)

    temp     = float(sess_temp.run(None, {"float_input": X})[0][0])
    humidity = float(sess_humidity.run(None, {"float_input": X})[0][0])
    rain_p   = float(sess_rain.run(None, {"float_input": X})[1][0][1])

    return temp, humidity, rain_p

def warm_greeting(text):
    text = text.lower()
    if any(w in text for w in ["hi","hello","hey"]): return "Hey! 😊 I am Adobe Edge — your offline weather buddy!"
    if "good morning" in text: return "Good morning! 🌅 Want to check today\'s weather?"
    if "good night"   in text: return "Good night! 🌙 Stay weather-ready tomorrow!"
    return None

exit_phrases = ["quit","exit","bye","goodbye","thanks","thank you","done","stop","cheers"]

print("=" * 55)
print("   🌤️  ADOBE EDGE — Offline Weather Assistant")
print("=" * 55)
print("   ⚡ Fully offline — no internet needed")
print("   🔒 Privacy-first — all predictions local")
print("   📦 Powered by Student Model (68.59 MB)")
print("\n   Examples:")
print("   • What\'s the weather in Noida tomorrow?")
print("   • Will it rain in Gurgaon next week?")
print("   • How will Rohini be on 20th July?")
print("   Type bye/thanks to exit")
print("=" * 55)

import random
while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input: continue
        if any(p in user_input.lower() for p in exit_phrases):
            exits = ["Stay dry! ☂️","Bye! 🌤️","See you soon! 🌈","Take care! ☀️"]
            print(f"\n🤖 Adobe Edge: {random.choice(exits)}")
            break
        greeting = warm_greeting(user_input)
        if greeting:
            print(f"\n🤖 Adobe Edge: {greeting}")
            continue
        location   = extract_location(user_input)
        date_str   = extract_date(user_input)
        date_obj   = datetime.strptime(date_str, "%Y-%m-%d")
        friendly   = date_obj.strftime("%A, %d %B %Y")
        temp, humidity, rain_p = predict_edge(date_str, location)
        rain_str   = f"Yes ☔ ({rain_p*100:.0f}% chance)" if rain_p > 0.5 else f"No ☀️ ({rain_p*100:.0f}% chance)"
        print(f"\n🤖 Adobe Edge: Prediction for {location} on {friendly}:")
        print(f"   🌡️  Temperature : {temp:.1f}°C")
        print(f"   💧 Humidity    : {humidity:.1f}%")
        print(f"   🌧️  Rain        : {rain_str}")
        print(f"   ⚡ Source      : Student Model (Offline) 🔒")
        if rain_p > 0.7:   print(f"   💡 Tip: High rain chance — umbrella recommended! ☂️")
        elif temp > 38:     print(f"   💡 Tip: Very hot — stay hydrated! 💧")
        elif temp < 12:     print(f"   💡 Tip: Cold day — wear warm clothes! 🧣")
    except KeyboardInterrupt:
        print("\n🤖 Adobe Edge: Goodbye! 🌤️")
        break
