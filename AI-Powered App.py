import pandas as pd
import numpy as np
import openai, os, joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data from CSV
df = pd.read_csv("smp_log.csv")
X = df[["sleep_hr", "water_glasses", "bench_kg"]].to_numpy()
y = (df["steps"] >= 10000).astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save trained model so you do not retrain every time
joblib.dump(clf, "smp_coach_model.pkl")

def get_coaching_message(sleep, water, bench, hit_goal, confidence):
    prompt = (f"Athlete logged: sleep {sleep}h, water {water} glasses, bench {bench}kg. "
              f"Model predicts: {'HIT GOAL' if hit_goal else 'MISS GOAL'} at {confidence:.0%} confidence. "
              "Give a direct 2-sentence coaching response. No filler. No hedging.")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an SMP performance coach. Data-driven. Direct. Two sentences maximum."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def analyze_day(sleep_hr, water_glasses, bench_kg, day_label=None):
    features = np.array([[sleep_hr, water_glasses, bench_kg]])
    prediction = clf.predict(features)[0]
    confidence = clf.predict_proba(features)[0][prediction]
    coaching = get_coaching_message(sleep_hr, water_glasses, bench_kg, prediction, confidence)
    return {
        "label":      day_label or "Day",
        "hit_goal":  bool(prediction),
        "confidence": confidence,
        "coaching":   coaching,
        "sleep_hr": sleep_hr,
        "water_glasses": water_glasses,
        "bench_kg": bench_kg,
    }
# --- Batch of incoming days ---
incoming_days = [
    ("Day 29", 8.0, 10, 92),
    ("Day 30", 5.5,  4, 70),
    ("Day 31", 7.2,  7, 84),
    ("Day 32", 8.5, 11, 95),
    ("Day 33", 6.1,  5, 76),
    ("Day 34", 7.8,  9, 88),
    ("Day 35", 5.9,  6, 73),
]

# Process all days, collect before printing
results = [analyze_day(sleep, water, bench, label)
           for label, sleep, water, bench in incoming_days]

# --- Individual reports ---
SEP = "=" * 62
print(SEP)
print("    SMP PERFORMANCE COACH  |  DAILY REPORTS")
print(SEP)

for r in results:
    outcome = "HIT GOAL" if r["hit_goal"] else "MISS GOAL"
    print(f"\n{r['label']}")
    print(f"  Outcome:  {outcome} ({r['confidence']:.0%} confidence)")
    print(f"  Inputs:   sleep={r['sleep_hr']}h  water={r['water_glasses']}gl  bench={r['bench_kg']}kg")
    print(f"  Coach:    {r['coaching']}")

# --- Summary ---
print()
print(SEP)
print("    SUMMARY")
print(SEP)

total  = len(results)
hits   = sum(1 for r in results if r["hit_goal"])
misses = total - hits
avg_conf = np.mean([r["confidence"] for r in results])

hit_days  = [r for r in results if r["hit_goal"]]
miss_days = [r for r in results if not r["hit_goal"]]

avg_sleep_hit  = np.mean([r["sleep_hr"]      for r in hit_days])  if hit_days  else 0
avg_water_hit  = np.mean([r["water_glasses"]  for r in hit_days])  if hit_days  else 0
avg_sleep_miss = np.mean([r["sleep_hr"]      for r in miss_days]) if miss_days else 0
avg_water_miss = np.mean([r["water_glasses"]  for r in miss_days]) if miss_days else 0

print(f"\n  Days analyzed:    {total}")
print(f"  Goals hit:        {hits} / {total}  ({hits/total:.0%})")
print(f"  Goals missed:     {misses} / {total}  ({misses/total:.0%})")
print(f"  Avg confidence:   {avg_conf:.0%}")
print()
print("  On HIT days:")
print(f"    avg sleep:  {avg_sleep_hit:.1f}h  |  avg water: {avg_water_hit:.1f} glasses")
print()
print("  On MISS days:")
print(f"    avg sleep:  {avg_sleep_miss:.1f}h  |  avg water: {avg_water_miss:.1f} glasses")
print()
sleep_diff = avg_sleep_hit - avg_sleep_miss
water_diff = avg_water_hit - avg_water_miss
print(f"  HIT days averaged {sleep_diff:+.1f}h more sleep and {water_diff:+.1f} more glasses of water")
print()
print(SEP)
print("    End of report")
print(SEP)