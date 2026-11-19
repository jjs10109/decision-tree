import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv("training_set.csv")

features = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]

X = df[features]
y = df["no_show"].map({"checked-in": 0, "no-show": 1})

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

print(export_text(tree, feature_names=features))

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    tree,
    filled=True,
    feature_names=features,
    class_names=["checked-in", "no-show"],
    fontsize=9,
    ax=ax,
)
fig.tight_layout()
fig.savefig("tree_v1.png")

print(f"tree depth: {tree.get_depth()}")
print(f"number of leaves: {tree.get_n_leaves()}")

used = []
for index in tree.tree_.feature:
    if index >= 0 and features[index] not in used:
        used.append(features[index])

print("features used in splits:")
for name in used:
    print(f"- {name}")


def report(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision for no-show: {precision:.3f}")
    print(f"recall for no-show: {recall:.3f}")
    return accuracy, tn, fp, fn, tp


test = pd.read_csv("test_set.csv")
X_test = test[features]
y_test = test["no_show"].map({"checked-in": 0, "no-show": 1})

v1_train_pred = tree.predict(X)
v1_test_pred = tree.predict(X_test)
baseline_test_pred = [0] * len(y_test)

print()
print("BASELINE TEST RESULTS")
_, baseline_tn, baseline_fp, baseline_fn, baseline_tp = report(
    y_test, baseline_test_pred
)

print()
print("V1 TRAINING RESULTS")
v1_train_accuracy, _, _, _, _ = report(y, v1_train_pred)

print()
print("V1 TEST RESULTS")
v1_test_accuracy, v1_tn, v1_fp, v1_fn, v1_tp = report(y_test, v1_test_pred)

print()
print(f"V1 training accuracy: {v1_train_accuracy:.3f}")
print(f"V1 test accuracy: {v1_test_accuracy:.3f}")


def should_overbook(applicant):
    row = pd.DataFrame([applicant], columns=features)
    predicted_class = int(tree.predict(row)[0])
    leaf_id = int(tree.apply(row)[0])
    if predicted_class == 1:
        decision = "overbook"
    else:
        decision = "hold"
    return predicted_class, decision, leaf_id


appointments = {
    "D": {
        "lead_time_days": 21,
        "prior_no_shows": 1,
        "age": 27,
        "distance_miles": 9.5,
        "sms_reminder": 0,
        "new_patient": 1,
        "appointment_hour": 15,
    },
    "E": {
        "lead_time_days": 3,
        "prior_no_shows": 0,
        "age": 61,
        "distance_miles": 2.0,
        "sms_reminder": 1,
        "new_patient": 0,
        "appointment_hour": 9,
    },
}

class_names = {0: "checked-in", 1: "no-show"}

for label, appointment in appointments.items():
    predicted_class, decision, leaf_id = should_overbook(appointment)
    print()
    print(f"APPOINTMENT {label}")
    print(f"predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"leaf node ID: {leaf_id}")
    print(f"decision: {decision}")


def value(tn, fp, fn, tp):
    return -60 * fp - 120 * fn


n_test = len(y_test)
baseline_total = value(baseline_tn, baseline_fp, baseline_fn, baseline_tp)
v1_total = value(v1_tn, v1_fp, v1_fn, v1_tp)
improvement = v1_total - baseline_total

print()
print("BASELINE TEST VALUE")
print(f"total value: ${baseline_total:.2f}")
print(f"value per appointment: ${baseline_total / n_test:.2f}")

print()
print("V1 TEST VALUE")
print(f"total value: ${v1_total:.2f}")
print(f"value per appointment: ${v1_total / n_test:.2f}")

print()
print("V1 IMPROVEMENT OVER BASELINE")
print(f"total improvement: ${improvement:.2f}")
print(f"improvement per appointment: ${improvement / n_test:.2f}")

annual_volume = 10000
improvement_per_appointment = improvement / n_test
annual_improvement = improvement_per_appointment * annual_volume

print()
print("ESTIMATED ANNUAL IMPROVEMENT")
print(f"Assumed annual appointment volume: {annual_volume:,}")
print(f"V1 improvement per appointment: ${improvement_per_appointment:.2f}")
print(f"Estimated annual improvement: ${annual_improvement:,.2f}")

baseline_per_appointment = baseline_total / n_test
v1_per_appointment = v1_total / n_test
chart_labels = ["Baseline", "V1"]
chart_values = [baseline_per_appointment, v1_per_appointment]

value_fig, value_ax = plt.subplots(figsize=(7, 5))
bars = value_ax.bar(
    chart_labels,
    chart_values,
    color=["#8A8F98", "#1F4E79"],
    width=0.55,
)
value_ax.set_ylabel("Value per Appointment ($)")
value_ax.set_title("Baseline vs V1 Value per Appointment")
value_ax.axhline(0, color="black", linewidth=0.8)
value_ax.spines["top"].set_visible(False)
value_ax.spines["right"].set_visible(False)

lowest = min(chart_values)
value_ax.set_ylim(lowest * 1.2, abs(lowest) * 0.28)

for bar, amount in zip(bars, chart_values):
    if amount < 0:
        label = f"-${abs(amount):.2f}"
    else:
        label = f"${amount:.2f}"
    value_ax.text(
        bar.get_x() + bar.get_width() / 2,
        abs(lowest) * 0.04,
        label,
        ha="center",
        va="bottom",
        fontsize=12,
    )

value_fig.tight_layout()
value_fig.savefig("value_per_appointment.png", dpi=200)
plt.close(value_fig)
