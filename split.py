import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("clinic_appointments.csv")
y = df["no_show"]

training_set, test_set = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

training_set.to_csv("training_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

train_no_shows = (training_set["no_show"] == "no-show").sum()
test_no_shows = (test_set["no_show"] == "no-show").sum()

print(f"total rows: {len(df)}")
print(f"training rows: {len(training_set)}")
print(f"test rows: {len(test_set)}")
print(
    f"training no-shows: {train_no_shows} "
    f"({100 * train_no_shows / len(training_set):.1f}%)"
)
print(
    f"test no-shows: {test_no_shows} "
    f"({100 * test_no_shows / len(test_set):.1f}%)"
)
