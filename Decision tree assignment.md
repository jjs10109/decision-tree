# Clinic no-shows: should the clinic overbook this slot?

## The question

A medical clinic loses money every time a patient does not show up: the slot sits empty. The clinic is considering **overbooking** slots where a no-show is likely, the way airlines oversell seats. **Can the clinic predict which appointments will be no-shows well enough to decide which slots to overbook, and how much is that worth compared to its current policy?**

## The model

**Decision tree (classification).** The outcome is categorical as opposed to numeric: did the patient miss the appointment? Use python to fit a decision tree.

## Context

If the clinic overbooks a slot and the patient does not show, the second patient fills it. If it overbooks and both patients show, the day runs over and the clinic pays overtime and refunds a copay. If it holds a slot and the patient does not show, the slot sits empty.

| Prediction | What happened | Value per appointment |
|---|---|---|
| show (hold) | patient came | $0 |
| no-show (overbook) | patient came: both came, overtime and refund | −$60 |
| show (hold) | patient did not come: slot sat empty | −$120 |
| no-show (overbook) | patient did not come: second patient filled the slot | $0 |

The **baseline** is the most common class: predict that every patient shows, which is the clinic's current policy of holding every slot.

## The data

**`clinic_appointments.csv`**: 400 past appointments, one row per appointment.

| Column | Meaning |
|---|---|
| `lead_time_days` | days between booking and the appointment |
| `prior_no_shows` | how many earlier appointments this patient missed |
| `age` | patient age |
| `distance_miles` | distance from the patient's home to the clinic |
| `sms_reminder` | 1 if a text reminder was sent, 0 if not |
| `new_patient` | 1 if this is the patient's first visit |
| `appointment_hour` | hour of the day, 8 to 16 |
| `no_show` | "no-show" if the patient did not show up, "checked-in" if they did.

Data checks: 400 rows, 8 columns, 91 no-shows (22.8%).

## The task

1. **Split the data.** Randomly set aside 30% of the 400 appointments as a test set, keeping the no-show rate about the same in both parts. Fit everything on the remaining 70% (the training set). The test appointments are only for evaluating fit.
2. **Fit the model.** Fit one classification tree on the training appointments: v1, the best model you think you can build from this data. Choose the features you think matter and a depth you think is sensible, and go straight to them; you do not need to step through versions with different feature combinations. Report which features the tree splits on, starting from the root, and what that means for the clinic.
3. **Evaluate fit on the test appointments.** For the baseline (predict show for everyone) and for v1, report the confusion matrix (TN, FP, FN, TP), accuracy, precision, and recall on no-shows, all on the test appointments. Also report v1's training accuracy next to its test accuracy: if training is much higher, the tree memorized the training rows. The comparison is **baseline vs v1 on appointments the model never saw**.
4. **Make the decision.** The rule: overbook the slot if v1 predicts a no-show, hold it if v1 predicts a show. Write it as a function that takes one appointment and returns overbook or hold. Apply it to two upcoming appointments and report the leaf each one lands in and the decision for each:
   - D: lead_time_days 21, prior_no_shows 1, age 27, distance_miles 9.5, sms_reminder 0, new_patient 1, appointment_hour 15
   - E: lead_time_days 3, prior_no_shows 0, age 61, distance_miles 2.0, sms_reminder 1, new_patient 0, appointment_hour 9
5. **Quantify it.** Using the cost table and the test confusion matrices, compute the total and per-appointment value of the baseline and of v1 on the test appointments.
6. **Present it.** Make two slides for the clinic manager, not for a statistics audience.

## Deliverables
Note: you may use AI throughout and it is expected that you do so for coding, however, do not use AI to generate your slides. The value of this assignment is interpreting the technical output and synthesizing for business decision-makers. It is better to have succinct black text on white background that you wrote yourself rather than volumes of AI-generated nonsense for your "boss" to have to parse. (Asking AI to write python code to create graphs/charts for your slides is expected, copying-pasting an a graph made by ChatGPT/Claude/Gemini is not.)

One submission: **one PDF.**

1. **Title slide:** your name and the link to your GitHub repo containing the code and analysis for the project. The code reproduces every number on your slides.
2. **Two slides covering:**
   - Data: source, rows, the 70/30 split.
   - Model specification: outcome, features, model family, depth. The tree picture if it fits.
   - Fit baseline vs v1 comparisons, on the test appointments: confusion matrix, accuracy, precision, recall; v1's training vs test accuracy.
   - Business decision and value quantification:
   -- Headline in one sentence- what overbooking changes for the clinic and what it is worth
   -- The decision rule and the decision for D and E
   -- Compare value quantification of baseline vs v1: value per appointment on the test appointments, and what that means per year at the clinic's volume (state the volume you assume)
