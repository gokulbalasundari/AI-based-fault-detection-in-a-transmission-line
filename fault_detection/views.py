from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from django.conf import settings
import os

# Create your views here.
csv_path = os.path.join(settings.BASE_DIR, 'static', 'dataset', 'fault_full_data.csv')

def inputform(request):

    if request.method == 'POST':
        try:
            Ia = float(request.POST.get('Ia'))
            Ib = float(request.POST.get('Ib'))
            Ic = float(request.POST.get('Ic'))
            Va = float(request.POST.get('Va'))
            Vb = float(request.POST.get('Vb'))
            Vc = float(request.POST.get('Vc'))

            # Load the dataset and train the model
            df = pd.read_csv(csv_path)

            print("Dataset Loaded Successfully!")

            X = df.iloc[:, 0:6]
            y = df.iloc[:, 6:10]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            multi_output_classifier = MultiOutputClassifier(rf_classifier, n_jobs=-1)
            multi_output_classifier.fit(X_train, y_train)
            y_pred = multi_output_classifier.predict(X_test)

            # Make prediction for the input values
            single_sample = [Ia, Ib, Ic, Va, Vb, Vc]
            predicted_output = multi_output_classifier.predict([single_sample])

            # Define conditions for output
            conditions = [
                [0, 0, 0, 0], "No Fault",
                [1, 0, 0, 0], "Ground Fault",
                [0, 0, 0, 1], "Fault in Line A",
                [0, 0, 1, 0], "Fault in Line B",
                [0, 1, 0, 0], "Fault in Line C",
                [1, 0, 0, 1], "LG fault (Between Phase A and Ground)",
                [1, 0, 1, 0], "LG fault (Between Phase B and Ground)",
                [1, 1, 0, 0], "LG fault (Between Phase C and Ground)",
                [0, 0, 1, 1], "LL fault (Between Phase B and Phase A)",
                [0, 1, 1, 0], "LL fault (Between Phase C and Phase B)",
                [0, 1, 0, 1], "LL fault (Between Phase C and Phase A)",
                [1, 0, 1, 1], "LLG Fault (Between Phases A,B and Ground)",
                [1, 1, 0, 1], "LLG Fault (Between Phases A,C and Ground)",
                [1, 1, 1, 0], "LLG Fault (Between Phases C,B and Ground)",
                [0, 1, 1, 1], "LLL Fault(Between all three phases)",
                [1, 1, 1, 1], "LLLG fault( Three phase symmetrical fault)"
            ]

            output_text = "Output does not match any Fault."
            for condition, text in zip(conditions[::2], conditions[1::2]):
                if predicted_output.tolist()[0] == condition:
                    output_text = text
                    break

            return render(request, 'result.html', {'output_text': output_text})

        except FileNotFoundError:
            return render(request, 'input_form.html', {'error': "CSV file not found!"})

        except Exception as e:
            return render(request, 'input_form.html', {'error': f"An error occurred: {e}"})

    return render(request, 'input_form.html')



