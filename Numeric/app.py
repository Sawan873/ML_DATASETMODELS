from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn.compose._column_transformer as column_transformer
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


MODEL_PATH = Path(__file__).with_name("student_model.pkl")

NUMERIC_FIELDS = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

CATEGORICAL_OPTIONS = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["A", "T"],
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
    "schoolsup": ["no", "yes"],
    "famsup": ["no", "yes"],
    "paid": ["no", "yes"],
    "activities": ["no", "yes"],
    "nursery": ["no", "yes"],
    "higher": ["no", "yes"],
    "internet": ["no", "yes"],
    "romantic": ["no", "yes"],
}

DEFAULT_VALUES = {
    "school": "GP",
    "sex": "F",
    "age": 18,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 2,
    "G1": 10,
    "G2": 10,
}

FIELD_HELP = {
    "age": "Student age",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "traveltime": "Home to school travel time",
    "studytime": "Weekly study time",
    "failures": "Past class failures",
    "famrel": "Family relationship quality",
    "freetime": "Free time after school",
    "goout": "Going out with friends",
    "Dalc": "Workday alcohol consumption",
    "Walc": "Weekend alcohol consumption",
    "health": "Current health status",
    "absences": "Number of absences",
    "G1": "First period grade",
    "G2": "Second period grade",
}


class _RemainderColsList(list):
    pass


column_transformer._RemainderColsList = _RemainderColsList
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


@st.cache_resource
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def collect_inputs(feature_names: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    left, right = st.columns(2)
    columns = [left, right]

    for index, feature in enumerate(feature_names):
        column = columns[index % 2]
        with column:
            if feature in CATEGORICAL_OPTIONS:
                default = DEFAULT_VALUES[feature]
                options = CATEGORICAL_OPTIONS[feature]
                values[feature] = st.selectbox(
                    feature,
                    options,
                    index=options.index(default),
                    key=feature,
                )
            else:
                values[feature] = st.number_input(
                    feature,
                    value=int(DEFAULT_VALUES[feature]),
                    step=1,
                    help=FIELD_HELP.get(feature),
                    key=feature,
                )

    return values


def main() -> None:
    st.set_page_config(
        page_title="Student Score Predictor",
        page_icon="📘",
        layout="wide",
    )

    st.title("Student Score Predictor")
    st.caption("Local Streamlit UI using `student_model.pkl` directly.")

    model = load_model()
    feature_names = list(getattr(model, "feature_names_in_", []))

    with st.form("prediction_form"):
        values = collect_inputs(feature_names)
        submitted = st.form_submit_button("Predict")

    if submitted:
        frame = pd.DataFrame([values], columns=feature_names)
        prediction = float(model.predict(frame)[0])
        st.success(f"Predicted final score: {prediction:.2f}")

    with st.expander("Model Features"):
        st.write(feature_names)


if __name__ == "__main__":
    main()
