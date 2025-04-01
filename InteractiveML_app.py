import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_curve, auc, 
                             r2_score, mean_squared_error)
from sklearn.preprocessing import LabelEncoder
import pickle

def initialize_state():
    """Ensure necessary session state keys are initialized."""
    defaults = {
        "numeric_feats": [],
        "categorical_feats": [],
        "target": "Select target...",
        "prev_dataset": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def load_sample_dataset(ds_name):
    """Load one of Seaborn's built-in datasets."""
    return sns.load_dataset(ds_name)

def load_csv_dataset(uploaded_file):
    """Read CSV data."""
    return pd.read_csv(uploaded_file)

def main():
    st.title("ML Model Trainer")
    st.markdown("""
    Train a machine learning model (regression or classification) using either built-in datasets or your own CSV file.
    Configure feature selection, model parameters, and view performance metrics and visualizations.
    """)

    st.header("1. Dataset Selection")
    source_option = st.radio("Data Source:", ["Built-in Dataset", "Upload CSV"])
    
    data = None
    ds_id = None

    if source_option == "Built-in Dataset":
        available_datasets = ["iris", "tips", "titanic", "penguins", "diamonds"]
        chosen_ds = st.selectbox("Choose a dataset:", available_datasets)
        if chosen_ds:
            @st.cache_data
            def get_sample(ds):
                return load_sample_dataset(ds)
            data = get_sample(chosen_ds)
            ds_id = f"sample-{chosen_ds}"
    else:
        csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if csv_file:
            @st.cache_data
            def get_csv(file):
                return load_csv_dataset(file)
            data = get_csv(csv_file)
            ds_id = f"uploaded-{csv_file.name}"

    initialize_state()  # Ensure session state keys exist

    # If a new dataset is loaded, reset selections
    if ds_id and st.session_state["prev_dataset"] != ds_id:
        st.session_state["numeric_feats"] = []
        st.session_state["categorical_feats"] = []
        st.session_state["target"] = "Select target..."
        st.session_state["prev_dataset"] = ds_id

    if data is not None:
        st.write(f"**Data Dimensions:** {data.shape[0]} rows, {data.shape[1]} columns")
        st.dataframe(data.head())

        # Identify numeric and categorical columns
        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_columns = [col for col in data.columns if col not in numeric_columns]

        with st.form("config_form"):
            st.subheader("2. Feature & Target Selection")
            colA, colB = st.columns(2)
            with colA:
                num_feats = st.multiselect("Numeric Features", 
                                           options=numeric_columns, 
                                           default=st.session_state["numeric_feats"], 
                                           key="numeric_feats")
            with colB:
                cat_feats = st.multiselect("Categorical Features", 
                                           options=categorical_columns, 
                                           default=st.session_state["categorical_feats"], 
                                           key="categorical_feats")
            target_choices = ["Select target..."] + list(data.columns)
            target_choice = st.selectbox("Target Variable", 
                                         options=target_choices, 
                                         index=0, 
                                         key="target")
            
            st.subheader("3. Model Configuration")
            model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest (Classification)"])
            test_frac = st.slider("Test Set Fraction", 0.1, 0.5, 0.2)
            
            # Parameters only needed for Random Forest
            rf_estimators = None
            rf_depth = None
            if model_type == "Random Forest (Classification)":
                col1, col2 = st.columns(2)
                with col1:
                    rf_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)
                with col2:
                    rf_depth = st.slider("Max Depth", 1, 20, 5, step=1)
            
            submit = st.form_submit_button("Train Model")

        if submit:
            # Retrieve user selections from session state
            selected_nums = st.session_state["numeric_feats"]
            selected_cats = st.session_state["categorical_feats"]
            chosen_target = st.session_state["target"]

            if chosen_target == "Select target..." or not chosen_target:
                st.error("Please choose a target variable.")
            elif len(selected_nums) + len(selected_cats) == 0:
                st.error("Select at least one feature (numeric or categorical).")
            else:
                features = [col for col in selected_nums + selected_cats if col != chosen_target]
                subdata = data[features + [chosen_target]].dropna()
                if subdata.empty:
                    st.error("No data remains after removing missing values.")
                else:
                    X = subdata[features]
                    y = subdata[chosen_target]

                    if model_type == "Linear Regression":
                        if not pd.api.types.is_numeric_dtype(y):
                            st.error("For regression, the target must be numeric.")
                            return
                        chosen_model = "regression"
                    else:
                        chosen_model = "classification"
                    
                    # One-hot encode categorical predictors
                    X_enc = pd.get_dummies(X, drop_first=True)
                    feat_list = X_enc.columns.tolist()

                    if chosen_model == "classification":
                        le = LabelEncoder()
                        y_enc = le.fit_transform(y)
                        class_labels = le.classes_
                    else:
                        y_enc = y.values
                        class_labels = None

                    strat_param = y_enc if chosen_model == "classification" else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_enc, y_enc, test_size=test_frac, stratify=strat_param, random_state=42
                    )

                    @st.cache_resource
                    def train_model(X_tr, y_tr, model_choice, n_est=None, max_d=None):
                        if model_choice == "Linear Regression":
                            mdl = LinearRegression()
                        else:
                            mdl = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                        mdl.fit(X_tr, y_tr)
                        return mdl

                    model_obj = train_model(X_train, y_train, model_type, n_est=rf_estimators, max_d=rf_depth)
                    y_pred = model_obj.predict(X_test)

                    if chosen_model == "classification":
                        acc = accuracy_score(y_test, y_pred)
                        st.write(f"**Accuracy:** {acc*100:.2f}%")

                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
                                    xticklabels=(class_labels if class_labels is not None else np.unique(y_test)),
                                    yticklabels=(class_labels if class_labels is not None else np.unique(y_test)),
                                    ax=ax_cm)
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        ax_cm.set_title("Confusion Matrix")
                        st.pyplot(fig_cm)

                        # If binary classification, plot ROC curve
                        if len(np.unique(y_test)) == 2:
                            y_prob = model_obj.predict_proba(X_test)[:, 1] if hasattr(model_obj, "predict_proba") else None
                            if y_prob is not None:
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                                ax_roc.plot([0, 1], [0, 1], "k--")
                                ax_roc.set_xlabel("False Positive Rate")
                                ax_roc.set_ylabel("True Positive Rate")
                                ax_roc.set_title("ROC Curve")
                                ax_roc.legend(loc="lower right")
                                st.pyplot(fig_roc)
                                st.write(f"**AUC:** {roc_auc:.3f}")

                        # Feature importance for classification
                        imp_vals = model_obj.feature_importances_
                        imp_series = pd.Series(imp_vals, index=feat_list).sort_values(ascending=True)
                        if imp_series.size > 20:
                            imp_series = imp_series.iloc[-20:]
                            st.caption("Top 20 features by importance.")
                        fig_imp, ax_imp = plt.subplots()
                        ax_imp.barh(imp_series.index, imp_series.values)
                        ax_imp.set_title("Feature Importance")
                        ax_imp.set_xlabel("Importance")
                        st.pyplot(fig_imp)

                    else:
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        st.write(f"**R²:** {r2:.3f}")
                        st.write(f"**RMSE:** {rmse:.3f}")

                        residuals = y_test - y_pred
                        fig_res, ax_res = plt.subplots()
                        sns.histplot(residuals, kde=True, ax=ax_res, color="cornflowerblue")
                        ax_res.set_title("Residual Distribution")
                        ax_res.set_xlabel("Residuals")
                        st.pyplot(fig_res)

                        if model_type == "Linear Regression":
                            coeffs = model_obj.coef_
                            coef_series = pd.Series(np.abs(coeffs), index=feat_list).sort_values(ascending=True)
                            if coef_series.size > 20:
                                coef_series = coef_series.iloc[-20:]
                                st.caption("Top 20 features by absolute coefficient.")
                            fig_coef, ax_coef = plt.subplots()
                            ax_coef.barh(coef_series.index, coef_series.values)
                            ax_coef.set_title("Coefficient Importance")
                            ax_coef.set_xlabel("Absolute Coefficient")
                            st.pyplot(fig_coef)

                    # Provide download for trained model
                    model_data = pickle.dumps(model_obj)
                    st.download_button("Download Trained Model", data=model_data,
                                       file_name="trained_model.pkl",
                                       mime="application/octet-stream")

if __name__ == "__main__":
    main()
