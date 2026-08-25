import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.inspection import PartialDependenceDisplay
from mapie.classification import SplitConformalClassifier

##############################################################
# PAGE CONFIGURATION
##############################################################

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

##############################################################
# LOAD MODEL AND PREPROCESSING OBJECTS
##############################################################

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl") 


#########################################################
# LOAD SHAP EXPLAINER
#########################################################

@st.cache_resource
def load_feature_columns():
    return joblib.load("models/feature_columns.pkl")


model = load_model()
feature_columns = load_feature_columns()

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

@st.cache_resource
def load_calibration_data():

    X_cal = joblib.load("models/X_calib.pkl")
    y_cal = joblib.load("models/y_calib.pkl")

    return X_cal, y_cal

X_calibration, y_calibration = load_calibration_data()

@st.cache_resource
def load_conformal(_model, X_cal, y_cal):

    conformal = SplitConformalClassifier(
        estimator=model,
        confidence_level=0.95,
        prefit=True
    )

    conformal.conformalize(
        X_cal,
        y_cal
    )

    return conformal

conformal_model = load_conformal(
    model,
    X_calibration,
    y_calibration
)

##############################################################
# STORE IN SESSION STATE
##############################################################

st.session_state["model"] = model
st.session_state["feature_columns"] = feature_columns

##############################################################
# SIDEBAR
##############################################################

# Display logo if available
logo_path = "assets/logo.png"

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.title("Customer Churn Prediction")

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select Prediction Type",
    (
        "Single Prediction",
        "Batch Prediction"
    )
)

##############################################################
# MAIN TITLE
##############################################################

st.title("📊 Customer Churn Prediction")

st.markdown(
"""
Predict whether a customer is likely to churn using intelligent models.
"""
)
st.markdown("---")

##############################################################
# EXPECTED FEATURES
##############################################################

with st.expander("View Expected Input Features"):

    feature_df = pd.DataFrame(
        feature_columns,
        columns=["Feature"]
    )

    st.dataframe(
        feature_df,
        use_container_width=True
    )


##############################################################
# SINGLE CUSTOMER PREDICTION
##############################################################

if page == "Single Prediction":

    st.header("Single Customer Prediction")

    st.write(
        "Enter the customer's details below and click **Predict**."
    )

    st.markdown("---")

    # -------------------------
    # INPUT FORM
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:

        CreditScore = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        Geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        Gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        Age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        Tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=20,
            value=5
        )

    with col2:

        Balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.00
        )

        NumOfProducts = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        HasCrCard = st.selectbox(
         "Has Credit Card",
         ["Yes", "No"]
        )   

        HasCrCard = 1 if HasCrCard == "Yes" else 0


        IsActiveMember = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
        )

        IsActiveMember = 1 if IsActiveMember == "Yes" else 0

        EstimatedSalary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=100000.00
        )

    st.markdown("---")

    ##########################################################
    # PREDICTION
    ##########################################################

    if st.button("Predict Customer Churn"):

        # --------------------------------------------
        # Create DataFrame
        # --------------------------------------------

        input_df = pd.DataFrame({

            "CreditScore":[CreditScore],

            "Geography":[Geography],

            "Gender":[Gender],

            "Age":[Age],

            "Tenure":[Tenure],

            "Balance":[Balance],

            "NumOfProducts":[NumOfProducts],

            "HasCrCard":[HasCrCard],

            "IsActiveMember":[IsActiveMember],

            "EstimatedSalary":[EstimatedSalary]

        })

        # --------------------------------------------
        # One-Hot Encode
        # --------------------------------------------

        input_df = pd.get_dummies(
            input_df,
            columns=["Geography","Gender"],
            drop_first=True
        )

        # --------------------------------------------
        # Match Training Columns
        # --------------------------------------------

        for col in feature_columns:

            if col not in input_df.columns:

                input_df[col] = 0

        input_df = input_df[feature_columns]

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

#########################################################
# LOCAL SHAP EXPLANATION
#########################################################


        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠ Customer is likely to CHURN")

        else:

            st.success("✅ Customer is NOT likely to churn")

        st.metric(
            "Probability of Churn",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

        st.write("Prediction Confidence")

        if probability >= 0.80:

            st.error("Very High Risk")

        elif probability >= 0.60:

            st.warning("High Risk")

        elif probability >= 0.40:

            st.info("Moderate Risk")

        else:

            st.success("Low Risk")

        ##################################################
        # Display Processed Data
        ##################################################

        with st.expander("Processed Input Sent to Model"):

            st.dataframe(input_df)
        ######################################
        ### MAPIE 
        st.markdown("---")
        st.subheader("Conformal Prediction")

        ############################################################
        # CONFORMAL PREDICTION
        ############################################################

        y_conformal_pred, y_prediction_sets = conformal_model.predict_set(
            input_df
        )

        # Get the prediction set for the first customer
        prediction_set = y_prediction_sets[0, :, 0]

        ############################################################
        # DISPLAY CONFORMAL PREDICTION
        ############################################################

        no_churn_in_set = bool(prediction_set[0])
        churn_in_set = bool(prediction_set[1])

        if no_churn_in_set and churn_in_set:

            st.warning(
                "Prediction Set: {No Churn, Churn}"
            )

            st.write(
                "The conformal prediction indicates uncertainty "
                "between the two classes."
            )

        elif churn_in_set:

            st.error(
                "Prediction Set: {Churn}"
            )

            st.write(
                "The conformal prediction set contains only "
                "the Churn class."
            )

        elif no_churn_in_set:

            st.success(
                "Prediction Set: {No Churn}"
            )

            st.write(
                "The conformal prediction set contains only "
                "the No Churn class."
            )

        else:

            st.warning(
                "Prediction Set: Empty"
            )

        ###############
        ##############################################################
        # CUSTOMER-FRIENDLY LOCAL EXPLANATION
        ##############################################################

        st.markdown("---")

        st.subheader("Why was this prediction made?")

        if prediction == 1:

            st.error(
                f"""
                **This customer is considered at risk of leaving.**

                The model estimates a **{probability:.1%} probability of churn**.

                The explanation below shows the customer characteristics
                that had the greatest influence on this prediction.
                """
            )

        else:

            st.success(
                f"""
                **This customer is currently considered unlikely to leave.**

                The model estimates a **{probability:.1%} probability of churn**.

                The explanation below shows which customer characteristics
                influenced the prediction.
                """
            )


        st.write(
            """
            The factors below do not necessarily cause a customer to leave.
            Instead, they show which customer characteristics had the greatest
            influence on the model's decision for this particular customer.
            """
        )


        ##############################################################
        # COMPUTE LOCAL SHAP EXPLANATION
        ##############################################################

        explanation = explainer(input_df)


        ##############################################################
        # GET LOCAL EXPLANATION
        ##############################################################

        if len(explanation.values.shape) == 3:

            local_exp = shap.Explanation(

                values=explanation.values[0, :, 1],

                base_values=explanation.base_values[0, 1],

                data=input_df.iloc[0].values,

                feature_names=input_df.columns.tolist()

            )

        else:

            local_exp = shap.Explanation(

                values=explanation.values[0],

                base_values=explanation.base_values[0],

                data=input_df.iloc[0].values,

                feature_names=input_df.columns.tolist()

            )


        ##############################################################
        # FEATURE CONTRIBUTION DATA
        ##############################################################

        importance_df = pd.DataFrame({

            "Feature": local_exp.feature_names,

            "Feature Value": local_exp.data,

            "SHAP Value": local_exp.values

        })


        importance_df["Absolute SHAP"] = (
            importance_df["SHAP Value"].abs()
        )


        importance_df = (
            importance_df
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
        )


        ##############################################################
        # TOP FACTORS
        ##############################################################

        top_features = importance_df.head(5)


        st.markdown("---")

        st.subheader(
            "Key Factors Influencing This Customer"
        )

        st.write(
            """
            These are the five customer characteristics that had the
            strongest influence on the model's prediction.
            """
        )


        for _, row in top_features.iterrows():

            feature = row["Feature"]

            shap_value = row["SHAP Value"]

            if shap_value > 0:

                st.warning(
                    f"**{feature}** — increases the customer's "
                    f"estimated churn risk."
                )

            elif shap_value < 0:

                st.success(
                    f"**{feature}** — reduces the customer's "
                    f"estimated churn risk."
                )

            else:

                st.info(
                    f"**{feature}** — had little influence on this prediction."
                )


        ##############################################################
        # SHAP WATERFALL PLOT
        ##############################################################

        st.markdown("---")

        st.subheader(
            "Detailed Prediction Explanation"
        )

        st.write(
            """
            The chart below provides a more detailed view of how the
            individual customer characteristics contributed to the prediction.
            """
        )


        fig = plt.figure(figsize=(8, 5))

        shap.plots.waterfall(
            local_exp,
            max_display=10,
            show=False
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        ##############################################################
        # EXPLANATION UNDER SHAP PLOT
        ##############################################################

        st.info(
            """
            **How to read this chart**

            Each feature represents information about this customer.

            • Features pushing the prediction toward **Churn** increase
            the customer's estimated risk of leaving. These are the red bars.

            • Features pushing the prediction toward **No Churn** reduce
            the estimated risk. These are the blue bars.

            • Larger contributions indicate a stronger influence on the
            model's prediction.

            **Important:** These values describe how the model arrived at
            its prediction. They should not be interpreted as proof that
            a particular feature directly causes customer churn.
            """
        )


        ##############################################################
        # TECHNICAL FEATURE CONTRIBUTIONS
        ##############################################################

        with st.expander(
            "View Technical Feature Contributions"
        ):

            technical_df = importance_df.drop(
                columns=["Absolute SHAP"]
            )

            st.dataframe(
                technical_df,
                use_container_width=True,
                hide_index=True
            )


        ##############################################################
        # POSITIVE AND NEGATIVE FACTORS
        ##############################################################

        with st.expander(
            "View Factors Increasing and Reducing Churn"
        ):

            positive_features = (

                importance_df[
                    importance_df["SHAP Value"] > 0
                ]

                .sort_values(
                    "SHAP Value",
                    ascending=False
                )

                .head(5)

                .drop(
                    columns=["Absolute SHAP"]
                )

            )


            negative_features = (

                importance_df[
                    importance_df["SHAP Value"] < 0
                ]

                .sort_values(
                    "SHAP Value"
                )

                .head(5)

                .drop(
                    columns=["Absolute SHAP"]
                )

            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "**Factors Increasing Churn Risk**"
                )

                st.dataframe(
                    positive_features,
                    use_container_width=True,
                    hide_index=True
                )


            with col2:

                st.write(
                    "**Factors Reducing Churn Risk**"
                )

                st.dataframe(
                    negative_features,
                    use_container_width=True,
                    hide_index=True
                )
      
##############################################################
# BATCH PREDICTION
##############################################################

elif page == "Batch Prediction":

    st.header("Batch Customer Churn Prediction")

    st.write(
        """
        Upload a CSV file containing customer information.
        The application will predict whether each customer
        is likely to churn.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        ##########################################################
        # READ DATA
        ##########################################################

        try:

            batch_df = pd.read_csv(uploaded_file)

        except Exception as e:

            st.error(f"Error reading file: {e}")
            st.stop()

        st.subheader("Uploaded Dataset")

        st.dataframe(batch_df.head())

        st.write(f"Number of records: **{len(batch_df)}**")

        ##########################################################
        # REQUIRED COLUMNS
        ##########################################################

        required_columns = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in batch_df.columns
        ]

        if len(missing_columns) > 0:

            st.error(
                "The following required columns are missing:"
            )

            st.write(missing_columns)

            st.stop()

        ##########################################################
        # PREPROCESSING
        ##########################################################

        processed_df = batch_df.copy()

        # One-hot encoding
        processed_df = pd.get_dummies(
            processed_df,
            columns=["Geography", "Gender"],
            drop_first=True
        )

        ##########################################################
        # MATCH TRAINING FEATURES
        ##########################################################

        for column in feature_columns:

            if column not in processed_df.columns:

                processed_df[column] = 0

        processed_df = processed_df[feature_columns]

        ##########################################################
        # MAKE PREDICTIONS
        ##########################################################

        predictions = model.predict(processed_df)

        probabilities = model.predict_proba(processed_df)[:, 1]

        ##########################################################
        # CREATE RESULTS
        ##########################################################

        results = batch_df.copy()

        results["Prediction"] = predictions

        results["Prediction"] = results["Prediction"].map(
            {
                0: "No Churn",
                1: "Churn"
            }
        )

        results["Churn Probability"] = probabilities

        ##########################################################
        # SUMMARY STATISTICS
        ##########################################################

        st.markdown("---")

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Customers",
                len(results)
            )

        with col2:

            churn_count = (results["Prediction"] == "Churn").sum()

            st.metric(
                "Predicted Churn",
                churn_count
            )

        with col3:

            non_churn = (results["Prediction"] == "No Churn").sum()

            st.metric(
                "Predicted Non-Churn",
                non_churn
            )

        ##########################################################
        # DISPLAY RESULTS
        ##########################################################

        ### MAPIE for batch
        y_conformal_pred, y_prediction_sets = conformal_model.predict_set(processed_df)

        ############################################################
        # MAPIE CONFORMAL PREDICTION
        ############################################################

        y_conformal_pred, y_prediction_sets = (
            conformal_model.predict_set(processed_df)
        )
 
        ############################################################
        # CONFORMAL PREDICTION SET
        ############################################################

        conformal_sets = []

        for i in range(len(processed_df)):

            # Extract prediction set for customer i
            prediction_set = y_prediction_sets[i, :, 0]

            no_churn_in_set = bool(prediction_set[0])
            churn_in_set = bool(prediction_set[1])

            if no_churn_in_set and churn_in_set:

                conformal_result = "{No Churn, Churn}"

            elif churn_in_set:

                conformal_result = "{Churn}"

            elif no_churn_in_set:

                conformal_result = "{No Churn}"

            else:

                conformal_result = "Empty"

            conformal_sets.append(conformal_result)


        ############################################################
        # ADD CONFORMAL RESULTS
        ############################################################

        results["Conformal Prediction"] = conformal_sets

        ############################################################
        # CONFORMAL PREDICTION SUMMARY
        ############################################################

        st.markdown("---")

        st.subheader("Conformal Prediction Summary")

        conformal_counts = (
            results["Conformal Prediction"]
            .value_counts()
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            churn_only = conformal_counts.get(
                "{Churn}",
                0
            )

            st.metric(
                "Conformal Churn",
                churn_only
            )


        with col2:

            no_churn_only = conformal_counts.get(
                "{No Churn}",
                0
            )

            st.metric(
                "Conformal No Churn",
                no_churn_only
            )


        with col3:

            uncertain = conformal_counts.get(
                "{No Churn, Churn}",
                0
            )

            st.metric(
                "Uncertain",
                uncertain
            )

        ############################################################
        # DISPLAY RESULTS
        ############################################################

        st.markdown("---")

        st.subheader("Batch Prediction Results")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        ##########################################################
        # FILTER
        ##########################################################

        st.markdown("---")

        option = st.selectbox(

            "View",

            [
                "All Customers",
                "Only Churn Customers",
                "Only Non-Churn Customers"
            ]

        )

        if option == "Only Churn Customers":

            st.dataframe(

                results[
                    results["Prediction"] == "Churn"
                ],

                use_container_width=True

            )

        elif option == "Only Non-Churn Customers":

            st.dataframe(

                results[
                    results["Prediction"] == "No Churn"
                ],

                use_container_width=True

            )

        ##########################################################
        # DOWNLOAD RESULTS
        ##########################################################

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            label="Download Predictions",

            data=csv,

            file_name="customer_churn_predictions.csv",

            mime="text/csv"

        )

        # COMPUTE SHAP VALUES

        ##############################################################
        # GLOBAL MODEL INTERPRETABILITY
        ##############################################################

        st.markdown("---")

        st.subheader(
            "What Drives Churn Across These Customers?"
        )

        st.write(
            """
            This analysis looks across all customers in the uploaded
            dataset and identifies the characteristics that the model
            relies on most heavily when predicting customer churn.

            These insights can help the customer-care team identify
            customers or characteristics that may require closer attention.
            """
        )


        ##############################################################
        # COMPUTE SHAP VALUES
        ##############################################################

        explanation = explainer(processed_df)


        ##############################################################
        # SELECT POSITIVE CLASS
        ##############################################################

        if len(explanation.values.shape) == 3:

            shap_values = explanation.values[:, :, 1]

        else:

            shap_values = explanation.values


        ##############################################################
        # GLOBAL SHAP EXPLANATION OBJECT
        ##############################################################

        summary_exp = shap.Explanation(

            values=shap_values,

            base_values=np.repeat(

                np.mean(
                    np.atleast_1d(
                        explanation.base_values
                    )
                ),

                processed_df.shape[0]

            ),

            data=processed_df.values,

            feature_names=processed_df.columns.tolist()

        )


        ##############################################################
        # CALCULATE GLOBAL IMPORTANCE
        ##############################################################

        importance = pd.DataFrame({

            "Feature": processed_df.columns,

            "Mean |SHAP|": np.abs(
                shap_values
            ).mean(axis=0)

        })


        importance = importance.sort_values(
            "Mean |SHAP|",
            ascending=False
        )


        ##############################################################
        # TOP GLOBAL FACTORS
        ##############################################################

        top_global_features = importance.head(5)


        st.subheader(
            "Most Influential Churn Factors"
        )

        for i, (_, row) in enumerate(
            top_global_features.iterrows(),
            start=1
        ):

            st.write(
                f"**{i}. {row['Feature']}** — "
                f"one of the strongest factors used by the model "
                f"when predicting churn."
            )


        ##############################################################
        # GLOBAL SHAP BAR CHART
        ##############################################################

        st.markdown("---")

        st.subheader(
            "Global Feature Importance"
        )

        st.write(
            """
            The chart shows the features that have the greatest overall
            influence on the model's churn predictions across the uploaded
            customer dataset.
            """
        )


        fig = plt.figure(figsize=(8, 5))

        shap.plots.bar(
            summary_exp,
            max_display=15,
            show=False
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        ##############################################################
        # EXPLANATION UNDER SHAP BAR CHART
        ##############################################################

        st.info(
            """
            **How to read this chart**

            Features appearing closer to the top have a greater overall
            influence on the model's predictions.

            A feature with a larger importance value is used more strongly
            by the model when distinguishing customers who are likely to
            churn from those who are not.

            **Important:** Feature importance describes the model's behaviour.
            It does not mean that a feature directly causes customer churn.
            """
        )


        ##############################################################
        # TECHNICAL SHAP ANALYSIS
        ##############################################################

        with st.expander(
            "View Technical SHAP Analysis"
        ):

            st.subheader(
                "SHAP Beeswarm Plot"
            )

            fig = plt.figure(
                figsize=(10, 7)
            )

            shap.plots.beeswarm(
                summary_exp,
                max_display=15,
                show=False
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


            st.info(
                """
                The beeswarm plot provides a more detailed technical view
                of how individual feature values affect model predictions
                across the customer population.
                """
            )


        ##############################################################
        # GLOBAL FEATURE IMPORTANCE TABLE
        ##############################################################

        with st.expander(
            "View Advanced Feature Dependence Analysis"
        ):

            st.subheader(
                "Feature Dependence Plot"
            )

            st.write(
                """
                This technical analysis shows how the relationship between
                an individual feature and the model's prediction changes
                across the customer dataset.
                """
            )

            feature = st.selectbox(
                "Select a Feature",
                processed_df.columns,
                key="shap_dependence_feature"
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            shap.dependence_plot(
                feature,
                shap_values,
                processed_df,
                ax=ax,
                show=False
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.info(
                """
                **Interpretation:** This plot helps technical users
                understand how changes in a feature are associated with
                changes in the model's output.
                """
            )
        ######################

        with st.expander(
            "View Advanced Partial Dependence Analysis"
        ):

            st.subheader(
                "Partial Dependence Plot"
            )

            st.write(
                """
                The Partial Dependence Plot shows how the model's predicted
                churn behaviour changes as a selected feature changes,
                while averaging over the other features in the dataset.
                """
            )

            feature = st.selectbox(
                "Select a feature",
                processed_df.columns.tolist(),
                key="pdp_feature"
            )

            pdp_data = processed_df.copy()

            numeric_cols = (
                pdp_data
                .select_dtypes(include=[np.number])
                .columns
            )

            pdp_data[numeric_cols] = (
                pdp_data[numeric_cols]
                .astype(float)
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=pdp_data,
                features=[feature],
                kind="average",
                grid_resolution=50,
                ax=ax
            )

            ax.set_title(
                f"Partial Dependence Plot for {feature}"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.info(
                """
                **How to read this chart**

                The plot shows the model's average predicted response
                as the selected feature changes.

                This is a model-behaviour analysis and should not be
                interpreted as evidence that changing the feature will
                necessarily cause a customer to churn or remain.
                """
            )