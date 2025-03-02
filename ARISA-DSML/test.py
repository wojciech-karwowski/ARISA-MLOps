import plotly.graph_objects as go
import pandas

# Create figure
fig = go.Figure()

# Add mean performance line
fig.add_trace(
    go.Scatter(
        x=cv_results["iterations"], y=cv_results["test-F1-mean"], mode="lines", name="Mean F1 Score", line=dict(color="blue")
    )
)

# Add shaded error region
fig.add_trace(
    go.Scatter(
        x=pd.concat([cv_results["iterations"], cv_results["iterations"][::-1]]),
        y=pd.concat([cv_results["test-F1-mean"]+cv_results["test-F1-std"], 
                     cv_results["test-F1-mean"]-cv_results["test-F1-std"]]),
        fill="toself", 
        fillcolor="rgba(0, 0, 255, 0.2)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        showlegend=False
    )
)

# Customize layout
fig.update_layout(
    title="Cross-Validation (N=5) Mean F1 score with Error Bands",
    xaxis_title="Training Steps",
    yaxis_title="Performance Score",
    template="plotly_white",
    yaxis=dict(range=[0.5, 1])
)

fig.show()

fig.write_image(outfolder / "test_f1_v2.png")

df_test = pd.read_csv(download_folder / "test.csv")
df_test = df_test.drop(columns=["Ticket"])
df_test_id = df_test.pop("PassengerId")
df_test = df_test.fillna({"Embarked": "N", "Age": X_train["Age"].mean()})

pattern = r'([A-Za-z]+)(\d+)'
matches = df_test['Cabin'].str.extractall(pattern)
matches.reset_index(inplace=True)
result = matches.pivot(index='level_0', columns='match', values=[0, 1])
result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
df_test = df_test.join(result[["0_0", "1_0"]])
df_test["1_0"] = df_test["1_0"].astype(float)
df_test = df_test.fillna({"0_0": "N", "1_0": X_train["CabinNumber"].mean()})
df_test["1_0"] = df_test["1_0"].astype(int)
df_test = df_test.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})

df_test["Title"] = df_test["Name"].apply(extract_title)

df_test.drop(columns=["Cabin", "Name"], axis=1, inplace=True)
df_test["Title"].unique()

preds = model.predict(df_test[X_train.columns])

import shap
import matplotlib.pyplot as plt
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_test[X_train.columns])

shap.summary_plot(shap_values, df_test, show=False)
plt.savefig(outfolder / "test_shap_overall_v2.png")

df_test["PassengerId"] = df_test_id
df_test["Survived"] = preds

df_test[["PassengerId", "Survived"]].to_csv(outfolder / "predictions_v2.csv", index=False)