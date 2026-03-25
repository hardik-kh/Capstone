# DATA VISUALIZATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EDAAgent:

    def __init__(self, output_dir="eda_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, df: pd.DataFrame, name="dataset"):
        results = {}

        # 1. Basic Info
        results["shape"] = df.shape
        results["columns"] = list(df.columns)
        results["summary"] = df.describe(include='all').to_dict()

        # 2. Missing Values
        results["missing"] = df.isnull().sum().to_dict()

        # 3. Generate ONLY required plots
        self._plot_correlations(df, name)
        self._plot_boxplots(df, name)
        self._plot_scatter(df, name)

        return results

    # -------------------------------
    # 1. Correlation Heatmap
    # -------------------------------
    def _plot_correlations(self, df, name):
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True)
            plt.title("Correlation Matrix")
            plt.savefig(f"{self.output_dir}/{name}_correlation.png")
            plt.close()

    # -------------------------------
    # 2. Boxplots (Outliers)
    # -------------------------------
    def _plot_boxplots(self, df, name):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"{col} Boxplot")
            plt.savefig(f"{self.output_dir}/{name}_{col}_box.png")
            plt.close()

    # -------------------------------
    # 3. Scatter Plots (Relationships)
    # -------------------------------
    def _plot_scatter(self, df, name):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    plt.figure()
                    plt.scatter(df[numeric_cols[i]], df[numeric_cols[j]])
                    plt.xlabel(numeric_cols[i])
                    plt.ylabel(numeric_cols[j])
                    plt.title(f"{numeric_cols[i]} vs {numeric_cols[j]}")
                    plt.savefig(f"{self.output_dir}/{name}_{numeric_cols[i]}_{numeric_cols[j]}_scatter.png")
                    plt.close()


# -------------------------------
# RUN TEST
# -------------------------------
if __name__ == "__main__":
    # Load datasets
    stores_df = pd.read_csv("data/raw/stores.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    eda = EDAAgent()

    print("Running EDA on stores...")
    eda.run(stores_df, name="stores")

    print("Running EDA on test...")
    eda.run(test_df, name="test")

    print("EDA complete ✅")