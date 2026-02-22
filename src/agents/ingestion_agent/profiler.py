def profile_dataframe(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_ratio": {
            c: float(df[c].isna().mean()) for c in df.columns
        },
        "numeric_summary": df.describe(include="number").to_dict(),
        "preview": df.head(10).to_dict(orient="records"),
    }
