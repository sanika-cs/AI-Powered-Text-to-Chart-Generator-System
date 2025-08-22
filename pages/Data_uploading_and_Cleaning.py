import streamlit as st
import pandas as pd
import numpy as np
from currency_converter import CurrencyConverter

st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title(" Data Cleaning for Visualization")

# === Sidebar: Upload File ===
st.header(" Upload File (CSV, Excel, JSON, Parquet, TXT, TSV)")
uploaded_file = st.file_uploader(
    "Upload your file",
    type=["csv", "xlsx", "xls", "json", "parquet", "txt", "tsv"]
)

# === Load Data into session_state ===
if uploaded_file:
    if "original_df" not in st.session_state:
        file_type = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_type in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_type == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "tsv":
                df = pd.read_csv(uploaded_file, sep="\t")
            elif file_type == "txt":
                df = pd.read_csv(uploaded_file, sep=None, engine="python")  # auto-detects delimiter
            elif file_type == "json":
                df = pd.read_json(uploaded_file)
            elif file_type == "parquet":
                df = pd.read_parquet(uploaded_file)
            else:
                st.error(" Unsupported file format")
                st.stop()

            # 🔹 Remove duplicates automatically
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            if before != after:
                st.sidebar.success(f"Removed {before - after} duplicate rows automatically ")
            else:
                st.sidebar.info("No duplicate rows found.")

            st.session_state["original_df"] = df.copy()
            st.session_state["df"] = df.copy()

        except Exception as e:
            st.error(f" Failed to read file: {e}")

    df = st.session_state["df"]


    # === Missing Value Summary ===
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]

    # === Sidebar: Missing Value Handling ===
    st.sidebar.header(" Handle Missing Values")
    if not missing_cols.empty:
        selected_col = st.sidebar.selectbox("Select column with NaN", missing_cols.index)
        method = st.sidebar.radio("Fill/Drop NaNs", ["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"])

        if st.sidebar.button("Apply Missing Handling"):
            if method == "Drop rows":
                df = df.dropna(subset=[selected_col])
            elif method == "Fill with Mean":
                df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
            elif method == "Fill with Median":
                df[selected_col] = df[selected_col].fillna(df[selected_col].median())
            elif method == "Fill with Mode":
                df[selected_col] = df[selected_col].fillna(df[selected_col].mode()[0])
            elif method == "Forward Fill":
                df[selected_col] = df[selected_col].fillna(method="ffill")
            elif method == "Backward Fill":
                df[selected_col] = df[selected_col].fillna(method="bfill")

            st.session_state["df"] = df
            st.success(f" Missing values in `{selected_col}` handled using `{method}`")
    else:
        st.sidebar.info(" No missing values.")

    # === Sidebar: Remove Characters from Column ===
    st.sidebar.header(" Remove Characters (e.g. $, %) from Column")
    clean_col = st.sidebar.selectbox("Select Column to Clean", df.columns)
    char_to_remove = st.sidebar.text_input("Character(s) to Remove (e.g. $ or % or ,)", key="char_remove")

    if st.sidebar.button("Clean Column"):
        try:
            df[clean_col] = df[clean_col].astype(str).str.replace(char_to_remove, "", regex=False)
            st.session_state["df"] = df
            st.success(f" Removed `{char_to_remove}` from `{clean_col}`")
        except Exception as e:
            st.error(f" Cleaning failed: {e}")

    # === Sidebar: Data Type Conversion ===
    st.sidebar.header(" Convert Data Types")
    dtype_col = st.sidebar.selectbox("Column to Convert Type", df.columns)
    new_type = st.sidebar.selectbox("New Data Type", ["int", "float", "str", "datetime"])

    if st.sidebar.button("Convert Type"):
        try:
            if new_type == "datetime":
                df[dtype_col] = pd.to_datetime(df[dtype_col])
            else:
                df[dtype_col] = df[dtype_col].astype(new_type)
            st.session_state["df"] = df
            st.success(f" Converted `{dtype_col}` to {new_type}")
        except Exception as e:
            st.error(f" Conversion failed: {e}")

    # === Sidebar: Replace Values ===
    st.sidebar.header(" Replace Specific Value")
    replace_col = st.sidebar.selectbox("Column to Replace", df.columns)
    to_replace = st.sidebar.text_input("Value to Replace")
    replace_with = st.sidebar.text_input("Replace With")

    if st.sidebar.button("Replace Value"):
        df[replace_col] = df[replace_col].replace(to_replace, replace_with)
        st.session_state["df"] = df
        st.success(f" `{to_replace}` replaced with `{replace_with}` in `{replace_col}`")
    st.sidebar.header(" Extract Date Parts from Column")
    date_col = st.sidebar.selectbox("Select Date Column", df.columns, key="date_extract")
    if st.sidebar.button("Extract Year/Month/Day"):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["Year"] = df[date_col].dt.year
            df["Month"] = df[date_col].dt.month
            df["Day"] = df[date_col].dt.day
            st.session_state["df"] = df
            st.success(f"Extracted Year, Month, Day from `{date_col}`")
        except Exception as e:
            st.error(f"Failed to extract date parts: {e}")

    # st.sidebar.header("Add a Calculated Column")
    #
    # # User inputs
    # new_col_name = st.text_input("Enter New Column Name", "NewColumn")
    # expression = st.text_area(
    #     "Enter Formula (use column names, e.g., Price * Quantity or Salary + Bonus)"
    # )
    #
    # if st.button("Add Column"):
    #     try:
    #         # Use eval to calculate formula safely
    #         df[new_col_name] = df.eval(expression)
    #         st.success(f" Column '{new_col_name}' added successfully!")
    #         st.dataframe(df)
    #     except Exception as e:
    #         st.error(f"Error: {e}")

    # === Currency Converter Section ===


    c = CurrencyConverter()

    st.sidebar.header("Currency Conversion")
    currency_col = st.sidebar.selectbox("Select Column with Currency Values", df.columns)

    source_currency = st.sidebar.selectbox("Source Currency", ["USD", "INR", "EUR", "GBP", "JPY"])
    target_currency = st.sidebar.selectbox("Target Currency", ["USD", "INR", "EUR", "GBP", "JPY"])

    if st.sidebar.button("Convert Currency"):
        try:
            df[currency_col] = df[currency_col].astype(float).apply(
                lambda x: c.convert(x, source_currency, target_currency)
            )
            st.session_state["df"] = df
            st.success(f"Converted `{currency_col}` from {source_currency} to {target_currency}")
        except Exception as e:
            st.error(f"Currency conversion failed: {e}")

    st.sidebar.header(" Rename Columns")

    # Dropdown to select which column to rename
    col_to_rename = st.sidebar.selectbox("Select a column to rename", df.columns)

    # Input for new column name
    new_col_name = st.sidebar.text_input("Enter the new column name")

    if st.sidebar.button("Rename Column"):
        if new_col_name.strip() != "":
            df = df.rename(columns={col_to_rename: new_col_name})
            st.session_state["df"] = df
            st.sidebar.success(f" Column '{col_to_rename}' renamed to '{new_col_name}'")
        else:
            st.error(" Please enter a valid new column name.")

    st.sidebar.header("Choose Columns")
    selected_columns = st.sidebar.multiselect("Choose columns", df.columns.tolist(), default=df.columns.tolist())

    df = df[selected_columns]
    st.session_state["df"] = df


    # === Main Area: Show Preview with Data Types ===
    st.subheader(" Data Preview (with Types)")
    typed_df = df.head(100).copy()
    typed_df.columns = [f"{col} ({df[col].dtype})" for col in df.columns]
    st.dataframe(typed_df)

    # === Show Missing Values Summary Again (Optional) ===
    st.subheader(" Current Missing Values Summary")
    updated_missing = df.isnull().sum()
    st.dataframe(updated_missing[updated_missing > 0].reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}))

    # === Download Cleaned Data ===
    st.subheader(" Download Cleaned Data")

    download_format = st.selectbox("Choose file format", ["CSV", "Excel"])

    if download_format == "CSV":
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    else:
        excel_file = pd.ExcelWriter("cleaned_data.xlsx", engine="xlsxwriter")
        df.to_excel(excel_file, index=False, sheet_name="Cleaned Data")
        excel_file.close()
        with open("cleaned_data.xlsx", "rb") as f:
            st.download_button(
                label="⬇ Download as Excel",
                data=f,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # Add this at the end of your current script
    if st.button("Next ➡ Go to Chart Creation"):
        st.switch_page("pages/Text_to_Chart.py")

else:
    st.info(" Upload a file to begin.")

