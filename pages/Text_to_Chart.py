
import streamlit as st
st.set_page_config(page_title="Text2Chart Dashboard", layout="wide")

from database import get_connection
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import plotly.express as px

import plotly.graph_objects as go

from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import tempfile

# -------------------------------
# Load Faster-Whisper once (cached)
# -------------------------------
@st.cache_resource
def load_whisper_model():
    # You can choose "tiny", "base", "small", "medium", "large-v2"
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()

# # -------------------------------
# # Model Loading
# # -------------------------------


@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(r"C:\Users\sanik\PycharmProjects\flant5-text2chart-tuned")
    tokenizer = T5Tokenizer.from_pretrained(r"C:\Users\sanik\PycharmProjects\flant5-text2chart-tuned")
    return model, tokenizer

@st.cache_resource
def load_filter_model():
    model = T5ForConditionalGeneration.from_pretrained(r"C:\Users\sanik\PycharmProjects\flan-t5-filter-tuned_final")
    tokenizer = T5Tokenizer.from_pretrained(r"C:\Users\sanik\PycharmProjects\flan-t5-filter-tuned_final")
    return model, tokenizer

# -------------------------------
# Utility Functions
# -------------------------------
def save_query(user_input, chart_type):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries (user_input, chart_type) VALUES (?, ?)", (user_input, chart_type))
    conn.commit()
    conn.close()

def generate_chart_text(model, tokenizer, input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(model.device)
    outputs = model.generate(inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
import re

def generate_filter_from_text(model, tokenizer, prompt: str, max_length: int = 64) -> dict:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    try:
        # Simple regex-based parse
        match = re.match(r"column:\s*(.+?);\s*operator:\s*(.+?);\s*value:\s*(.+)", output_text)
        if match:
            column = match.group(1).strip()
            operator = match.group(2).strip()
            value = match.group(3).strip().strip('"').strip("'")

            # Clean up
            if operator == "=":
                operator = "=="

            # Handle numeric range
            if operator.lower() == "between" and "-" in value:
                min_val, max_val = value.split("-")
                return {
                    "column": column,
                    "operator": "between",
                    "value": (float(min_val.strip()), float(max_val.strip()))
                }

            return {
                "column": column,
                "operator": operator,
                "value": value
            }
        else:
            raise ValueError("Could not parse filter output string.")
    except Exception as e:
        raise ValueError(f"Failed to parse generated output: {output_text}") from e

from rapidfuzz import process, fuzz

def match_column_name(input_col, df_columns, threshold=75):
    """
    Match input_col to the best column name from df_columns using multiple RapidFuzz scorers.
    Returns the best match if score >= threshold, else None.
    """
    scorers = [
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_sort_ratio,
        fuzz.token_set_ratio
    ]

    best_match = None
    best_score = 0

    for scorer in scorers:
        match, score, _ = process.extractOne(input_col, df_columns, scorer=scorer)
        if score > best_score:
            best_match = match
            best_score = score

    return best_match if best_score >= threshold else None



def apply_filter_gen(df, filter_dict):
    col = match_column_name(filter_dict.get('column'), df.columns)
    if not col:
        raise ValueError(f"No matching column found for '{filter_dict.get('column')}'")

    op = filter_dict.get('operator')
    val = filter_dict.get('value')

    col_dtype = df[col].dtype

    def convert_value(v):
        if pd.api.types.is_numeric_dtype(col_dtype):
            return float(v)
        elif pd.api.types.is_datetime64_any_dtype(col_dtype):
            return pd.to_datetime(v)
        else:
            return str(v)

    try:
        if isinstance(val, (list, tuple)):  # handles both lists and tuples
            val = [convert_value(v) for v in list(val)]  # convert tuple to list
        else:
            val = convert_value(val)

    except Exception as e:
        raise ValueError(f"Error converting value '{val}' to match column '{col}' dtype {col_dtype}: {e}")

    # Apply filter based on operator
    if op == '>':
        return df[df[col] > val]
    elif op == '<':
        return df[df[col] < val]
    elif op == '>=':
        return df[df[col] >= val]
    elif op == '<=':
        return df[df[col] <= val]
    elif op in ['==', '=', 'equals']:
        return df[df[col] == val]
    elif op == '!=':
        return df[df[col] != val]
    elif op == 'contains':
        return df[df[col].astype(str).str.contains(str(val), case=False)]
    elif op == 'starts with':
        return df[df[col].astype(str).str.startswith(str(val))]
    elif op == 'ends with':
        return df[df[col].astype(str).str.endswith(str(val))]
    elif op in ['between', 'range']:
        if not (isinstance(val, list) and len(val) == 2):
            raise ValueError(f"'between' operator requires a list of two values, got: {val}")
        lower, upper = sorted(val)
        return df[(df[col] >= lower) & (df[col] <= upper)]
    else:
        raise ValueError(f"Unsupported operator: {op}")

def create_chart(df, t5_output, filter_model=None, filter_tokenizer=None, chart_model=None, chart_tokenizer=None):
    chart_info = {}
    parts = [part.strip() for part in t5_output.split(",")]
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            chart_info[k.strip()] = v.strip()

    x = match_column_name(chart_info.get("x", ""), df.columns)
    y = match_column_name(chart_info.get("y", ""), df.columns)
    group = match_column_name(chart_info.get("group", ""), df.columns)
    stack = match_column_name(chart_info.get("stack", ""), df.columns)
    size_col= match_column_name(chart_info.get("size", ""), df.columns)
    y1 = match_column_name(chart_info.get("y1", ""), df.columns)
    y2 = match_column_name(chart_info.get("y2", ""), df.columns)
    location = match_column_name(chart_info.get("location", ""), df.columns)
    color_y = match_column_name(chart_info.get("color", ""), df.columns)

    agg = chart_info.get("agg", "sum").lower()
    filter_text = chart_info.get("filter", "")
    chart_type = chart_info.get("chart", "")

    # Apply filter
    if filter_text and filter_model and filter_tokenizer:
        try:
            filter_dict = generate_filter_from_text(filter_model, filter_tokenizer,f"Extract filters: {filter_text}")
            df = apply_filter_gen(df, filter_dict)
            #st.success(f"Applied: `{filter_dict}`")
        except Exception as e:
            st.error(f"Filter failed: {e}")



    # KPI: skip aggregation
    if chart_type == "kpi" or chart_type == "gauge" :
        try:
            if agg in ["sum", "total"]:
                agg_value = df[y].sum()
            elif agg in ["mean", "average"]:
                agg_value = df[y].mean()
            elif agg == "count":
                agg_value = df[y].count()
            elif agg in ["min", "minimum"]:
                agg_value = df[y].min()
            elif agg in ["max", "maximum"]:
                agg_value = df[y].max()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_value = df[y].sum()

            # st.metric(label=y, value=round(agg_value, 2))

        except Exception as e:
            st.error(f"KPI calculation error: {e}")
            return
    elif chart_type=='box' or chart_type=='violin' or chart_type=='histogram' or chart_type=='correlation_heatmap' :
        agg_df=df
    elif chart_type=='choropleth':
        group_cols = [location]
        if agg in ["sum", "total"]:
            agg_df = df.groupby(group_cols)[color_y].sum().reset_index()
        elif agg in ["mean", "average"]:
            agg_df = df.groupby(group_cols)[color_y].mean().reset_index()
        elif agg == "count":
            agg_df = df.groupby(group_cols)[color_y].count().reset_index()
        elif agg in ["min", "minimum"]:
            agg_df = df.groupby(group_cols)[color_y].min().reset_index()
        elif agg in ["max", "maximum"]:
            agg_df = df.groupby(group_cols)[color_y].max().reset_index()
        else:
            st.warning("Unknown aggregation, using sum.")
            agg_df = df.groupby(group_cols)[color_y].sum().reset_index()
    elif chart_type=='bubble':
        if  size_col == None:
            size_col=y
        bubble_group_cols = [x]
        try:
            if agg in ["sum", "total"]:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'sum'
                }).reset_index()
            elif agg in ["mean", "average"]:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'mean'
                }).reset_index()
            elif agg == "count":
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'count'
                }).reset_index()
            elif agg in ["min", "minimum"]:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'min'
                }).reset_index()
            elif agg in ["max", "maximum"]:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col:'max'
                }).reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(bubble_group_cols).agg({
                          y: 'sum',
                          size_col: 'sum'
                          }).reset_index()
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return
    elif chart_type=='combo':
        combo_group_cols = [x]
        try:
            if agg in ["sum", "total"]:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'sum'
                }).reset_index()
            elif agg in ["mean", "average"]:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'mean'
                }).reset_index()
            elif agg == "count":
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'count'
                }).reset_index()
            elif agg in ["min", "minimum"]:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'min'
                }).reset_index()
            elif agg in ["max", "maximum"]:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2:'max'
                }).reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(combo_group_cols).agg({
                          y1: 'sum',
                          y2: 'sum'
                          }).reset_index()
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return
    else:
        # Aggregation for charts (excluding KPI)
        try:
            if chart_type == "grouped_bar" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_bar" and stack:
                group_cols = [x, stack]
            elif chart_type == "multi_line" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_area" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_area" and stack:
                group_cols = [x, stack]

            else:
                group_cols = [x]

            if agg in ["sum", "total"]:
                agg_df = df.groupby(group_cols)[y].sum().reset_index()
            elif agg in ["mean", "average"]:
                agg_df = df.groupby(group_cols)[y].mean().reset_index()
            elif agg == "count":
                agg_df = df.groupby(group_cols)[y].count().reset_index()
            elif agg in ["min", "minimum"]:
                agg_df = df.groupby(group_cols)[y].min().reset_index()
            elif agg in ["max", "maximum"]:
                agg_df = df.groupby(group_cols)[y].max().reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(group_cols)[y].sum().reset_index()

        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

    # Color picker
    color =  "#1f77b4" #st.color_picker("Pick color", "#1f77b4", key=f"color_{x}_{y}_{chart_type}_{group}_{stack}_{size_col}")

    try:
        if chart_type == "bar":
            fig = px.bar(agg_df, x=x, y=y, color_discrete_sequence=[color])
        elif chart_type == "horizontal_bar":
            fig= px.bar(agg_df, x=y, y=x, orientation="h", title="Horizontal Bar Chart")
        elif chart_type == "grouped_bar" and group:
            fig = px.bar(agg_df, x=x, y=y, color=group, barmode="group")
        elif chart_type == "stacked_bar":
            fig= px.bar(agg_df, x=x, y=y, color=stack, barmode="stack", title="Stacked Bar Chart")
        elif chart_type == "line":
            fig = px.line(agg_df, x=x, y=y, line_shape="linear")
        elif chart_type == "multi_line" and group:
            fig= px.line(agg_df, x=x, y=y, color=group,  title="Multi-Line Chart")
        elif chart_type == "step_line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agg_df[x],
                y=agg_df[y],
                mode='lines',
                line=dict(shape='hv'),  # "hv" = horizontal then vertical steps
                name="Step Line"
            ))
            fig.update_layout(title="Step Line Chart", xaxis_title=x, yaxis_title=y)

        elif chart_type == "area":
            fig = px.area(agg_df, x=x, y=y, color=group) if group else px.area(agg_df, x=x, y=y)
            fig.update_layout(title="Area Chart", xaxis_title=x, yaxis_title=y)


        elif chart_type == "stacked_area":
            fig = px.area(agg_df, x=x, y=y, color=stack or group , groupnorm="percent")
            fig.update_layout(title="Stacked Area Chart", xaxis_title=x, yaxis_title=y)
        elif chart_type == "histogram":
            fig = px.histogram(agg_df, x=y, color=group if group else None, nbins=30)
            fig.update_layout(title="Histogram", xaxis_title=x, yaxis_title="Count")

        elif chart_type == "box":
            fig = px.box(agg_df, x=x, y=y)
            fig.update_layout(title="Box Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "violin":
            fig = px.violin(agg_df, x=x , y=y,  box=True)
            fig.update_layout(title="Violin Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "pie":
            fig = px.pie(agg_df, names=x, values=y)
        elif chart_type == "donut":
            fig = px.pie(agg_df, names=x, values=y, hole=0.5)
        elif chart_type == "treemap":
            fig = px.treemap(agg_df, path=[x], values=y, title="Treemap Chart")
        elif chart_type == "waterfall":
            fig = go.Figure(go.Waterfall(
                x=agg_df[x],
                y=agg_df[y],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(
                title="Waterfall Chart",
                xaxis_title=x,
                yaxis_title=y
            )
        elif chart_type == "funnel":
            agg_df = agg_df.sort_values(by=y, ascending=False)
            total = agg_df[y].sum()
            stages=agg_df[x]
            values=agg_df[y]
            labels = [f"{round(v / total * 100, 2)}%" for stage, v in zip(stages, values)]
            n = len(stages)
            colors = [f'rgba(0, 0, 255, {opacity})' for opacity in [1.0 - 0.7 * (i / (n - 1)) for i in range(n)]]

            fig = go.Figure(go.Funnel(

                            y=stages,
                            x=values,
                            text=labels,
                            textposition="inside",
                            marker={"color": colors},
                            textfont=dict(size=36),
                            opacity=0.9
                        ))
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y
            )

        elif chart_type == "kpi":


              # or .mean(), max, etc., based on context
            fig = go.Figure(go.Indicator(
                mode="number",
                value=agg_value,
                title={"text": f"KPI: {y}"},
                number={'font': {'size': 48}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
        # Gauge chart
        elif chart_type == "gauge":
            # User input for target/reference value
            target_value = st.number_input("Enter target value (reference for delta):", value=0.0)

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=agg_value,
                delta={'reference': target_value},
                gauge={
                    'axis': {'range': [None, df[y].max()]},
                    'bar': {'color': "mediumslateblue"},
                    'steps': [
                        {'range': [0, df[y].max() * 0.5], 'color': 'lightgray'},
                        {'range': [df[y].max() * 0.5, df[y].max()], 'color': 'gray'}
                    ],
                },
                title={'text': f"{y} Gauge"}
            ))

        elif chart_type == "combo":
            if not y1 or not y2:
                st.warning("Combo chart requires both y1 and y2.")
                return

            fig = go.Figure()

            # Bar chart (y1 - left axis)
            fig.add_trace(go.Bar(
                x=agg_df[x],
                y=agg_df[y1],
                name=y1,
                yaxis='y1'
            ))

            # Line chart (y2 - right axis)
            fig.add_trace(go.Scatter(
                x=agg_df[x],
                y=agg_df[y2],
                name=y2,
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='orange')
            ))

            # Set layout with secondary y-axis
            fig.update_layout(
                title="Combo Chart",
                xaxis=dict(title=x),
                yaxis=dict(title=y1),  # Primary y-axis (left)
                yaxis2=dict(
                    title=y2,
                    overlaying='y',
                    side='right',
                    showgrid=False  # Optional: hides grid from second y-axis
                ),
                legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'),
                bargap=0.3,
                height=500
            )

        elif chart_type == "scatter":
            fig = px.scatter(agg_df, x=x, y=y)
            fig.update_layout(title="Scatter Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "bubble":
            size_scaled = (agg_df[size_col] - agg_df[size_col].min()) / (
                        agg_df[size_col].max() - agg_df[size_col].min())
            agg_df["scaled_size"] = 10 + size_scaled * 90  # Bubble size between 10 and 100

            fig = px.scatter(agg_df, x=x, y=y, size=size_scaled, color=size_col if size_col else None, hover_name=x,size_max=100)
            fig.update_layout(title="Bubble Chart", xaxis_title=x, yaxis_title=y)
        elif chart_type == "correlation_heatmap":
            corr = agg_df.select_dtypes(include='number').corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(title="Correlation Heatmap")

        elif chart_type == "choropleth":

            fig = px.choropleth(
                df,
                locations=location,  # e.g., 'country'
                locationmode="country names",  # or "ISO-3" based on your data
                color=color_y,  # e.g., 'value' or 'sales'
                hover_name=location,
                color_continuous_scale="Viridis",
                template="plotly_white"
            )


        else:
            st.error(f"Unsupported chart type: `{chart_type}`")
            return

        st.plotly_chart(fig, use_container_width=True)


    except Exception as e:
        st.error(f"Chart rendering failed: {e}")





# -------------------------------
# Streamlit UI (Multi-chart Dashboard)
# -------------------------------


# Load dataframe
if "df" not in st.session_state:
    st.warning("Please upload and clean data on the previous page.")
    st.stop()

df = st.session_state.df.copy()
model, tokenizer = load_model()
filter_model, filter_tokenizer = load_filter_model()

# Sidebar - Control Panel
st.sidebar.header("Chart Controls")

# Number of charts
num_charts = st.sidebar.slider("Number of charts", min_value=1, max_value=25, value=3)
chart_instructions = []
# -------------------------------
# Chart Instructions Loop (Fixed)
# -------------------------------
for i in range(num_charts):
    st.sidebar.markdown(f"### Chart {i+1} Instruction")
    col1, col2 = st.sidebar.columns([3, 1])

    # Keys
    transcribed_key = f"instruction_{i}_transcribed"
    edited_key = f"instruction_{i}_edited"

    # -------------------------------
    # Column 1: Text Areas
    # -------------------------------
    with col1:
        # Show last transcription (read-only)
        transcription = st.session_state.get(transcribed_key, "")
        # st.text_area(
        #     " Transcription (auto)",
        #     transcription,
        #     key=f"display_{transcribed_key}",
        #     height=70,
        #     disabled=True,
        # )

        # Editable field (final input)
        edited_text = st.text_area(
            f" Edit instruction {i+1}",
            st.session_state.get(edited_key, ""),
            key=edited_key,
            height=80,
        )

    # -------------------------------
    # Column 2: Audio Recorder
    # -------------------------------
    with col2:
        audio_bytes = audio_recorder(text="", icon_size="2x", key=f"recorder_{i}")

        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                temp_audio_file = f.name

            segments, info = whisper_model.transcribe(temp_audio_file, beam_size=5)
            transcription = " ".join([seg.text for seg in segments])

            # Save transcription separately (will show next rerun)
            st.session_state[transcribed_key] = transcription
            st.success(f"{transcription}")

    # -------------------------------
    # Final Instruction Preview
    # -------------------------------
    final_instruction = st.session_state.get(edited_key, "").strip() or st.session_state.get(transcribed_key, "")
    # if final_instruction:
    #     st.markdown(f"** Final Instruction (Chart {i+1}):** `{final_instruction}`")

    chart_instructions.append(final_instruction)


# Chart Instructions
# chart_instructions = []
# for i in range(num_charts):
#     chart_instructions.append(st.sidebar.text_input(f"Chart {i+1} Instruction", key=f"instruction_{i}"))
#
# # Chart Types (for each)
# -------------------------------
#  Natural Language Global Filters
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader(" Global Filters (Natural Language)")

num_nl_filters = st.sidebar.slider("Number of Filters", min_value=0, max_value=5, value=2)
nl_filters = []
for i in range(num_nl_filters):
    nl_filters.append(
        st.sidebar.text_input(f"Filter {i+1}", key=f"nl_filter_{i}")
    )


#
parsed_filters = []
for raw_filter in nl_filters:
    if raw_filter.strip():
        try:
            parsed = generate_filter_from_text(
                filter_model, filter_tokenizer, f"Extract filters: {raw_filter.strip()}"
            )
            parsed_filters.append(parsed)
            st.sidebar.success(f"Parsed: {parsed}")
        except Exception as e:
            st.sidebar.error(f"Failed to parse: {raw_filter}\n{e}")

filtered_df = df.copy()
for pf in parsed_filters:
    try:
        filtered_df = apply_filter_gen(filtered_df, pf)
    except Exception as e:
        st.warning(f"Skipping filter due to error: {e}")

# Main Data View
st.subheader(" Filtered Dataset View")
st.dataframe(filtered_df.head(100))

# Render Charts
st.subheader("AI-Generated Charts")



for i, instruction in enumerate(chart_instructions):
     if instruction.strip():
        st.markdown(f"### Chart {i+1}: `{instruction}`")
        with st.spinner("Generating..."):
            t5_output = generate_chart_text(model, tokenizer, instruction)
            # st.code(t5_output, language="text")
            # Chart function modified below to support types
            chart_info = {}
            parts = [part.strip() for part in t5_output.split(",")]
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    chart_info[k.strip()] = v.strip()
            chart_type = chart_info.get("chart", "")
            save_query(instruction, chart_type)
            create_chart(filtered_df, t5_output, filter_model, filter_tokenizer)




