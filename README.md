# AI-Powered Text-to-Chart Generator

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Flan--T5-orange)](https://huggingface.co/)

Transform plain text into interactive charts with AI. This project leverages **Flan-T5 models**, advanced **prompt engineering**, and **Streamlit** to provide an end-to-end solution for text-to-chart generation.

---

## 🚀 Project Overview

This application allows users to generate charts directly from **text queries** or **uploaded datasets** (CSV/Excel). It supports **aggregation**, **fuzzy column matching**, and corrects **spelling mistakes** using `rapidfuzz`.

Two Flan-T5 models are used:

1. **Chart Information Extraction** – Converts user text into structured chart instructions.  
2. **Filter Information Extraction** – Extracts filtering criteria from user text and is evaluated using **ROUGE metrics**.  

The app also includes a **data cleaning module** for uploaded datasets.

---

## 📂 Custom Dataset for Fine-Tuning

To fine-tune the Flan-T5 model for **filter information extraction**, a **custom dataset** of 7,500 examples was created using **synthetic data generation via prompt engineering**:

- **Prompt Engineering** – Carefully designed prompts were used to guide ChatGPT to produce realistic input-output pairs.  
- **Looped Generation** – Multiple variations were generated programmatically to cover diverse filtering scenarios.  
- **High-Quality Data** – Each example was validated to ensure consistency and usability for training.  
- **Purpose** – This dataset enables Flan-T5 to accurately extract filtering criteria from user text, improving the text-to-chart application.

---

## 🛠️ Key Features

- **Text-to-Chart Conversion** – Automatically converts text queries into charts  
- **CSV & Excel Support** – Upload datasets for visualization  
- **Aggregation Support** – Sum, mean, count, and custom aggregations  
- **Fuzzy Column Matching** – Handles spelling mistakes using `rapidfuzz`  
- **Interactive Streamlit App** – User-friendly web interface  
- **Plotly Charts** – Dynamic and interactive visualizations  
- **Custom Dataset** – Synthetic dataset fine-tuned with Flan-T5 using prompt engineering  

---

## 📈 Example

**Input:**  
> "Show total sales by region for products starting with 'A' in 2024"

**Output:**  
- Bar chart with regions on x-axis  
- Total sales on y-axis  
- Filtered for products starting with "A" in 2024  

---

## 🛠️ Technologies Used

- **Python** – Core programming language  
- **PyTorch & Hugging Face Transformers** – Flan-T5 fine-tuning & inference  
- **Streamlit** – Interactive web interface  
- **Plotly** – Chart visualization  
- **RapidFuzz** – Fuzzy column matching  
- **ROUGE** – Evaluation for filter extraction  

---

## 💻 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/text-to-chart-generator.git
cd text-to-chart-generator
