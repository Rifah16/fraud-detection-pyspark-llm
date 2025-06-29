
# 🚨 Fraud Detection with PySpark + Transformers

This project demonstrates an end-to-end pipeline for detecting fraudulent insurance claims using structured data and unstructured text fields.

It uses PySpark for scalable data processing and Hugging Face's `MiniLM` transformer to generate semantic embeddings from claim descriptions. The final model is a Gradient-Boosted Tree (GBT) classifier built using Spark MLlib.

## 🔍 Key Features

- Large-scale preprocessing with PySpark
- Advanced data cleaning & feature engineering
- Text embedding with Hugging Face MiniLM (`sentence-transformers`)
- Combined structured + unstructured features
- Gradient Boosted Tree classifier
- ROC AUC evaluation with Spark's BinaryClassificationEvaluator

## 📁 Repo Structure

```
fraud-detection-pyspark-llm/
├── notebooks/
│   └── fraud_detection_pyspark_llm.ipynb
├── src/
│   └── embedding_utils.py           
├── data/                            # synthetic data
├── requirements.txt
└── README.md
```

---

## 📓 Notebook

📘 [fraud_detection_pyspark_llm.ipynb](notebooks/fraud_detection_pyspark_llm.ipynb)  
Includes data loading, cleaning, embedding, and modeling steps with full PySpark and Python code.

## 🛠 Tech Stack

- PySpark (3.x)
- Hugging Face Transformers
- Pandas + Torch
- Gradient Boosted Trees (GBTClassifier)

## 🧪 How to Run

1. Clone the repo  
   `git clone https://github.com/yourusername/fraud-detection-pyspark-llm.git`

2. Install requirements  
   `pip install -r requirements.txt`

3. Run the notebook  
   Use Jupyter or Databricks depending on your environment.

## 📚 Related Article

📖 Read the full write-up on Medium: [How I Built a Fraud Detection Engine in PySpark with LLMs](#)

## 💡 Recommended Resources

- 🤖 [Hugging Face Transformers](https://huggingface.co/?ref=affiliate-id)
- 📘 [DataCamp PySpark Courses](https://datacamp.pxf.io/affiliate-id)
- 📋 [Notion for ML Project Tracking](https://notion.grsm.io/affiliate-id)
- 🧪 [Weights & Biases for Experiment Logging](https://wandb.ai/site?ref=affiliate-id)

## 🛡 License

This project is licensed under the MIT License.

