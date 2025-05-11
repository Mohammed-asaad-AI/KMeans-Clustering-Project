# KMeans Clustering Project

This project demonstrates clustering using the KMeans algorithm across multiple datasets and complexity levels, including manual implementation and real-world application on the Titanic dataset.

## 📚 Contents

1. **Manual KMeans** — Built from scratch using NumPy.
2. **KMeans with scikit-learn** — Applied to synthetic 2D point clusters.
3. **Titanic Dataset Clustering**:
   - Data preprocessing and encoding
   - Visual exploration of survival patterns
   - KMeans clustering with evaluation metrics (accuracy + silhouette score)
   - PCA projection of clusters for visual understanding

## 📈 Visual Analysis Highlights

Before applying KMeans, we explore:
- Survival distribution
- Age distribution
- Fare vs survival
- Survival by gender
- Survival by age group (child vs adult)

These provide context for understanding clustering outcomes.

## 🧠 Insights

- **Females had higher survival rates** than males.
- **Children (<16 years)** had better survival chances than adults.
- **Higher ticket fare** correlated with higher survival — likely due to better class access.

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python kmeans_clustering_project.py
```

Place `titanic.xls` in the project root directory or download it from [Kaggle Titanic Data](https://www.kaggle.com/c/titanic/data).

## 📊 Evaluation Metrics

- **Accuracy**: Based on clustering label agreement with real survival outcomes.
- **Silhouette Score**: Measures cohesion/separation of formed clusters.
- **PCA Projection**: Reduces dimensions for intuitive visual interpretation.

## 🚀 Future Enhancements

- Compare with DBSCAN and Hierarchical Clustering
- Add GUI or Streamlit interface for interactive exploration
