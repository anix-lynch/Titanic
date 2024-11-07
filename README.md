

# Titanic Survival Analysis and Prediction

This project explores and predicts passenger survival on the Titanic using machine learning techniques. It combines both classification and clustering approaches to gain insights into factors that contributed to passenger survival. The dataset is the classic Titanic dataset, originally sourced from Kaggle.

## Project Overview

The primary goals of this project are to:
1. **Predict Survival**: Use machine learning algorithms to predict whether a passenger survived based on features such as age, gender, fare, class, and more.
2. **Explore Relationships**: Understand how features like passenger class, fare, and family size correlate with survival rates.
3. **Cluster Passengers**: Perform clustering to segment passengers based on socioeconomic factors, which can reveal insights into survival patterns.

## Dataset

The data comes from Kaggle's Titanic competition. The dataset includes various passenger details, such as age, sex, fare, class, number of siblings/spouses aboard, and the port of embarkation. 

If you wish to recreate this analysis with the same dataset, you may need to [download it from Kaggle](https://www.kaggle.com/c/titanic/data) or use the data upload feature on Kaggle.

## Project Structure

- `notebooks/`: Contains the main Jupyter notebook(s) used for the analysis and model training.
- `data/`: Placeholder for the Titanic data files (`train.csv`, `test.csv`). 
- `models/`: Directory for saving trained models, such as the final pickled Gradient Boosting model.
- `README.md`: Project documentation (this file).

## Libraries Used

- **Pandas**: For data manipulation and preprocessing.
- **Seaborn & Matplotlib**: For data visualization.
- **Scikit-learn**: For model training, evaluation, and preprocessing.
- **SHAP & LIME**: For model interpretability.
- **KMeans (Clustering)**: For unsupervised clustering.
- **Gradient Boosting & Decision Trees**: For supervised learning and survival prediction.

## Analysis Steps

1. **Data Preprocessing**:
   - Drop irrelevant columns like `PassengerId`, `Name`, `Ticket`, and `Cabin`.
   - Fill missing values for `Age` and `Embarked`.
   - Encode categorical variables like `Sex` and `Embarked`.

2. **Feature Engineering**:
   - Created features like `FamilySize` and extracted `Deck` and `Title` from existing columns to add more context for predictions.
   - Applied log transformations to `Fare` to reduce the impact of outliers.

3. **Exploratory Data Analysis (EDA)**:
   - Visualized the distributions of key features to understand data patterns.
   - Explored relationships between features like `Pclass`, `Sex`, and `Fare` with survival.

4. **Model Training**:
   - Used multiple models including **Decision Trees**, **Random Forests**, **Gradient Boosting**, and **KMeans Clustering**.
   - Applied **GridSearchCV** for hyperparameter tuning to optimize model performance.

5. **Model Evaluation**:
   - Evaluated models using **Mean Squared Error (MSE)** and **R-squared (R²)**.
   - Used interpretability techniques like **SHAP** and **LIME** to understand feature importance and model decisions.

6. **Clustering Analysis**:
   - Performed KMeans clustering to segment passengers based on `Age` and `Fare`, identifying socioeconomic clusters.

7. **Pickle Model**:
   - Saved the optimized model using pickle to reuse the trained model for predictions on unseen data.

## Results and Insights

- **Important Features for Survival**: `Sex`, `Pclass`, and `Fare` were among the most influential features in predicting survival.
- **Socioeconomic Clustering**: Clustering revealed distinct groups, such as high-fare passengers who had a higher likelihood of survival, highlighting the socioeconomic divide on the Titanic.
- **Log Transformation Impact**: Log transformation on `Fare` improved model performance by reducing the effect of outliers.

## Requirements

To run this project, you’ll need the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `shap`
- `lime`
- `joblib`

You can install these dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime joblib
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-analysis.git
   cd titanic-analysis
   ```

2. Download the Titanic dataset from Kaggle and place `train.csv` and `test.csv` in the `data/` folder.

3. Open and run the notebook in the `notebooks/` directory to reproduce the analysis.

## Future Enhancements

- **Implement Additional Models**: Experiment with models like **XGBoost** and **LightGBM** to further improve prediction accuracy.
- **A/B Testing Simulations**: Use the dataset to simulate and analyze hypothetical survival scenarios based on passenger features.
- **Deploy Model**: Set up a web application using Flask or Streamlit to allow users to input passenger details and get survival predictions.


