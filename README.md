Machine Learning Project 

Group 46: Afonso Ascensão, 20240684; Duarte Marques, 20240522; Joana Esteves, 20240746; Rita Serra, 20240515; Rodrigo Luís, 20240742.


Overview: This project involves multiple notebooks that must be run in sequence. Some notebooks generate intermediate files required by subsequent steps.


Notebook and file dependencies:

1. ML_Project_46_EDA.ipynb:
- Requires: train_data.csv and test_data.csv.

2. ML_Project_46_preprocess.ipynb: 
- Requires: train_data.csv, test_data.csv and transformers.py.
- Generates: tranformed_train_data.parquet, tranformed_test_data.parquet, tranformed_train_split.parquet, transformed_val_split.parquet and pipeline.joblib.

3. ML_Project_46_Feature_Selection.ipynb: 
- Requires: tranformed_train_split.parquet, tranformed_val_split.parquet, pipeline.joblib and transformers.py.

4.ML_Project_46_Agreement Reached.ipynb: 
- Requires: transformed_test_data.parquet, transformed_train_data.parquet, pipeline.joblib and transformers.py.
- Generates: test_tranformed_agreement.parquet.

5. ML_Project_46_RandomForest; ML_Project_46_Stacking; ML_Project_46_XGBoost; ML_Project_46_NN.ipynb; ML_Project_46_kNN.ipynb:
- Require:  tranformed_train_data.parquet, test_tranformed_agreement.parquet, tranformed_train_split.parquet, transformed_val_split.parquet, pipeline.joblib and transformers.py.


Notebook running order:

- EDA: ML_Project_46_EDA.ipynb.

- Feature Selection: ML_Project_46_preprocess.ipynb > ML_Project_46_Feature_Selection.ipynb.

- Models: ML_Project_46_preprocess.ipynb > ML_Project_46_Agreement Reached.ipynb > ML_Project_46_RandomForest; ML_Project_46_Stacking; ML_Project_46_XGBoost; ML_Project_46_NN.ipynb; ML_Project_46_kNN.ipynb.