
# ====== IMPORTS ====== #
import numpy as np
import pandas as pd
# Preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# Feature selection
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency, levene


# ====== CLASSES ====== #
   
# Imputation Transformer
class ImputeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.wage_medians = X.groupby('Industry Code')['Average Weekly Wage'].median()  
        self.median_accident_year = X['Accident Year'].median()
        self.gender_mode = X['Gender'].mode()[0]
        self.adr_mode = X['Alternative Dispute Resolution'].mode()[0]
        self.median_age_injury = X['Age at Injury'].median()
        self.median_ime4_count = X['IME-4 Count'].median()

        self.mode_values = {  
            col: X[col].mode()[0]
            for col in ['Industry Code', 'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code']
        }

        return self

    def transform(self, X):
        X = X.copy()

        # Fill wage nan with median according to industry code
        def fill_nan_wage(row):
            if pd.isnull(row['Average Weekly Wage']):
                return self.wage_medians.get(row['Industry Code'], self.wage_medians.median())
            else:
                return row['Average Weekly Wage']

        # Apply preprocessing with pre-calculated values based on train
        X['Average Weekly Wage'] = X.apply(fill_nan_wage, axis=1)

        # Calculate 'Age at Injury' when accident year and birth year are available
        X['Age at Injury'] = np.where(
            X['Accident Year'].notna() & X['Birth Year'].notna(),
            (X['Accident Year'] - X['Birth Year']).astype('Int64'),
            X['Age at Injury']
        )

        # Fill remaining age nan with median
        X['Age at Injury'] = X['Age at Injury'].fillna(self.median_age_injury).astype('int64')

        # This variable is 9% nan and is redundant with "Age at Injury" 
        X = X.drop(columns=["Birth Year"])

        # Fill accident year nan with median
        X['Accident Year'] = X['Accident Year'].fillna(self.median_accident_year).astype('int64')

        # Fill ime-4 count nan with median
        X['IME-4 Count'] = X['IME-4 Count'].fillna(self.median_ime4_count).astype('int64')

        # Correct cases where accident year is after assembly year after inputation
        X.loc[X['Assembly Year'] < X['Accident Year'], "Accident Year"] = X['Assembly Year']

        # Fill gender nan and map to boolean
        X['Gender'] = X['Gender'].fillna(self.gender_mode)
        X['Gender'] = X['Gender'].astype('bool')

        # Fill ADR nan and map to boolean
        X['Alternative Dispute Resolution'] = X['Alternative Dispute Resolution'].fillna(self.adr_mode)
        X['Alternative Dispute Resolution'] = X['Alternative Dispute Resolution'].astype('bool')


        # Fill categorical variables with mode
        for col, mode_value in self.mode_values.items():
            X[col] = X[col].fillna(mode_value)
        
        return X
    
# Handle outliers
class OutlierHandlingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_at_injury_bounds = None
        self.accident_year_bounds = None
        self.ime_4_count_limits = None
        self.avg_weekly_wage_limits = None

    def fit(self, X, y=None):
        # Calculate IQR bounds for 'Age at Injury' 
        Q1 = X['Age at Injury'].quantile(0.25)
        Q3 = X['Age at Injury'].quantile(0.75)
        IQR = Q3 - Q1
        self.age_at_injury_bounds = (None, Q3 + 1.5 * IQR)  # Only upper bound

        # Calculate IQR bounds for 'Accident Year' 
        Q1 = X['Accident Year'].quantile(0.25)
        Q3 = X['Accident Year'].quantile(0.75)
        IQR = Q3 - Q1
        self.accident_year_bounds = (Q1 - 1.5 * IQR, None)  # Only lower bound

        # Calculate capping limits for 'IME-4 Count' (top 1%)
        lower_limit_ime = np.percentile(X['IME-4 Count'], 0)
        print('l-ime',lower_limit_ime)
        upper_limit_ime = np.percentile(X['IME-4 Count'], 99)
        print('u-ime',upper_limit_ime)
        self.ime_4_count_limits = (lower_limit_ime, upper_limit_ime)

        # Calculate capping limits for 'Average Weekly Wage' (top 5%)
        lower_limit_wage = np.percentile(X['Average Weekly Wage'], 0.0)
        print('l-wage',lower_limit_wage)
        upper_limit_wage = np.percentile(X['Average Weekly Wage'], 95.0)
        print('u-wage',upper_limit_wage)
        self.avg_weekly_wage_limits = (lower_limit_wage, upper_limit_wage)

        return self

    def transform(self, X):
        X = X.copy()

        X['Age at Injury'] = X['Age at Injury'].astype('float64')
        # Apply IQR-based capping 
        if self.age_at_injury_bounds[1] is not None:
            X['Age at Injury'] = X['Age at Injury'].clip(upper=self.age_at_injury_bounds[1]).round()

        X['Accident Year'] = X['Accident Year'].astype('float64')
        # Apply IQR-based capping
        if self.accident_year_bounds[0] is not None:
            X['Accident Year'] = X['Accident Year'].clip(lower=self.accident_year_bounds[0]).round()

        X['IME-4 Count'] = X['IME-4 Count'].astype('float64')
        # Cap top 1%
        X['IME-4 Count'] = X['IME-4 Count'].clip(lower=self.ime_4_count_limits[0], upper=self.ime_4_count_limits[1]).round()

        # Cap top 5%
        X['Average Weekly Wage'] = X['Average Weekly Wage'].clip(lower=self.avg_weekly_wage_limits[0], upper=self.avg_weekly_wage_limits[1])

        return X

# Feature Engeneering
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.wage_95th = None
        self.ime4_95th = None

    def fit(self, X, y=None):
        # Compute and store quantiles based on training data
        self.wage_95th = X['Average Weekly Wage'].quantile(0.95)
        self.ime4_95th = X['IME-4 Count'].quantile(0.95)
        return self

    def transform(self, X):
        X = X.copy()
        
        # ===== Binary variables =====
        X['Hearing Held'] = X['First Hearing Date'].notna()
        X['C-2 Delivered'] = X['C-2 Date'].notna()
        X['C-3 Delivered'] = X['C-3 Date'].notna()
        # C-2 Delivered on Time <= 10 days from Accident
        X['C-2 Delivered on Time'] = (
        ((X['C-2 Date'] - X['Accident Date']).dt.days)
        .where(X['C-2 Delivered'] & X['Accident Date'].notna(), other=1000)
        .astype('int64') <= 10
        )

        # C-3 Delivered on Time <= 730 days from Accident
        X['C-3 Delivered on Time'] = (
        ((X['C-3 Date'] - X['Accident Date']).dt.days)
        .where(X['C-3 Delivered'] & X['Accident Date'].notna(), other=1000)
        .astype('int64') <= 730
        )
        
        # ===== Mathematical transformations =====
        X['Average Weekly Wage Log'] = X['Average Weekly Wage'].apply(lambda x: np.log(x))
        X['IME-4 Count Log'] = X['IME-4 Count'].apply(lambda x: np.log1p(x))

        # ===== Temporal lags =====
        # Calculate 'Time Accident to Assembly' only if both dates are present
        X['Time Accident to Assembly'] = np.where(X['Assembly Date'].notna() & X['Accident Date'].notna(),(X['Assembly Date'] - X['Accident Date']).dt.days, 0)
        # Calculate 'Time Assembly to Hearing' only if both dates are present
        X['Time Assembly to Hearing'] = np.where(X['First Hearing Date'].notna() & X['Assembly Date'].notna(),(X['First Hearing Date'] - X['Assembly Date']).dt.days,0)

        # ===== Union of binary variables =====
        X['At/rp OR altr dispute'] = (X['Attorney/Representative'] | X['Alternative Dispute Resolution'])
        X['Covid OR altr dispute'] = (X['COVID-19 Indicator'] | X['Alternative Dispute Resolution'])
        X['Covid OR At/rp'] = (X['COVID-19 Indicator'] | X['Attorney/Representative'])
        
        # ===== Ratios =====
        X['Wage Age Ratio'] = X['Average Weekly Wage'] / X['Age at Injury']

        # ===== Extreme outlier flags =====
        # Use quantiles computed during fit
        X['High Wage Flag'] = (X['Average Weekly Wage'] > self.wage_95th)
        X['High count IME-4 Flag'] = (X['IME-4 Count'] > self.ime4_95th)

        # ===== Drop original date columns =====
        cols_to_drop = ['C-2 Date', 'C-3 Date', 'First Hearing Date', 'Accident Date', 'Assembly Date']
        for c in cols_to_drop:
            if c in X.columns:
                X = X.drop(c, axis=1)

        return X
 
# High Cardinality Transformer
class HighCardinalityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.valid_body_codes = set()
        self.valid_injury_codes = set()
        self.valid_cause_codes = set()

    def fit(self, X, y=None):
        # Determine which codes exist in the train data
        self.valid_body_codes = set((X['WCIO Part Of Body Code'].astype('float64').unique()))
        self.valid_injury_codes = set(X['WCIO Nature of Injury Code'].astype('float64').unique())
        self.valid_cause_codes = set(X['WCIO Cause of Injury Code'].astype('float64').unique())

        # Select categories with higher than 1% frequency
        self.frequency_categories = {
            col: X[col].value_counts(normalize=True)[lambda freq: freq > 0.01].index
            for col in ["Industry Code", "Carrier Name", "County of Injury"]
        }
        return self

    def transform(self, X):
        X = X.copy()

        # Replace -9 with 999 to relabel it -> from "Multiple" to "Multiple Body Parts"
        X['WCIO Part Of Body Code'] = X['WCIO Part Of Body Code'].astype('float64')
        X['WCIO Nature of Injury Code'] = X['WCIO Nature of Injury Code'].astype('float64')
        X['WCIO Cause of Injury Code'] = X['WCIO Cause of Injury Code'].astype('float64')

        # Create "Part of Body Group" to lower cardinality of the feature 'WCIO Part Of Body Code'
        def assign_body_group(code):
            if 10 <= code <= 19:
                return "Head"
            elif 20 <= code <= 26:
                return "Neck"
            elif 30 <= code <= 39:
                return "Upper Extremities"
            elif (40 <= code <= 49) or (60 <= code <= 63):
                return "Trunk"
            elif 50 <= code <= 58:
                return "Lower Extremities"
            elif (64 <= code <= 66) or (90 <= code <= 91) or (code == 99) or (code == 999):
                return "Multiple Body Parts"

        X['Part of Body Group'] = X['WCIO Part Of Body Code'].apply(
            lambda code: assign_body_group(code) if code in self.valid_body_codes else "Other"
        )

        # Create "Nature of Injury Group"
        def assign_nature_injury_group(code):
            if 1 <= code <= 59:
                return "Specific Injury"
            elif 60 <= code <= 83:
                return "Occupational Disease or Cumulative Injury"
            elif code >= 90:
                return "Multiple Injuries"

        X['Nature of Injury Group'] = X['WCIO Nature of Injury Code'].apply(
            lambda code: assign_nature_injury_group(code) if code in self.valid_injury_codes else "Other"
        )

        # Create "Cause of Injury Group"
        def assign_cause_injury_group(code):
            if (1 <= code <= 9) or (code == 11) or (code == 14) or (code == 84):
                return "I"  # Burn or Scald – Heat or Cold Exposures– Contact With
            elif (code == 10) or (12 <= code <= 13) or (code == 20):
                return "II"  # Caught In, Under or Between
            elif 15 <= code <= 19:
                return "III"  # Cut, Puncture, Scrape Injured By
            elif 25 <= code <= 33:
                return "IV"  # Fall, Slip or Trip Injury
            elif 40 <= code <= 50:
                return "V"  # Motor Vehicle
            elif (52 <= code <= 61) or (code == 97):
                return "VI"  # Strain or Injury By
            elif 65 <= code <= 70:
                return "VII"  # Striking Against or Stepping On
            elif (74 <= code <= 81) or (85 <= code <= 86):
                return "VIII"  # Struck or Injured By
            elif (code == 94) or (code == 95):
                return "IX"  # Rubbed or Abraded By
            else:
                return "X"  # Miscellaneous Causes

        X['Cause of Injury Group'] = X['WCIO Cause of Injury Code'].apply(
            lambda code: assign_cause_injury_group(code) if code in self.valid_cause_codes else "Other"
        )

        # Label categories with less than 1% frequency (very rare) or not present in train data as "Other"
        for col, freq in self.frequency_categories.items():
            X[col] = X[col].apply(lambda x: x if x in freq else "Other")

        # Drop original code columns
        X = X.drop(['WCIO Part Of Body Code', 
                    'WCIO Nature of Injury Code', 
                    'WCIO Cause of Injury Code'], axis=1)

        return X
    
# Encoding Transformer
class EncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # <=10 categories
        self.low_cardinality_cols = ["Carrier Type", 
                                     "District Name", 
                                     "Medical Fee Region",
                                     "Nature of Injury Group", 
                                     "Part of Body Group", 
                                     "Cause of Injury Group"]
        
        # >10 categories
        self.high_cardinality_cols = ["Carrier Name", 
                                      "County of Injury",
                                      "Industry Code"]

        self.freq_encoder = None
        self.onehot_encoder = None

    def fit(self, X, y=None):
        X = X.copy()

        self.freq_encoder = {col: X[col].value_counts().to_dict() for col in self.high_cardinality_cols}

        # Fit OneHotEncoder on low-cardinality features
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.onehot_encoder.fit(X[self.low_cardinality_cols])

        return self
    
    def transform(self, X):
        X = X.copy()

        # Encode boolean features as int
        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)

        # Tranform high cardinality features using frequency encoder
        for col, mapping in self.freq_encoder.items():
            X[col] = X[col].map(mapping).fillna(0) # Fill nan w/ 0 (unseen categories)

        # One-hot to encode low cardinality features
        onehot_encoded = pd.DataFrame(
            self.onehot_encoder.transform(X[self.low_cardinality_cols]),
            columns=self.onehot_encoder.get_feature_names_out(self.low_cardinality_cols),
            index=X.index
        )
        
        # Join features encoded with one hot encoder in X
        X = X.drop(self.low_cardinality_cols, axis=1).join(onehot_encoded)
        
        return X

# Tranformer for scaling features
class ScalingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else MinMaxScaler()

    def fit(self, X, y=None):
        # Fit the scaler
        self.scaler.fit(X)
        # Get feature names from train columns
        self.feature_names = X.columns
        return self

    def transform(self, X):
        # Transform the data and retain column names and index
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index
        )

    def get_params(self, deep=True):
        # Make parameters are retrievable for cloning
        return {"scaler": self.scaler}

    def set_params(self, **params):
        # Dynamic setting of parameters
        self.scaler = params.get("scaler", MinMaxScaler())
        return self
    
# Feature selection tranformer
class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,levene_threshold=0.05, cramers_v_threshold=0.02, lasso_alpha=0.01, tree_threshold=0.01, xgb_threshold=0.01):
        self.levene_threshold = levene_threshold
        self.cramers_v_threshold = cramers_v_threshold
        self.lasso_alpha = lasso_alpha
        self.tree_threshold = tree_threshold
        self.xgb_threshold = xgb_threshold
        self.selected_features = []

    def fit(self, X, y):
        X_copy = X.copy()

        metric_features=['Age at Injury', 'Average Weekly Wage', 'Carrier Name',
       'County of Injury', 'IME-4 Count',
       'Industry Code', 'Number of Dependents',
       'Accident Year', 'Assembly Year',
       'Average Weekly Wage Log', 'IME-4 Count Log',
       'Time Accident to Assembly', 'Time Assembly to Hearing',
       'Wage Age Ratio']
        
        X_cat = X_copy.drop(columns = metric_features)
        X_num = X_copy[metric_features]
        X_cat = X_cat.astype('int64')

        # 1. Remove features based on redundancy and irrelevance - filter methods

        # Levene's Test Function
        def levene_test(x, y):
            results = {}
            for column in x.columns:
                y_train_encoded_series = pd.Series(y,index= x.index)
                groups = [x[column][y_train_encoded_series == category] 
                        for category in y_train_encoded_series.unique()]
                stat, p_value = levene(*groups)
                results[column] = (stat, p_value)              
            return results

        # Perform Levene's Test for each feature in X_num
        levene_results = levene_test(X_num, y)

        # Remove irrelevant features with p-value > 0.05
        insignificant_features =[col for col, (_, p_value) in levene_results.items() 
                          if p_value > self.levene_threshold]
 
        # Previously selected numeric features redundant or with very low variance
        features_to_drop = [ "Average Weekly Wage", "IME-4 Count", "Accident Year", 
        "Wage Age Ratio"]
        for col in insignificant_features:
            if col not in features_to_drop:
                features_to_drop.append(col)

        
        # Cramér's V Function
        def cramers_v(x, y):
            contingency = pd.crosstab(x, y).values
            chi2 = chi2_contingency(contingency, correction=False)[0]
            n = contingency.sum()
            r, k = contingency.shape
            return np.sqrt(chi2 / (n * (min(r, k) - 1)))    

        # Check correlation of redundant categoric features and target
        redundant_pairs = [('C-3 Delivered', 'C-3 Delivered on Time'),
                           ('C-2 Delivered', 'C-2 Delivered on Time'),
                           ('COVID-19 Indicator', 'Covid OR altr dispute'),
                           ('Attorney/Representative', 'At/rp OR altr dispute'),
                           ('Attorney/Representative', 'Covid OR At/rp'),
                           ('Nature of Injury Group_Occupational Disease or Cumulative Injury',
                           'Nature of Injury Group_Specific Injury')]

        # Remove redundant features based on correlation w/ target
        for pair in redundant_pairs:
            feature_1, feature_2 = pair
            v1 = cramers_v(X_copy[feature_1], y)
            v2 = cramers_v(X_copy[feature_2], y)
            if v1 < v2:
                features_to_drop.append(feature_1)
            else:
                features_to_drop.append(feature_2)

        X_copy.drop(columns=features_to_drop, inplace=True)

        # Remove irrelevant categoric features with a cramers v below 0.02
        low_v_features = []
        for col in X_cat.columns:
            if col in X_copy.columns: 
                v = cramers_v(X_copy[col], y)
                if v < self.cramers_v_threshold:
                    low_v_features.append(col)
        X_copy.drop(columns=low_v_features, inplace=True)
        
        # 2. Lasso Ranking
        lasso = Lasso(alpha=self.lasso_alpha)
        lasso.fit(X_copy, y)
        lasso_coef = pd.Series(lasso.coef_, index=X_copy.columns)
        lasso_rank = lasso_coef[abs(lasso_coef) > 0].abs().rank(ascending=False).reset_index()
        lasso_rank.columns = ['Feature', 'Rank_lasso']

        # 3. Decision Tree Ranking
        tree = DecisionTreeClassifier(criterion='entropy')
        tree.fit(X_copy, y)
        tree_importance = pd.Series(tree.feature_importances_, index=X_copy.columns)
        tree_rank = tree_importance[tree_importance > self.tree_threshold].rank(ascending=False).reset_index()
        tree_rank.columns = ['Feature', 'Rank_decision_tree']

        # 4. XGBoost Ranking
        xgb = XGBClassifier()
        xgb.fit(X_copy, y)
        xgb_importance = pd.Series(xgb.feature_importances_, index=X_copy.columns)
        xgb_rank = xgb_importance[xgb_importance > self.xgb_threshold].rank(ascending=False).reset_index()
        xgb_rank.columns = ['Feature', 'Rank_xgboost']

        # 5. Combine rankings
        combined_rank = (lasso_rank
                         .merge(tree_rank, on='Feature', how='outer')
                         .merge(xgb_rank, on='Feature', how='outer'))

        combined_rank['Ranking_Count'] = combined_rank[['Rank_lasso', 'Rank_decision_tree', 'Rank_xgboost']].notnull().sum(axis=1)
        combined_rank = combined_rank[combined_rank['Ranking_Count'] >= 2]
        combined_rank['Total_Rank'] = combined_rank[['Rank_lasso', 'Rank_decision_tree', 'Rank_xgboost']].replace(0, np.nan).mean(axis=1)

        # 6. Select final features
        self.selected_features = combined_rank.sort_values(by='Total_Rank', ascending=True)['Feature'].tolist()

        return self 

    def transform(self, X):
        return X[self.selected_features] 
