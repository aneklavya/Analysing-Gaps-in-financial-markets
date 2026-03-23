"""
MSME Finance Gap Predictive Modeling - Comprehensive Analysis
==============================================================
This script provides complete exploratory data analysis, visualization,
and predictive modeling for MSME Finance Gap data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MSMEFinanceGapAnalyzer:
    """
    Comprehensive analyzer for MSME Finance Gap data
    """
    
    def __init__(self, file_path):
        """Initialize and load data"""
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        # Load main database
        self.df = pd.read_excel(self.file_path, 
                               sheet_name='Main Database (2025 Report)', 
                               header=1)
        
        print(f"✓ Loaded {len(self.df)} countries")
        print(f"✓ {len(self.df.columns)} features available")
        
        return self.df
    
    def exploratory_analysis(self):
        """Comprehensive EDA with visualizations"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Create output directory for plots
        import os
        os.makedirs('/mnt/user-data/outputs', exist_ok=True)
        
        # 1. Data Overview
        print("\n1. DATA OVERVIEW")
        print("-" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()[self.df.isnull().sum() > 0]}")
        
        # 2. Target Variable Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MSME Finance Gap - Target Variable Analysis', fontsize=16, fontweight='bold')
        
        # Distribution of Finance Gap
        axes[0, 0].hist(self.df['MSME Finance Gap'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of MSME Finance Gap (Absolute)', fontweight='bold')
        axes[0, 0].set_xlabel('Finance Gap (USD)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of Finance Gap as % of GDP
        axes[0, 1].hist(self.df['MSME Finance Gap  (as % of GDP)'].dropna(), 
                       bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].set_title('Distribution of Finance Gap (% of GDP)', fontweight='bold')
        axes[0, 1].set_xlabel('Finance Gap (% of GDP)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot by Region
        region_data = self.df.groupby('Region')['MSME Finance Gap  (as % of GDP)'].apply(list)
        axes[1, 0].boxplot([x for x in region_data if len(x) > 0], 
                          labels=region_data.index[:len([x for x in region_data if len(x) > 0])])
        axes[1, 0].set_title('Finance Gap by Region', fontweight='bold')
        axes[1, 0].set_ylabel('Finance Gap (% of GDP)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot by Income Level
        income_data = self.df.groupby('Income')['MSME Finance Gap  (as % of GDP)'].apply(list)
        axes[1, 1].boxplot([x for x in income_data if len(x) > 0],
                          labels=income_data.index[:len([x for x in income_data if len(x) > 0])])
        axes[1, 1].set_title('Finance Gap by Income Level', fontweight='bold')
        axes[1, 1].set_ylabel('Finance Gap (% of GDP)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/01_target_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_target_analysis.png")
        plt.close()
        
        # 3. Gender Gap Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gender-Disaggregated Finance Gap Analysis', fontsize=16, fontweight='bold')
        
        # Women-led vs Men-led Gap
        valid_data = self.df[['Women-Led MSME Gap', 'Men-Led MSME Gap']].dropna()
        axes[0, 0].scatter(valid_data['Men-Led MSME Gap'], 
                          valid_data['Women-Led MSME Gap'], alpha=0.6)
        axes[0, 0].plot([valid_data.min().min(), valid_data.max().max()],
                       [valid_data.min().min(), valid_data.max().max()],
                       'r--', label='Equal Gap Line')
        axes[0, 0].set_title('Women-Led vs Men-Led MSME Gap', fontweight='bold')
        axes[0, 0].set_xlabel('Men-Led MSME Gap (USD)')
        axes[0, 0].set_ylabel('Women-Led MSME Gap (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Women-led gap as % of total
        axes[0, 1].hist(self.df['W-MSME Gap (as a % of overall MSME Gap)'].dropna(),
                       bins=25, edgecolor='black', alpha=0.7, color='purple')
        axes[0, 1].set_title('Women-Led Gap as % of Total MSME Gap', fontweight='bold')
        axes[0, 1].set_xlabel('Percentage')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Credit constraints by gender
        constraint_data = self.df[['W-MSME Fully Constrained %', 
                                   'M-MSME Fully Constrained %']].dropna()
        x = np.arange(len(constraint_data))
        width = 0.35
        axes[1, 0].bar(x - width/2, constraint_data['W-MSME Fully Constrained %'], 
                      width, label='Women-Led', alpha=0.8)
        axes[1, 0].bar(x + width/2, constraint_data['M-MSME Fully Constrained %'],
                      width, label='Men-Led', alpha=0.8)
        axes[1, 0].set_title('Fully Constrained MSMEs by Gender', fontweight='bold')
        axes[1, 0].set_ylabel('Percentage Fully Constrained')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # % MSME Women Led vs Finance Gap
        valid_data = self.df[['% MSME Women Led', 
                              'MSME Finance Gap  (as % of GDP)']].dropna()
        axes[1, 1].scatter(valid_data['% MSME Women Led'],
                          valid_data['MSME Finance Gap  (as % of GDP)'], alpha=0.6)
        axes[1, 1].set_title('% Women-Led MSMEs vs Finance Gap', fontweight='bold')
        axes[1, 1].set_xlabel('% MSMEs Women-Led')
        axes[1, 1].set_ylabel('Finance Gap (% of GDP)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/02_gender_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_gender_analysis.png")
        plt.close()
        
        # 4. Credit Constraint Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credit Constraint Analysis', fontsize=16, fontweight='bold')
        
        # Stacked bar chart of constraint levels
        constraint_cols = ['MSME Fully Constrained %', 'MSME Partly Constrained %', 
                          'MSME Unconstrained %']
        constraint_data = self.df[constraint_cols].dropna()
        
        # Top 15 countries by fully constrained %
        top_constrained = self.df.nlargest(15, 'MSME Fully Constrained %')
        axes[0, 0].barh(range(len(top_constrained)), 
                       top_constrained['MSME Fully Constrained %'])
        axes[0, 0].set_yticks(range(len(top_constrained)))
        axes[0, 0].set_yticklabels(top_constrained['Country'], fontsize=8)
        axes[0, 0].set_title('Top 15 Countries - Fully Constrained MSMEs', fontweight='bold')
        axes[0, 0].set_xlabel('Percentage Fully Constrained')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Constraint level vs Finance Gap
        axes[0, 1].scatter(self.df['MSME % Constrained'],
                          self.df['MSME Finance Gap  (as % of GDP)'], alpha=0.6)
        axes[0, 1].set_title('% Constrained vs Finance Gap', fontweight='bold')
        axes[0, 1].set_xlabel('% MSMEs Constrained')
        axes[0, 1].set_ylabel('Finance Gap (% of GDP)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average constraint levels by income
        income_constraints = self.df.groupby('Income')[constraint_cols].mean()
        income_constraints.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Credit Constraints by Income Level', fontweight='bold')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].legend(title='Constraint Level', bbox_to_anchor=(1.05, 1))
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Potential demand vs Current volume
        valid_data = self.df[['MSME Current Volume  (as a % of GDP)',
                              'MSME Potential Demand (as a % of GDP)']].dropna()
        axes[1, 1].scatter(valid_data['MSME Current Volume  (as a % of GDP)'],
                          valid_data['MSME Potential Demand (as a % of GDP)'], alpha=0.6)
        axes[1, 1].plot([0, valid_data.max().max()], [0, valid_data.max().max()],
                       'r--', label='Supply = Demand Line')
        axes[1, 1].set_title('Current Volume vs Potential Demand', fontweight='bold')
        axes[1, 1].set_xlabel('Current Volume (% of GDP)')
        axes[1, 1].set_ylabel('Potential Demand (% of GDP)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/03_constraint_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 03_constraint_analysis.png")
        plt.close()
        
        # 5. Correlation Analysis
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.df[numeric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix - All Numeric Features', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 04_correlation_matrix.png")
        plt.close()
        
        # 6. Informal Sector Analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Informal Sector Analysis', fontsize=16, fontweight='bold')
        
        # Informal demand distribution
        axes[0].hist(self.df['Informal Potential Demand (as a % of GDP)'].dropna(),
                    bins=25, edgecolor='black', alpha=0.7, color='orange')
        axes[0].set_title('Informal Potential Demand Distribution', fontweight='bold')
        axes[0].set_xlabel('Informal Demand (% of GDP)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Informal vs Formal demand
        valid_data = self.df[['Informal Potential Demand (as a % as formal Potential Demand)',
                              'MSME Finance Gap  (as % of GDP)']].dropna()
        axes[1].scatter(valid_data['Informal Potential Demand (as a % as formal Potential Demand)'],
                       valid_data['MSME Finance Gap  (as % of GDP)'], alpha=0.6)
        axes[1].set_title('Informal Demand vs Finance Gap', fontweight='bold')
        axes[1].set_xlabel('Informal Demand (% of Formal)')
        axes[1].set_ylabel('Finance Gap (% of GDP)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/05_informal_sector.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 05_informal_sector.png")
        plt.close()
        
        print("\n✓ Exploratory Analysis Complete - 5 visualization files saved")
        
    def prepare_features(self, target_variable='MSME Finance Gap  (as % of GDP)'):
        """Prepare features for modeling"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # Define feature sets
        numeric_features = [
            'MSME Current Volume  (as a % of GDP)',
            'MSME Potential Demand (as a % of GDP)',
            'MSME Fully Constrained %',
            'MSME Partly Constrained %',
            'MSME % Constrained',
            '% MSME Women Led',
            'W-MSME Fully Constrained %',
            'M-MSME Fully Constrained %',
            'W-MSME Gap (as a % of overall MSME Gap)',
            'Informal Potential Demand (as a % of GDP)',
            'Informal Potential Demand (as a % as formal Potential Demand)'
        ]
        
        categorical_features = ['Region', 'Income']
        
        # Create working dataframe
        df_model = self.df.copy()
        
        # Handle missing values in target
        df_model = df_model[df_model[target_variable].notna()].copy()
        
        print(f"\n✓ Target Variable: {target_variable}")
        print(f"✓ Valid samples: {len(df_model)}")
        
        # Prepare X and y
        X = df_model[numeric_features + categorical_features].copy()
        y = df_model[target_variable].copy()
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Impute missing values in features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"\n✓ Features: {len(X_imputed.columns)}")
        print(f"  - Numeric: {len(numeric_features)}")
        print(f"  - Categorical: {len(categorical_features)}")
        
        # Feature importance preview
        print(f"\n✓ Feature correlation with target:")
        correlations = pd.DataFrame({
            'Feature': X_imputed.columns,
            'Correlation': [X_imputed[col].corr(y) for col in X_imputed.columns]
        }).sort_values('Correlation', key=abs, ascending=False)
        print(correlations.to_string(index=False))
        
        return X_imputed, y
    
    def train_models(self, X, y):
        """Train multiple regression models"""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.01),
            'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5)
        }
        
        # Train and evaluate each model
        results = []
        
        for name, model in models.items():
            print(f"\n{name}")
            print("-" * 50)
            
            # Use scaled data for linear models
            if name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net']:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                X_train_cv = X_train_scaled
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                X_train_cv = self.X_train
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_cv, self.y_train, 
                                       cv=5, scoring='r2')
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std()
            })
            
            self.models[name] = model
        
        # Create results dataframe
        self.results = pd.DataFrame(results)
        
        # Visualize model comparison
        self._plot_model_comparison()
        
        # Get best model
        best_model_name = self.results.loc[self.results['R2'].idxmax(), 'Model']
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"{'='*80}")
        
        return self.results
    
    def _plot_model_comparison(self):
        """Visualize model performance comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score
        axes[0].barh(self.results['Model'], self.results['R2'])
        axes[0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[0].set_xlabel('R² Score')
        axes[0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1].barh(self.results['Model'], self.results['RMSE'], color='coral')
        axes[1].set_title('RMSE (Lower is Better)', fontweight='bold')
        axes[1].set_xlabel('RMSE')
        axes[1].grid(True, alpha=0.3)
        
        # MAE
        axes[2].barh(self.results['Model'], self.results['MAE'], color='lightgreen')
        axes[2].set_title('MAE (Lower is Better)', fontweight='bold')
        axes[2].set_xlabel('MAE')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/06_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 06_model_comparison.png")
        plt.close()
    
    def feature_importance_analysis(self):
        """Analyze feature importance from tree-based models"""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get Random Forest model
        rf_model = self.models.get('Random Forest')
        
        if rf_model is None:
            print("⚠ Random Forest model not trained")
            return
        
        # Get feature importances
        importances = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importances.head(10).to_string(index=False))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(importances)), importances['Importance'])
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances['Feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/07_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07_feature_importance.png")
        plt.close()
        
        return importances
    
    def prediction_analysis(self):
        """Analyze predictions vs actual values"""
        print("\n" + "="*80)
        print("PREDICTION ANALYSIS")
        print("="*80)
        
        # Get best model
        best_model_name = self.results.loc[self.results['R2'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        # Make predictions
        if best_model_name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net']:
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            y_pred = best_model.predict(scaler.transform(self.X_test))
        else:
            y_pred = best_model.predict(self.X_test)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Prediction Analysis - {best_model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Actual vs Predicted
        axes[0].scatter(self.y_test, y_pred, alpha=0.6)
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Finance Gap (% of GDP)')
        axes[0].set_ylabel('Predicted Finance Gap (% of GDP)')
        axes[0].set_title('Actual vs Predicted Values', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Finance Gap (% of GDP)')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/08_predictions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 08_predictions.png")
        plt.close()
        
        return y_pred
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report = f"""
MSME FINANCE GAP - PREDICTIVE MODELING REPORT
{'='*80}

1. DATASET OVERVIEW
{'-'*80}
Total Countries: {len(self.df)}
Total Features: {len(self.df.columns)}
Target Variable: MSME Finance Gap (as % of GDP)

Missing Values:
{self.df.isnull().sum()[self.df.isnull().sum() > 0].to_string()}

2. KEY STATISTICS
{'-'*80}
Finance Gap (% of GDP):
  Mean: {self.df['MSME Finance Gap  (as % of GDP)'].mean():.4f}
  Median: {self.df['MSME Finance Gap  (as % of GDP)'].median():.4f}
  Std Dev: {self.df['MSME Finance Gap  (as % of GDP)'].std():.4f}
  Min: {self.df['MSME Finance Gap  (as % of GDP)'].min():.4f}
  Max: {self.df['MSME Finance Gap  (as % of GDP)'].max():.4f}

3. MODEL PERFORMANCE
{'-'*80}
{self.results.to_string(index=False)}

4. BEST MODEL
{'-'*80}
Model: {self.results.loc[self.results['R2'].idxmax(), 'Model']}
R² Score: {self.results['R2'].max():.4f}
RMSE: {self.results.loc[self.results['R2'].idxmax(), 'RMSE']:.4f}
MAE: {self.results.loc[self.results['R2'].idxmax(), 'MAE']:.4f}

5. RECOMMENDATIONS
{'-'*80}
Based on the analysis:

a) KEY PREDICTORS:
   - MSME Potential Demand (as % of GDP)
   - Credit Constraint Levels (Fully/Partly Constrained %)
   - Current Volume (as % of GDP)
   - Gender-disaggregated gaps
   - Informal sector demand

b) MODEL SELECTION:
   - Random Forest or Gradient Boosting recommended for best performance
   - These models handle non-linear relationships well
   - Feature importance analysis available

c) NEXT STEPS:
   1. Hyperparameter tuning for top models
   2. Feature engineering (interaction terms, regional dummies)
   3. Ensemble methods combining multiple models
   4. Time-series analysis if historical data available
   5. Incorporate external economic indicators (GDP growth, inflation, etc.)

{'='*80}
"""
        
        # Save report
        with open('/mnt/user-data/outputs/ANALYSIS_REPORT.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("\n✓ Report saved: ANALYSIS_REPORT.txt")
        
        return report


# Main execution
if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MSME FINANCE GAP - PREDICTIVE ANALYSIS" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Initialize analyzer
    file_path = '/mnt/user-data/uploads/MSME_Finance_Gap_Main_Data_Accompanying_2024_Report_Final_-_For_Website_20250422.xlsx'
    analyzer = MSMEFinanceGapAnalyzer(file_path)
    
    # Step 1: Load data
    df = analyzer.load_data()
    
    # Step 2: Exploratory analysis
    analyzer.exploratory_analysis()
    
    # Step 3: Prepare features
    X, y = analyzer.prepare_features()
    
    # Step 4: Train models
    results = analyzer.train_models(X, y)
    
    # Step 5: Feature importance
    importance = analyzer.feature_importance_analysis()
    
    # Step 6: Prediction analysis
    predictions = analyzer.prediction_analysis()
    
    # Step 7: Generate report
    report = analyzer.generate_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. 01_target_analysis.png - Target variable distributions")
    print("  2. 02_gender_analysis.png - Gender gap analysis")
    print("  3. 03_constraint_analysis.png - Credit constraints")
    print("  4. 04_correlation_matrix.png - Feature correlations")
    print("  5. 05_informal_sector.png - Informal sector analysis")
    print("  6. 06_model_comparison.png - Model performance")
    print("  7. 07_feature_importance.png - Important features")
    print("  8. 08_predictions.png - Prediction quality")
    print("  9. ANALYSIS_REPORT.txt - Comprehensive report")
    print("\n" + "="*80 + "\n")
