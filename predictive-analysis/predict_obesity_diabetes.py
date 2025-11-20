"""
Predictive Model for Obesity and Diabetes Rates (2030-2040)
Based on dietary consumption trends and historical data

This script uses machine learning models (Linear Regression, Random Forest, Polynomial Regression)
to predict future obesity and diabetes rates based on dietary consumption patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Define data paths
DATA_DIR = Path("../raw-data")

class HealthPredictor:
    """Class to predict obesity and diabetes rates"""
    
    def __init__(self):
        self.diabetes_df = None
        self.obesity_df = None
        self.diet_df = None
        self.merged_df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load all datasets"""
        print("Loading data...")
        
        # Load diabetes data
        self.diabetes_df = pd.read_csv(
            DATA_DIR / "Diabetes" / "NCD_RisC_Lancet_2016_DM_crude_countries (1).csv"
        )
        
        # Load obesity data
        self.obesity_df = pd.read_csv(
            DATA_DIR / "Obesity-Trends" / "FAOSTAT_data_en_11-5-2025.csv"
        )
        
        # Load diet data
        self.diet_df = pd.read_csv(
            DATA_DIR / "Diet-Compositions" / "Diet compositions by commodity categories - FAO (2017).csv"
        )
        
        print(f"✓ Loaded diabetes data: {self.diabetes_df.shape}")
        print(f"✓ Loaded obesity data: {self.obesity_df.shape}")
        print(f"✓ Loaded diet data: {self.diet_df.shape}")
        
    def prepare_data(self):
        """Prepare and merge datasets for modeling"""
        print("\nPreparing data for modeling...")
        
        # Average diabetes prevalence across sex for each country-year
        diabetes_avg = self.diabetes_df.groupby(['Country/Region/World', 'Year']).agg({
            'Crude diabetes prevalence': 'mean'
        }).reset_index()
        diabetes_avg.columns = ['Country', 'Year', 'Diabetes_Prevalence']
        diabetes_avg['Diabetes_Prevalence'] *= 100  # Convert to percentage
        
        # Clean obesity data
        obesity_clean = self.obesity_df[['Area', 'Year', 'Value']].copy()
        obesity_clean.columns = ['Country', 'Year', 'Obesity_Prevalence']
        
        # Clean diet data - get nutrient columns
        nutrient_cols = [col for col in self.diet_df.columns if 'FAO' in col]
        diet_clean = self.diet_df[['Entity', 'Year'] + nutrient_cols].copy()
        
        # Simplify nutrient column names
        diet_clean.columns = ['Country', 'Year'] + [
            col.split('(')[0].strip().replace(' ', '_') for col in nutrient_cols
        ]
        
        # Calculate total calories and percentages
        nutrient_simple = [col for col in diet_clean.columns if col not in ['Country', 'Year']]
        diet_clean['Total_Calories'] = diet_clean[nutrient_simple].sum(axis=1)
        
        # Merge all datasets
        print("Merging datasets...")
        merged = diabetes_avg.merge(obesity_clean, on=['Country', 'Year'], how='inner')
        merged = merged.merge(diet_clean, on=['Country', 'Year'], how='inner')
        
        # Remove rows with missing values
        merged = merged.dropna()
        
        print(f"✓ Merged dataset shape: {merged.shape}")
        print(f"✓ Countries: {merged['Country'].nunique()}")
        print(f"✓ Year range: {merged['Year'].min()} - {merged['Year'].max()}")
        
        self.merged_df = merged
        return merged
    
    def build_models(self):
        """Build and train multiple ML models"""
        print("\n" + "="*80)
        print("BUILDING PREDICTIVE MODELS")
        print("="*80)
        
        # Get feature columns (all dietary nutrients)
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['Country', 'Year', 'Diabetes_Prevalence', 'Obesity_Prevalence']]
        
        print(f"\nFeatures used: {len(feature_cols)}")
        print(f"Features: {feature_cols}")
        
        # Prepare data for each target
        X = self.merged_df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Add year as a feature for trend analysis
        X_with_year = np.column_stack([X_scaled, self.merged_df['Year'].values])
        
        targets = {
            'Diabetes': self.merged_df['Diabetes_Prevalence'].values,
            'Obesity': self.merged_df['Obesity_Prevalence'].values
        }
        
        # Train models for each target
        for target_name, y in targets.items():
            print(f"\n{'='*60}")
            print(f"Training models for {target_name}")
            print(f"{'='*60}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_with_year, y, test_size=0.2, random_state=42
            )
            
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            }
            
            best_model = None
            best_score = -np.inf
            
            for model_name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                print(f"\n{model_name}:")
                print(f"  Train R²: {train_r2:.4f}")
                print(f"  Test R²:  {test_r2:.4f}")
                print(f"  Test RMSE: {test_rmse:.4f}")
                print(f"  Test MAE:  {test_mae:.4f}")
                print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Track best model
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = (model_name, model)
            
            print(f"\n✓ Best model for {target_name}: {best_model[0]} (R² = {best_score:.4f})")
            self.models[target_name] = best_model[1]
    
    def predict_future(self, start_year=2030, end_year=2040):
        """Predict obesity and diabetes rates for 2030-2040"""
        print("\n" + "="*80)
        print(f"PREDICTING RATES FOR {start_year}-{end_year}")
        print("="*80)
        
        # Get recent data to calculate trends
        recent_years = sorted(self.merged_df['Year'].unique())[-10:]  # Use more years for trend
        print(f"\nUsing data from years: {recent_years} to extrapolate trends")
        
        predictions = {}
        
        # For each country, predict future values
        countries = self.merged_df['Country'].unique()
        future_years = list(range(start_year, end_year + 1))
        
        all_predictions = []
        
        feature_cols = [col for col in self.merged_df.columns 
                       if col not in ['Country', 'Year', 'Diabetes_Prevalence', 'Obesity_Prevalence']]
        
        for country in countries:
            country_data = self.merged_df[self.merged_df['Country'] == country].copy()
            
            if len(country_data) < 5:  # Need enough historical data
                continue
            
            # Get recent dietary trends
            recent_data = country_data[country_data['Year'].isin(recent_years)].sort_values('Year')
            
            if len(recent_data) < 3:
                continue
            
            # Calculate trends for each nutrient using linear regression
            from scipy import stats
            
            nutrient_trends = {}
            baseline_features = recent_data[feature_cols].iloc[-1].values  # Last known values
            
            for idx, col in enumerate(feature_cols):
                if len(recent_data) >= 3:
                    # Fit linear trend
                    slope, intercept, _, _, _ = stats.linregress(
                        recent_data['Year'].values, 
                        recent_data[col].values
                    )
                    nutrient_trends[col] = (slope, intercept)
                else:
                    # No trend, use mean
                    nutrient_trends[col] = (0, recent_data[col].mean())
            
            # Predict for each future year with trending features
            for year in future_years:
                # Project features forward based on trends
                projected_features = []
                for idx, col in enumerate(feature_cols):
                    slope, intercept = nutrient_trends[col]
                    # Project value, but don't let it go negative or grow unreasonably
                    projected_value = slope * year + intercept
                    projected_value = max(0, projected_value)  # No negative calories
                    if col == 'Total_Calories':
                        # Limit total calories to reasonable range
                        projected_value = min(projected_value, 4000)
                        projected_value = max(projected_value, 1000)
                    else:
                        # Limit individual nutrients
                        projected_value = min(projected_value, 3000)
                    projected_features.append(projected_value)
                
                projected_features = np.array(projected_features).reshape(1, -1)
                
                # Scale features and add year
                features_scaled = self.scaler.transform(projected_features)
                X_pred = np.column_stack([features_scaled, [[year]]])
                
                # Predict using both models
                diabetes_pred = self.models['Diabetes'].predict(X_pred)[0]
                obesity_pred = self.models['Obesity'].predict(X_pred)[0]
                
                all_predictions.append({
                    'Country': country,
                    'Year': year,
                    'Predicted_Diabetes': max(0, min(100, diabetes_pred)),  # Between 0-100%
                    'Predicted_Obesity': max(0, min(100, obesity_pred))
                })
        
        predictions_df = pd.DataFrame(all_predictions)
        
        print(f"\n✓ Generated predictions for {len(countries)} countries")
        print(f"✓ Prediction years: {start_year} - {end_year}")
        
        # Save predictions
        predictions_df.to_csv('predictions_2030_2040.csv', index=False)
        print(f"\n✓ Saved predictions to 'predictions_2030_2040.csv'")
        
        return predictions_df
    
    def visualize_predictions(self, predictions_df):
        """Visualize predictions for selected countries"""
        print("\nCreating visualizations...")
        
        # Get countries that exist in both historical and prediction data
        hist_countries = set(self.merged_df['Country'].unique())
        pred_countries = set(predictions_df['Country'].unique())
        common_countries = hist_countries & pred_countries
        
        print(f"\nCountries in historical data: {len(hist_countries)}")
        print(f"Countries in predictions: {len(pred_countries)}")
        print(f"Common countries: {len(common_countries)}")
        
        # Select diverse sample countries
        sample_countries = ['United States', 'India', 'China', 'United Kingdom', 'Brazil', 
                          'Japan', 'Germany', 'France', 'Australia', 'Canada',
                          'Mexico', 'Indonesia', 'Nigeria', 'South Africa', 'Egypt']
        
        # Filter for countries in both datasets
        available_countries = [c for c in sample_countries if c in common_countries]
        
        # If none of the preferred countries are available, take top 10 by population/data availability
        if len(available_countries) == 0:
            available_countries = sorted(list(common_countries))[:10]
            print(f"\nUsing first 10 available countries: {available_countries}")
        else:
            print(f"\nSelected countries for visualization: {available_countries}")
        
        if len(available_countries) == 0:
            print("ERROR: No common countries found for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Historical and predicted diabetes
        for country in available_countries:
            # Historical data
            hist_data = self.merged_df[self.merged_df['Country'] == country]
            axes[0, 0].plot(hist_data['Year'], hist_data['Diabetes_Prevalence'], 
                          'o-', label=country, alpha=0.7, linewidth=2)
            
            # Predicted data
            pred_data = predictions_df[predictions_df['Country'] == country]
            axes[0, 0].plot(pred_data['Year'], pred_data['Predicted_Diabetes'], 
                          '--', linewidth=2, alpha=0.7)
        
        axes[0, 0].set_xlabel('Year', fontsize=12)
        axes[0, 0].set_ylabel('Diabetes Prevalence (%)', fontsize=12)
        axes[0, 0].set_title('Diabetes Prevalence: Historical (solid) vs Predicted (dashed)', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=2023, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        # Plot 2: Historical and predicted obesity
        for country in available_countries:
            hist_data = self.merged_df[self.merged_df['Country'] == country]
            axes[0, 1].plot(hist_data['Year'], hist_data['Obesity_Prevalence'], 
                          'o-', label=country, alpha=0.7, linewidth=2)
            
            pred_data = predictions_df[predictions_df['Country'] == country]
            axes[0, 1].plot(pred_data['Year'], pred_data['Predicted_Obesity'], 
                          '--', linewidth=2, alpha=0.7)
        
        axes[0, 1].set_xlabel('Year', fontsize=12)
        axes[0, 1].set_ylabel('Obesity Prevalence (%)', fontsize=12)
        axes[0, 1].set_title('Obesity Prevalence: Historical (solid) vs Predicted (dashed)', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=2023, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        # Plot 3: Average predicted rates across all countries
        avg_predictions = predictions_df.groupby('Year').agg({
            'Predicted_Diabetes': 'mean',
            'Predicted_Obesity': 'mean'
        }).reset_index()
        
        avg_historical = self.merged_df.groupby('Year').agg({
            'Diabetes_Prevalence': 'mean',
            'Obesity_Prevalence': 'mean'
        }).reset_index()
        
        axes[1, 0].plot(avg_historical['Year'], avg_historical['Diabetes_Prevalence'], 
                       'o-', label='Historical Diabetes', color='blue', linewidth=2, markersize=6)
        axes[1, 0].plot(avg_predictions['Year'], avg_predictions['Predicted_Diabetes'], 
                       's--', label='Predicted Diabetes', color='blue', linewidth=2, markersize=6)
        axes[1, 0].plot(avg_historical['Year'], avg_historical['Obesity_Prevalence'], 
                       'o-', label='Historical Obesity', color='red', linewidth=2, markersize=6)
        axes[1, 0].plot(avg_predictions['Year'], avg_predictions['Predicted_Obesity'], 
                       's--', label='Predicted Obesity', color='red', linewidth=2, markersize=6)
        
        axes[1, 0].set_xlabel('Year', fontsize=12)
        axes[1, 0].set_ylabel('Prevalence (%)', fontsize=12)
        axes[1, 0].set_title('Global Average Prevalence Trends', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=2023, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        # Plot 4: Distribution of predicted rates for 2040
        pred_2040 = predictions_df[predictions_df['Year'] == 2040]
        
        axes[1, 1].hist(pred_2040['Predicted_Diabetes'], bins=20, alpha=0.6, 
                       label='Diabetes 2040', color='blue', edgecolor='black')
        axes[1, 1].hist(pred_2040['Predicted_Obesity'], bins=20, alpha=0.6, 
                       label='Obesity 2040', color='red', edgecolor='black')
        
        axes[1, 1].set_xlabel('Prevalence (%)', fontsize=12)
        axes[1, 1].set_ylabel('Number of Countries', fontsize=12)
        axes[1, 1].set_title('Distribution of Predicted Rates in 2040', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization to 'predictions_visualization.png'")
        
        # Don't block - just save the file
        plt.close()
        
        # Print summary statistics
        print("\n" + "="*80)
        print("PREDICTION SUMMARY STATISTICS")
        print("="*80)
        
        for year in [2030, 2035, 2040]:
            year_data = predictions_df[predictions_df['Year'] == year]
            print(f"\n{'='*60}")
            print(f"YEAR {year}")
            print(f"{'='*60}")
            print(f"  Diabetes Prevalence:")
            print(f"    Mean:   {year_data['Predicted_Diabetes'].mean():.2f}%")
            print(f"    Median: {year_data['Predicted_Diabetes'].median():.2f}%")
            print(f"    Range:  {year_data['Predicted_Diabetes'].min():.2f}% - {year_data['Predicted_Diabetes'].max():.2f}%")
            print(f"  Obesity Prevalence:")
            print(f"    Mean:   {year_data['Predicted_Obesity'].mean():.2f}%")
            print(f"    Median: {year_data['Predicted_Obesity'].median():.2f}%")
            print(f"    Range:  {year_data['Predicted_Obesity'].min():.2f}% - {year_data['Predicted_Obesity'].max():.2f}%")
            
        # Show top 10 countries with highest predictions for 2040
        print("\n" + "="*80)
        print("TOP 10 COUNTRIES WITH HIGHEST PREDICTED RATES IN 2040")
        print("="*80)
        
        year_2040 = predictions_df[predictions_df['Year'] == 2040].copy()
        
        print("\nHighest Diabetes Rates:")
        top_diabetes = year_2040.nlargest(10, 'Predicted_Diabetes')[['Country', 'Predicted_Diabetes']]
        for idx, (_, row) in enumerate(top_diabetes.iterrows(), 1):
            print(f"  {idx}. {row['Country']}: {row['Predicted_Diabetes']:.2f}%")
        
        print("\nHighest Obesity Rates:")
        top_obesity = year_2040.nlargest(10, 'Predicted_Obesity')[['Country', 'Predicted_Obesity']]
        for idx, (_, row) in enumerate(top_obesity.iterrows(), 1):
            print(f"  {idx}. {row['Country']}: {row['Predicted_Obesity']:.2f}%")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("PREDICTIVE MODELING FOR OBESITY AND DIABETES (2030-2040)")
    print("="*80)
    
    # Initialize predictor
    predictor = HealthPredictor()
    
    # Load data
    predictor.load_data()
    
    # Prepare merged dataset
    predictor.prepare_data()
    
    # Build and train models
    predictor.build_models()
    
    # Make predictions for 2030-2040
    predictions_df = predictor.predict_future(start_year=2030, end_year=2040)
    
    # Visualize results
    predictor.visualize_predictions(predictions_df)
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - predictions_2030_2040.csv: Detailed predictions for all countries")
    print("  - predictions_visualization.png: Visual analysis of predictions")

if __name__ == "__main__":
    main()
