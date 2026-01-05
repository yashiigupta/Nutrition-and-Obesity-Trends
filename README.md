# Nutrition and Obesity Trends

## Project Overview

This study analyzes global nutrition and obesity trends to assess population health, evaluate dietary transitions, and design effective public health interventions. Over recent decades, changing lifestyles, evolving food environments, and the rise of ultra-processed foods have resulted in rapid increases in overweight and obesity rates across both developed and developing nations. This project explores long-term trends in caloric intake, macronutrient composition, and obesity prevalence using curated global datasets.

**Authors:** Yashi Gupta, Pranay Vishwakarma, Vishesh Rao  
**Department:** Computer Science, Rishihood University

---

## Problem Statement

The core problem addressed in this study is the rising global burden of obesity and diabetes and their strong connection to nutritional patterns. Countries vary significantly in their intake of nutrients such as sugar, fats, proteins, and fiber, leading to substantial differences in health outcomes. This study identifies the key nutritional, economic, and lifestyle factors contributing to these trends and uses both descriptive and predictive analysis to uncover patterns, evaluate risk factors, and support data-driven public health decision-making.

---

## Literature Review

This research is grounded in well-established global studies:
- **Health effects of dietary risks in 195 countries, 1990–2017:** A Systematic Analysis for the Global Burden of Disease Study 2017, The Lancet (2019)
- **Global Burden of Diabetes,** GBD Collaborators, The Lancet

These research papers provide deep insights into diet-related risks, global diabetes burden, and long-term epidemiological patterns, guiding our dataset selection, feature engineering, modeling choices, and interpretation of results.

---

## Key Research Questions Analyzed

### 1. Food Group Consumption Changes (1961-Present)
**Question:** Which food groups (processed foods, dairy, cereals, fruits/vegetables) have changed most in consumption over the past 50 years?

**Findings:** A clear "Nutrition Transition" is occurring globally. Since 1961, there has been a steady increase in energy-dense foods (vegetable oils, meat, sugar), while traditional staples (starchy roots, pulses) have stagnated or declined. This shift towards caloric, processed, and animal-based diets is a primary driver of the modern metabolic health crisis.

### 2. Regional Dietary Patterns
**Question:** Which countries consume the highest levels of sugar, fat, protein, and fiber? Are there regional patterns?

**Findings:** Western diets (North America, Europe, Oceania) show 2–4 times higher intake of sugar, fat, and protein compared to Asia and Africa. Asia and Africa lead in fiber consumption. While "Westernization" is spreading, regional agricultural and cultural practices still heavily influence consumption patterns.

### 3. Income and Nutritional Quality
**Question:** How does income (GDP per capita) affect nutritional quality and obesity rates?

**Findings:** Strong positive correlation between GDP per capita and obesity rates. Economic development increases access to ultra-processed, calorie-dense foods rather than nutrient-rich whole foods. Notable exceptions include Japan and France, which maintain lower obesity rates despite high GDP due to strong cultural food norms and effective public health policies.

### 4. Obesity-Diabetes-Diet Correlation
**Question:** How do obesity and diabetes rates correlate with national dietary consumption patterns?

**Findings:** Strong positive correlations between obesity/diabetes prevalence and consumption of sugar, meat, and oils/fats. Cereals, pulses, and fiber show negative or weak correlation, suggesting plant-based, high-fiber diets are protective.

### 5. Predictive Nutrient Analysis
**Question:** Which nutrients have the strongest predictive power for obesity and diabetes?

**Findings:**
- **Obesity Prediction:** Sugar and Animal Fats have the strongest predictive power with the largest positive standardized coefficients
- **Diabetes Prediction:** Sugar and Other Nutrients exhibit the strongest predictive power, particularly linked to insulin resistance

### 6. Future Projections (2030-2040)
**Question:** What are the projected obesity and diabetes rates for 2030–2040?

**Findings:** Based on current dietary trends, predictive models forecast a relentless rise in global obesity and diabetes rates through 2040. Without structural interventions, the burden of metabolic disease will place unsustainable strain on healthcare systems worldwide.

---

## Research Methodology

### Data Collection and Processing
- **Dataset Merging:** Combined obesity, diabetes, and nutrient datasets using country and year as primary keys
- **Preprocessing:** Handled missing values, corrected inconsistencies, standardized feature names, and aligned units
- **Sources:** Comprehensive datasets from reliable global sources (FAO, USDA, GBD studies)

### Analysis Techniques
- Summary statistics and correlation analysis
- Extensive exploratory data analysis (EDA) using static and interactive visualizations
- Pattern and anomaly detection
- Multi-decade time-series trend analysis
- Predictive modeling using linear regression with standardized coefficients
- Literature-based reasoning to maintain analytical rigor

---

## Key Findings Summary

1. **Nutrition Transition:** Global shift towards energy-dense foods (oils, meat, sugar) over the last 50 years
2. **Regional Disparities:** Western diets significantly higher in sugar, fat, and protein; Asian and African diets higher in fiber
3. **Economic Impact:** GDP strongly predicts obesity rates, though cultural factors can mitigate this effect
4. **Disease Correlation:** Sugar, meat, and oils/fats strongly correlate with obesity and diabetes prevalence
5. **Primary Risk Factors:** Sugar and animal fats are the strongest predictors of obesity; sugar and processed nutrients predict diabetes
6. **Alarming Projections:** Without intervention, obesity and diabetes rates will continue rising through 2040

---

## Policy Recommendations

### 1. Sugar-Sweetened Beverage (SSB) Taxes
Implement or increase taxes on sugary drinks. Revenue can subsidize healthy foods. In India, many beverages contain artificial flavourings and colours restricted elsewhere due to low consumer awareness and insufficient regulation.

### 2. Front-of-Package Labeling
Mandate clear warning labels (e.g., "High in Sugar") on processed foods. Japan's strict packaging standards serve as a model where external packaging closely replicates the actual food item.

### 3. Subsidies for Whole Foods
Shift agricultural subsidies from corn/soy to fruits, vegetables, and pulses. Support companies driving minimally processed foods:
- **Slurrp Farm:** Millet-based, child-friendly foods
- **Mille Super Grains:** Whole-grain and millet-based products
- **Conscious Foods:** Organic, minimally processed staples

### 4. School Nutrition Standards
Enforce strict guidelines for school meals and limit unhealthy food marketing to children. Follow examples from the US (Healthy, Hunger-Free Kids Act), Japan, and UK. Ensure vending machines prioritize fruits, nuts, and whole grains.

### 5. Public Awareness Campaigns
Launch national campaigns educating the public on links between ultra-processed foods and metabolic disease, with government funding for national visibility.

---

## Project Status

| Task ID | Task Description                                                    | Status      | Notes                                                                    |
|---------|---------------------------------------------------------------------|-------------|--------------------------------------------------------------------------|
| 1       | Data collection and cleaning from USDA and FAO datasets            | ✅ Complete | Aggregated per capita nutrient consumption yearly by country             |
| 2       | Exploratory data analysis and visualization                         | ✅ Complete | Created static and interactive visualizations                            |
| 3       | Trend analysis across decades                                       | ✅ Complete | Examined multi-decade time-series data                                   |
| 4       | Correlation analysis between diet and health outcomes              | ✅ Complete | Identified key relationships between nutrients and diseases              |
| 5       | Predictive modeling for obesity and diabetes                        | ✅ Complete | Linear regression with standardized coefficients                         |
| 6       | Future projections (2030-2040)                                     | ✅ Complete | Forecasted obesity and diabetes rates                                    |
| 7       | Policy recommendations development                                  | ✅ Complete | Evidence-based interventions identified                                  |
| 8       | Final report compilation                                           | ✅ Complete | Comprehensive analysis documented                                        |

---

## Datasets and Resources

- **USDA Nutrition and Obesity Data**  
  [https://www.ers.usda.gov/topics/food-choices-health/obesity](https://www.ers.usda.gov/topics/food-choices-health/obesity)

- **FAO Food and Agriculture Data**  
  [https://www.fao.org/faostat/en/#home](https://www.fao.org/faostat/en/#home)

- **Project Report (PDF)**  
  Access the complete Phase 3 report: [https://drive.google.com/file/d/1xWd7mWVQIyfK5_hQqa0JOCCyI7vQnJ6y/view?usp=sharing] (located in project directory)

---

## Conclusion

This study confirms that the global rise in obesity and diabetes results from specific, measurable shifts in diet—specifically increased consumption of sugar, meat, and processed fats. Economic growth has accelerated this transition. Our forecasts indicate that without decisive action, the burden of metabolic disease will continue to grow. However, by targeting key dietary drivers through robust policy interventions, it is possible to alter this trajectory and improve global health outcomes.

---

## Index Terms

nutrition, obesity, diabetes, descriptive analysis, predictive modeling, global dietary trends, regression, public health policy