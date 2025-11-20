import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("../raw-data")

class NutrientVisualizer:
    """Interactive visualization of per capita nutrient consumption"""
    
    def __init__(self):
        self.diet_df = None
        self.gdp_df = None
        self.population_df = None
        self.per_capita_df = None
        
    def load_data(self):
        """Load dietary and population data"""
        print("Loading data...")
        
        self.diet_df = pd.read_csv(
            DATA_DIR / "Diet-Compositions" / "Diet compositions by commodity categories - FAO (2017).csv"
        )
        
        print(f"Loaded diet composition data: {self.diet_df.shape}")
        print(f"Countries: {self.diet_df['Entity'].nunique()}")
        print(f"Years: {self.diet_df['Year'].min()} - {self.diet_df['Year'].max()}")
        
        nutrient_cols = [col for col in self.diet_df.columns if 'FAO' in col]
        print(f"  Nutrients: {len(nutrient_cols)}")
        
        return nutrient_cols
    
    def prepare_per_capita_data(self):
        print("\nPreparing per capita consumption data...")
        
        nutrient_cols = [col for col in self.diet_df.columns if 'FAO' in col]
        
        nutrient_mapping = {}
        clean_names = []
        for col in nutrient_cols:
            clean_name = col.split('(')[0].strip()
            nutrient_mapping[col] = clean_name
            clean_names.append(clean_name)
        
        per_capita = self.diet_df.copy()
        per_capita = per_capita.rename(columns=nutrient_mapping)
        
        id_vars = ['Entity', 'Year']
        value_vars = clean_names
        
        melted_df = per_capita.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Nutrient',
            value_name='Consumption_Per_Capita'
        )
        
        melted_df = melted_df.dropna()
        
        self.per_capita_df = melted_df
        
        print(f"✓ Per capita data shape: {melted_df.shape}")
        print(f"✓ Countries: {melted_df['Entity'].nunique()}")
        print(f"✓ Nutrients: {melted_df['Nutrient'].nunique()}")
        print(f"✓ Years: {melted_df['Year'].min()} - {melted_df['Year'].max()}")
        
        return melted_df
    
    def create_interactive_visualization(self):
        print("\nCreating interactive visualization...")
        
        countries = sorted(self.per_capita_df['Entity'].unique())
        nutrients = sorted(self.per_capita_df['Nutrient'].unique())
        
        print(f"Available countries: {len(countries)}")
        print(f"Available nutrients: {len(nutrients)}")
        
        fig = go.Figure()
        
        default_country = 'United States' if 'United States' in countries else countries[0]
        default_nutrient = nutrients[0]
        
        for country in countries:
            for nutrient in nutrients:
                data = self.per_capita_df[
                    (self.per_capita_df['Entity'] == country) & 
                    (self.per_capita_df['Nutrient'] == nutrient)
                ].sort_values('Year')
                
                visible = (country == default_country and nutrient == default_nutrient)
                
                fig.add_trace(
                    go.Bar(
                        x=data['Year'],
                        y=data['Consumption_Per_Capita'],
                        name=f"{country} - {nutrient}",
                        visible=visible,
                        marker=dict(
                            color=data['Consumption_Per_Capita'],
                            colorscale='Viridis',
                            showscale=visible,
                            colorbar=dict(
                                title="kcal/person/day",
                                x=1.02
                            )
                        ),
                        text=data['Consumption_Per_Capita'].round(1),
                        textposition='outside',
                        hovertemplate=(
                            '<b>Year:</b> %{x}<br>'
                            '<b>Consumption:</b> %{y:.1f} kcal/person/day<br>'
                            '<extra></extra>'
                        )
                    )
                )
        
        buttons_country = []
        buttons_nutrient = []
        
        for i, country in enumerate(countries):
            visibility = []
            for c in countries:
                for n in nutrients:
                    visibility.append(c == country and n == default_nutrient)
            
            buttons_country.append(
                dict(
                    label=country,
                    method='update',
                    args=[
                        {'visible': visibility},
                        {'title': f'Nutrient Consumption in {country} - {default_nutrient}'}
                    ]
                )
            )
        
        for i, nutrient in enumerate(nutrients):
            visibility = []
            for c in countries:
                for n in nutrients:
                    visibility.append(c == default_country and n == nutrient)
            
            buttons_nutrient.append(
                dict(
                    label=nutrient,
                    method='update',
                    args=[
                        {'visible': visibility},
                        {'title': f'Nutrient Consumption in {default_country} - {nutrient}'}
                    ]
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f'Nutrient Consumption in {default_country} - {default_nutrient}',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis=dict(
                title='Year',
                tickmode='linear',
                dtick=5,
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='Consumption (kilocalories per person per day)',
                gridcolor='lightgray',
                showgrid=True
            ),
            updatemenus=[
                dict(
                    buttons=buttons_country,
                    direction='down',
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    x=0.02,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                    bgcolor='lightblue',
                    bordercolor='navy',
                    font=dict(size=11)
                ),
                dict(
                    buttons=buttons_nutrient,
                    direction='down',
                    pad={'r': 10, 't': 10},
                    showactive=True,
                    x=0.35,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                    bgcolor='lightgreen',
                    bordercolor='darkgreen',
                    font=dict(size=11)
                ),
            ],
            annotations=[
                dict(
                    text='<b>Select Country:</b>',
                    x=0.02,
                    xref='paper',
                    y=1.20,
                    yref='paper',
                    align='left',
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    text='<b>Select Nutrient:</b>',
                    x=0.35,
                    xref='paper',
                    y=1.20,
                    yref='paper',
                    align='left',
                    showarrow=False,
                    font=dict(size=12)
                ),
            ],
            height=700,
            width=1400,
            template='plotly_white',
            hovermode='x unified',
            showlegend=False,
            margin=dict(t=150, l=80, r=120, b=80)
        )
        
        output_file = 'interactive_nutrient_consumption.html'
        fig.write_html(output_file)
        print(f"\n✓ Saved interactive visualization to '{output_file}'")
        print(f"  Open this file in a web browser to interact with the visualization")
        
        fig.show()
        
        return fig
    
    def create_multi_country_comparison(self, selected_countries=None, selected_nutrient='Cereals and Grains'):
        print("\nCreating multi-country comparison...")
        
        if selected_countries is None:
            selected_countries = ['United States', 'China', 'India', 'Brazil', 'United Kingdom']
        
        available = [c for c in selected_countries if c in self.per_capita_df['Entity'].unique()]
        
        if not available:
            print("None of the selected countries are available in the data")
            return
        
        print(f"Comparing: {', '.join(available)}")
        print(f"Nutrient: {selected_nutrient}")
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, country in enumerate(available):
            data = self.per_capita_df[
                (self.per_capita_df['Entity'] == country) & 
                (self.per_capita_df['Nutrient'] == selected_nutrient)
            ].sort_values('Year')
            
            fig.add_trace(
                go.Scatter(
                    x=data['Year'],
                    y=data['Consumption_Per_Capita'],
                    mode='lines+markers',
                    name=country,
                    line=dict(width=3, color=colors[idx % len(colors)]),
                    marker=dict(size=8),
                    hovertemplate=(
                        f'<b>{country}</b><br>'
                        'Year: %{x}<br>'
                        'Consumption: %{y:.1f} kcal/person/day<br>'
                        '<extra></extra>'
                    )
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f'{selected_nutrient} Consumption Comparison',
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Arial Black')
            ),
            xaxis=dict(
                title='Year',
                tickmode='linear',
                dtick=5,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Consumption (kilocalories per person per day)',
                gridcolor='lightgray'
            ),
            height=600,
            width=1200,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )
        
        output_file = 'multi_country_comparison.html'
        fig.write_html(output_file)
        print(f"✓ Saved comparison to '{output_file}'")
        
        fig.show()
        
        return fig
    
    def create_summary_statistics(self):
        print("\nCreating summary statistics...")
        
        avg_consumption = self.per_capita_df.groupby(['Entity', 'Nutrient']).agg({
            'Consumption_Per_Capita': 'mean'
        }).reset_index()
        
        pivot_data = avg_consumption.pivot(
            index='Entity',
            columns='Nutrient',
            values='Consumption_Per_Capita'
        )
        
        top_countries = pivot_data.sum(axis=1).nlargest(30).index
        pivot_subset = pivot_data.loc[top_countries]
        
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_subset.values,
                x=pivot_subset.columns,
                y=pivot_subset.index,
                colorscale='YlOrRd',
                text=np.round(pivot_subset.values, 1),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="kcal/person/day"),
                hovertemplate=(
                    'Country: %{y}<br>'
                    'Nutrient: %{x}<br>'
                    'Avg Consumption: %{z:.1f}<br>'
                    '<extra></extra>'
                )
            )
        )
        
        fig.update_layout(
            title='Average Nutrient Consumption by Country (Top 30)',
            xaxis_title='Nutrient Category',
            yaxis_title='Country',
            height=900,
            width=1400,
            xaxis=dict(tickangle=45),
            template='plotly_white'
        )
        
        output_file = 'nutrient_consumption_heatmap.html'
        fig.write_html(output_file)
        print(f"✓ Saved heatmap to '{output_file}'")
        
        fig.show()
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        for nutrient in sorted(self.per_capita_df['Nutrient'].unique()):
            nutrient_data = self.per_capita_df[self.per_capita_df['Nutrient'] == nutrient]
            print(f"\n{nutrient}:")
            print(f"  Mean:   {nutrient_data['Consumption_Per_Capita'].mean():.2f} kcal/person/day")
            print(f"  Median: {nutrient_data['Consumption_Per_Capita'].median():.2f} kcal/person/day")
            print(f"  Std:    {nutrient_data['Consumption_Per_Capita'].std():.2f} kcal/person/day")
            print(f"  Range:  {nutrient_data['Consumption_Per_Capita'].min():.2f} - {nutrient_data['Consumption_Per_Capita'].max():.2f}")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("INTERACTIVE NUTRIENT CONSUMPTION VISUALIZATION")
    print("="*80)
    
    visualizer = NutrientVisualizer()
    
    visualizer.load_data()
    visualizer.prepare_per_capita_data()
    
    print("\n" + "="*60)
    print("Creating main interactive visualization...")
    print("="*60)
    visualizer.create_interactive_visualization()
    
    print("\n" + "="*60)
    print("Creating multi-country comparison...")
    print("="*60)
    visualizer.create_multi_country_comparison(
        selected_countries=['United States', 'China', 'India', 'Brazil', 'United Kingdom', 
                          'Germany', 'Japan', 'France', 'Australia', 'Canada'],
        selected_nutrient='Cereals and Grains'
    )
    
    print("\n" + "="*60)
    print("Creating summary heatmap...")
    print("="*60)
    visualizer.create_summary_statistics()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. interactive_nutrient_consumption.html")
    print("     - Main interactive visualization with country and nutrient dropdowns")
    print("  2. multi_country_comparison.html")
    print("     - Comparison of multiple countries for a specific nutrient")
    print("  3. nutrient_consumption_heatmap.html")
    print("     - Heatmap showing average consumption across countries and nutrients")
    print("\nOpen these HTML files in a web browser to interact with the visualizations!")

if __name__ == "__main__":
    main()