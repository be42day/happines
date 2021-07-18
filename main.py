#!/usr/bin/env python
# coding: utf-8

# Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Functions
def plot_corrolation(y_label):
    '''
    Plot correlation between 'Life Ladder' and 'y_label'
    '''
    plt.figure(figsize=(20, 8))
    g1=sns.regplot(x='Life Ladder', y=y_label, data=target_country_rep, ci=None)
    g1.set(xlabel='Life Ladder')
    g1.set(ylabel=y_label)
    plt.title(f'Correlation between "Life Ladder" & "{y_label}"', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(report_file, format='pdf')
    plt.close()
    
def plot_yearly(factor):
    '''
    Plot lineplot 'factor' over the years
    '''
    plt.figure(figsize=(20, 8))
    sns.lineplot(x='year', y=factor, data=target_country_rep, marker='o', markersize=10);
    sns.set_style('whitegrid')
    sns.despine(left=True)
    plt.title(f'{country}\'s "{factor}" over the years', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('year')
    plt.ylabel(factor)
    plt.savefig(report_file, format='pdf')
    plt.close()



# Read the data in 2021
df_2021 = pd.read_csv('./Datasets/world-happiness-report-2021.csv',
            usecols=['Country name','Regional indicator','Ladder score',
                    'Logged GDP per capita','Social support','Healthy life expectancy',
                    'Freedom to make life choices', 'Generosity','Perceptions of corruption']
                     )


# Add the Year feature to its columns
df_2021.insert(1, column='year', value=[2021 for i in range(df_2021.shape[0])])


# Rename the columns to have identical columns name
df_2021.rename(columns={"Ladder score":"Life Ladder", "Logged GDP per capita":"Log GDP per capita",
                       "Healthy life expectancy":"Healthy life expectancy at birth"},inplace=True)


# Get the target country
while True:
    country = input('Enter the country name = ').title()
    if country in list(df_2021['Country name']):
        break
    else:
        print("The Country Name That You Entered Is Incorrect !")

# To create a PDF report in ./reports folder
report_file = PdfPages(f'./reports/{country}.pdf')

# Read the data before 2021
df_under_2021 = pd.read_csv('./Datasets/world-happiness-report.csv')

# Add the Region feature to its columns
df_under_2021.insert(1, column='Regional indicator', value=[np.nan for i in range(df_under_2021.shape[0])])

# Create the combination of the two previous data-frame
total_df = pd.concat([df_under_2021, df_2021], ignore_index=True, sort=False)

# Sort the data-frame by name of the country and the year,
# decreasingly (to fill the Region column by the previous one)
total_df.sort_values(by=['Country name', 'year'], inplace=True, ignore_index=True, ascending=False)

# Fill in the blank in the Region column by the related one
total_df['Regional indicator'].fillna(method='ffill', inplace=True)

# Sort the data-frame by name of the country and the year, increasingly
total_df.sort_values(by=['Country name', 'year'], inplace=True, ignore_index=True)

#target_country_rep = total_df[total_df.country_name == country]
target_country_rep = total_df[total_df['Country name'] == country]

# The related Region
region = list(target_country_rep['Regional indicator'])[0]

# Correlation matrix
columns = target_country_rep[['Regional indicator', 'Life Ladder','Log GDP per capita',
                              'Social support', 'Healthy life expectancy at birth',
                               'Freedom to make life choices', 'Generosity',
                               'Perceptions of corruption', 'Positive affect',
                              'Negative affect']]
plt.figure(figsize=(20, 8))
sns.heatmap(columns.corr(), annot = True, cmap='RdYlGn_r', mask=np.triu(np.ones_like(columns.corr())))
plt.title('Correlations between factors', fontsize=20, fontweight='bold', pad=20)
plt.xticks(rotation=45)
plt.savefig(report_file, format='pdf')
plt.close()

# Plot the correlation between Life Ladder and other factors
happy_factors = columns.corr()['Life Ladder'].sort_values().index
for i,factor in enumerate(happy_factors[:-1]):
    plot_corrolation(y_label=factor)

# Plot happines factors over the years  
for factor in happy_factors:
    plot_yearly(factor)

# Plot ranking (among the countries with the same region and
# also the best and the worst countries in the world) 
happiness_factor = df_2021.columns[3:]
for h_factor in happiness_factor:
    factor_df = df_2021.sort_values(h_factor, ascending=False).reset_index(drop=True)
    
    first_country = factor_df[['Country name',h_factor]].iloc[[0]]
    last_country = factor_df[['Country name',h_factor]].iloc[[-1]]
    region_countries = factor_df[factor_df['Regional indicator']==region][['Country name',h_factor]]
    
    # Check if first_country or last_country also exist in region_countries
    if (first_country.iloc[0][0] == region_countries.iloc[0][0]) and                         (last_country.iloc[0][0] == region_countries.iloc[-1][0]):
        ranking_df = region_countries
    elif (first_country.iloc[0][0] == region_countries.iloc[0][0]):
        ranking_df = pd.concat([region_countries, last_country])
    elif (last_country.iloc[0][0] == region_countries.iloc[-1][0]):
        ranking_df = pd.concat([first_country, region_countries])
    else:
        ranking_df = pd.concat([first_country, region_countries, last_country])
    # Bar plor
    plt.figure(figsize=(20, 8))
    x_vals = np.array(ranking_df['Country name'])
    y_vals = np.array(ranking_df[h_factor])
    clrs = ['cyan' if (c_name != country) else 'red' for c_name in x_vals ]
    bar=sns.barplot(x=x_vals, y=y_vals, palette=clrs)
    for i in range(1, len(ranking_df)):
        bar.text(x=i, y=(list(ranking_df[h_factor])[i])/2, s=str(ranking_df.index[i]+1)+'th',
             fontdict=dict(color='white', fontsize=12, fontweight='bold', horizontalalignment='center'))
    bar.text(x=0, y=(ranking_df[h_factor][0])/2, s='1st',
             fontdict=dict(color='white', fontsize=12, fontweight='bold', horizontalalignment='center'))
    mean_score = df_2021[h_factor].mean()
    bar.text(x=len(ranking_df)-0.4, y = mean_score, s = 'Global Average: {:.2f}'.format(mean_score),
                fontdict = dict(color='white', backgroundcolor='grey', fontsize=10, fontweight='bold'))
    bar.axhline(mean_score, color='grey', linestyle='--')
    plt.title(f'Where does {country}\'s "{h_factor}" Ranks in 2021?', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.savefig(report_file, format='pdf')
    plt.close()


report_file.close()
