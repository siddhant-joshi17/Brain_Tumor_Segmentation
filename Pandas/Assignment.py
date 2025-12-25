import pandas as pd
import matplotlib.pyplot as plt

# 1. Import the dataset
df = pd.read_csv('videogamesales.csv')

# 2. Add 'global_sales' column (sum of all regional sales)
# Usually NA_Sales, EU_Sales, JP_Sales, and Other_Sales
df['global_sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']

# 3. Sort (highest sales first) and print the DataFrame
df_sorted = df.sort_values(by='global_sales', ascending=False)
print("--- Sorted DataFrame by Global Sales ---")
print(df_sorted.head())

# 4. Display a plot of the total number of copies sold of each genre globally
genre_sales = df.groupby('Genre')['global_sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
genre_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Total Global Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Sales (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Filter 'Grand Theft Auto' games and display specific information
# Requirement: Name, Platform, Year, and Sum of EU + JP sales
gta_games = df[df['Name'].str.contains('Grand Theft Auto', case=False)].copy()
gta_games['EU_JP_Sales'] = gta_games['EU_Sales'] + gta_games['JP_Sales']

gta_filtered_df = gta_games[['Name', 'Platform', 'Year', 'EU_JP_Sales']]
print("\n--- Grand Theft Auto Games Information ---")
print(gta_filtered_df)

# 6. Display a pie chart of total sales of all GTA games combined by region
regions = ['North America', 'Europe', 'Japan', 'Other']
gta_regional_totals = [
    gta_games['NA_Sales'].sum(),
    gta_games['EU_Sales'].sum(),
    gta_games['JP_Sales'].sum(),
    gta_games['Other_Sales'].sum()
]

plt.figure(figsize=(8, 8))
plt.pie(gta_regional_totals, labels=regions, autopct='%1.1f%%', startangle=140)
plt.title('Grand Theft Auto Series: Global Sales Distribution by Region')
plt.axis('equal') 
plt.show()
