import matplotlib
matplotlib.use("Agg")  # Disable Tkinter GUI issues
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 🔹 Load the dataset correctly
df = pd.read_excel("road_accidents.xls", engine="xlrd", header=None)

# 🔹 Set column names manually
df.columns = ['Name of City', 'Total number of Fatal Accidents - 2011', 'All Accidents - 2011',
              'Persons Killed - 2011', 'Persons Injured - 2011', 'Total number of Fatal Accidents - 2012',
              'All Accidents - 2012', 'Persons Killed - 2012', 'Persons Injured - 2012', 'Fatal Accidents - 2013',
              'Total Accidents - 2013', 'Killed - 2013', 'Injured - 2013', 'Severity - 2013',
              'Fatal Accidents - 2014', 'Greviously Injured Accidents - 2014', 'Minor Accidents - 2014',
              'Non-Injurey Accidents - 2014', 'Total Accidents - 2014', 'Killed - 2014', 'Injured - 2014',
              'Severity - 2014', 'Fatal Accidents - 2015', 'Greviously Injured Accidents - 2015',
              'Minor Accidents - 2015', 'Non-Injurey Accidents - 2015', 'Total Accidents - 2015',
              'Killed - 2015', 'Injured - 2015', 'Severity - 2015']

# 🔹 Fix missing city names
df["Name of City"] = df["Name of City"].fillna("Unknown City")

# 🔹 Plot Accident Trends Over Years
plt.figure(figsize=(12, 6))
for year in range(2011, 2016):
    column_name = f"Total Accidents - {year}"
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")  # Convert non-numeric values
        df[column_name] = df[column_name].fillna(0)  # ✅ Proper assignment instead of inplace=True

        sns.histplot(df[column_name], bins=20, kde=True, label=str(year))

plt.title("📊 Total Accidents Distribution Across Cities (2011-2015)")
plt.xlabel("Number of Accidents")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("accident_trends.png")  # ✅ Saves the graph
print("\n✅ Accident trends saved as 'accident_trends.png'.")

# 🔹 Boxplot of Fatality Severity Levels (Fixed TypeError)
plt.figure(figsize=(10, 5))
df["Severity - 2015"] = pd.to_numeric(df["Severity - 2015"], errors="coerce")  # Convert severity values
df["Severity - 2015"] = df["Severity - 2015"].fillna(0)  # ✅ Proper assignment instead of inplace=True
sns.boxplot(y=df["Severity - 2015"])  # ✅ Set 'y' to avoid categorical issues
plt.title("🚦 Severity Distribution (2015)")
plt.savefig("severity_distribution.png")  # ✅ Saves the boxplot
print("\n✅ Severity distribution saved as 'severity_distribution.png'.")

# 🔹 Identify High-Severity Cities (Now Uses 25th Percentile for Filtering)
threshold = df["Severity - 2015"].quantile(0.25)  # Uses the 25th percentile
severe_cities = df[df["Severity - 2015"] > threshold]

# 🔹 Ensure city names are correctly displayed
severe_cities = severe_cities.loc[:, ["Name of City", "Severity - 2015"]].dropna()
print("\n🚨 High-Severity Cities (2015):")
print(severe_cities)
