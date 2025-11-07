"""
   Data Cleaning and Processing """

import re
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- Utilities ----------
DATA_PATH = Path("C:/Users/Abdullah Umer/Desktop/Internee.pk Internship/Task 5/Data_Salaries.csv")
OUT_CSV = Path("/mnt/data/cleaned_data_salaries.csv")
OUT_DIR = DATA_PATH.parent / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_salary_text(s):
    """
    Parse a salary text string and return (min_val, max_val, unit, note)
    - min_val/max_val numeric in original unit (e.g., hourly or annual)
    - unit: 'hour', 'year', or None
    Examples it can handle:
      "$20.00 - $30.00 Per Hour (Employer est.)"
      "$77K - $130K (Glassdoor est.)"
      "$100,000"
      "Employer Provided Salary:$40,000 - $60,000"
    """
    if pd.isna(s):
        return (np.nan, np.nan, None, None)
    txt = str(s).replace('\u00a0', ' ')  # clean non-breaking space
    txt = txt.replace(',', '')  # remove thousands separator
    txt_low = txt.lower()
    
    # detect unit
    unit = None
    if "per hour" in txt_low or "/hour" in txt_low or "hour" in txt_low and "per" in txt_low:
        unit = 'hour'
    if "per year" in txt_low or "per annum" in txt_low or "year" in txt_low or "yr" in txt_low or "annual" in txt_low or "k)" in txt_low:
        # note: don't override 'hour' detection if both words present; prefer hour when "per hour" matched explicitly
        if unit is None:
            unit = 'year'
    
    # find all numeric tokens like 21.50, 130K, 100000
    # extract tokens with optional K or M
    tokens = re.findall(r'\$?\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)', txt)
    nums = []
    for num, suffix in tokens:
        try:
            v = float(num)
            if suffix.lower() == 'k':
                v = v * 1000.0
            if suffix.lower() == 'm':
                v = v * 1_000_000.0
            nums.append(v)
        except:
            continue
    # If tokens found, pick min and max
    if len(nums) == 0:
        # try extracting single number patterns with commas removed
        single = re.findall(r'([0-9]{3,})', txt)
        if single:
            try:
                v = float(single[0])
                nums = [v]
            except:
                pass
    if len(nums) == 1:
        return (nums[0], nums[0], unit, txt)
    elif len(nums) >= 2:
        return (min(nums), max(nums), unit, txt)
    else:
        return (np.nan, np.nan, None, txt)

def convert_to_annual(min_val, max_val, unit):
    

    if math.isnan(min_val) and math.isnan(max_val):
        return np.nan
    vals = []
    if not math.isnan(min_val):
        vals.append(min_val)
    if not math.isnan(max_val):
        vals.append(max_val)
    mean_val = float(np.mean(vals))
    if unit == 'hour':
        return mean_val * 2080.0
    elif unit == 'year':
        return mean_val
    else:
        # guess: if value seems small (<5000) treat as hourly; if >5000 treat as annual
        if mean_val > 0 and mean_val < 5000:
            return mean_val * 2080.0
        elif mean_val >= 5000:
            return mean_val
        else:
            return np.nan


# ---------- Load ----------
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)


# ---------- Initial inspection ----------
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())


# ---------- Basic cleaning ----------
# Trim whitespace in string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip().replace({'nan': None})


# Standardize Company Score: ensure numeric
if 'Company Score' in df.columns:
    df['Company Score'] = pd.to_numeric(df['Company Score'], errors='coerce')


# Handle missing Company names: fill with 'Unknown Company'
df['Company'] = df['Company'].fillna('Unknown Company')


# Handle Location missing
df['Location'] = df['Location'].replace({'None': None})
df['Location'] = df['Location'].fillna('Not Specified')


# Drop exact duplicates (if any)
before_dups = df.shape[0]
df.drop_duplicates(inplace=True)
after_dups = df.shape[0]
print(f"\nDropped {before_dups - after_dups} exact duplicate rows.")


# ---------- Salary parsing ----------
salary_parsed = df['Salary'].apply(parse_salary_text)
df[['Salary_min_raw', 'Salary_max_raw', 'Salary_unit_raw', 'Salary_note']] = pd.DataFrame(salary_parsed.tolist(), index=df.index)


# Convert raw parsed numbers to floats; if blank string -> NaN
df['Salary_min_raw'] = pd.to_numeric(df['Salary_min_raw'], errors='coerce')
df['Salary_max_raw'] = pd.to_numeric(df['Salary_max_raw'], errors='coerce')


# Infer unit if missing using heuristics on the text
df['Salary_unit'] = df['Salary_unit_raw'].copy()
# if unit missing and values < 5000 assume hourly else annual
mask_no_unit = df['Salary_unit'].isna() & df['Salary_min_raw'].notna()
df.loc[mask_no_unit, 'Salary_unit'] = np.where(df.loc[mask_no_unit, 'Salary_min_raw'] < 5000, 'hour', 'year')


# Create Avg Salary in original unit
df['Salary_avg_raw'] = df[['Salary_min_raw', 'Salary_max_raw']].mean(axis=1)


# Create annual salary estimate
df['Salary_annual_est'] = df.apply(lambda r: convert_to_annual(r['Salary_min_raw'] if not pd.isna(r['Salary_min_raw']) else np.nan,
                                                              r['Salary_max_raw'] if not pd.isna(r['Salary_max_raw']) else np.nan,
                                                              r['Salary_unit']), axis=1)


# Clean some obvious anomalies: if Salary_annual_est extremely large (>5 million) set to NaN
df.loc[df['Salary_annual_est'] > 5_000_000, 'Salary_annual_est'] = np.nan

# ---------- Company Score missing handling ----------
if 'Company Score' in df.columns:
    median_score = df['Company Score'].median(skipna=True)
    df['Company Score'] = df['Company Score'].fillna(median_score)

# ---------- Job Title cleaning ----------
df['Job Title'] = df['Job Title'].astype(str).str.strip().replace({'nan': 'Unknown'})
# normalize case
df['Job Title_clean'] = df['Job Title'].str.title()




# ---------- Location splitting ----------
# If location contains comma (City, ST) try splitting
def split_location(loc):
    if pd.isna(loc) or loc == 'Not Specified':
        return (None, None, loc)
    loc = loc.strip()
    # common format: "City, ST" or "City, State"
    if ',' in loc:
        parts = [p.strip() for p in loc.split(',') if p.strip()!='']
        if len(parts) >= 2:
            city = parts[0]
            state = parts[1]
            return (city, state, loc)
    # common single tokens: "Remote", "United States"
    return (loc, None, loc)

loc_split = df['Location'].apply(split_location)
df[['Loc_City', 'Loc_State', 'Location_original']] = pd.DataFrame(loc_split.tolist(), index=df.index)

# Standardize company names lightly
df['Company_clean'] = df['Company'].str.replace(r'\s+', ' ', regex=True).str.title()

# ---------- Outlier handling for Salary_annual_est ----------
# We'll identify outliers using IQR and cap them at [Q1-1.5*IQR, Q3+1.5*IQR] for demonstration (winsorization)
sal = df['Salary_annual_est']
q1 = sal.quantile(0.25)
q3 = sal.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"\nSalary annual est IQR bounds: lower={lower_bound:.2f}, upper={upper_bound:.2f}")

# Create capped salary column
df['Salary_annual_capped'] = df['Salary_annual_est'].copy()
df.loc[df['Salary_annual_capped'] < lower_bound, 'Salary_annual_capped'] = lower_bound
df.loc[df['Salary_annual_capped'] > upper_bound, 'Salary_annual_capped'] = upper_bound



# ---------- Final tidy columns & save ----------
final_cols = ['Company', 'Company_clean', 'Company Score', 'Job Title', 'Job Title_clean',
              'Location', 'Loc_City', 'Loc_State', 'Salary', 'Salary_min_raw', 'Salary_max_raw',
              'Salary_unit', 'Salary_avg_raw', 'Salary_annual_est', 'Salary_annual_capped']

# ensure all final cols exist
for c in final_cols:
    if c not in df.columns:
        df[c] = None

cleaned = df[final_cols].copy()
cleaned.to_csv(OUT_CSV, index=False)
print(f"\nSaved cleaned dataset to: {OUT_CSV}")




# ---------- VISUALIZATIONS ----------


plt.style.use('default')  # start from default; we'll set background color manually per figure

# Helper to save fig with a bright background and friendly dark colors
def save_fig(fig, filename, fig_size=(10,6), bgcolor='#ffffff'):
    fig.set_facecolor(bgcolor)
    # also set axes background
    for ax in fig.get_axes():
        ax.set_facecolor('#ffffff')  # bright background for axes
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print("Saved:", path)

# 1) Top 10 companies by count of listings
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
top_comp = cleaned['Company_clean'].value_counts().nlargest(10)
bars = ax.barh(top_comp.index[::-1], top_comp.values[::-1], color=['#2E4057', '#3E7CB1', '#5DA5A4', '#F39C12', '#E67E22', '#8E44AD', '#27AE60', '#C0392B', '#D35400', '#1F618D'][0:len(top_comp)])
ax.set_title("Top 10 Companies (by number of internship listings)", fontsize=14, weight='bold')
ax.set_xlabel("Number of Listings")
ax.grid(axis='x', linestyle='--', alpha=0.4)
save_fig(fig, "01_top_companies.png")

plt.close(fig)

# 2) Average (mean) annual salary by top 10 job titles (based on counts)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
top_titles = cleaned['Job Title_clean'].value_counts().nlargest(10).index.tolist()
mean_sal_by_title = cleaned.groupby('Job Title_clean')['Salary_annual_est'].mean().loc[top_titles].sort_values()
bars = ax.barh(mean_sal_by_title.index[::-1], mean_sal_by_title.values[::-1], color=['#34495E', '#7FB3D5', '#73C6B6', '#F7DC6F', '#F1948A', '#BB8FCE', '#76D7C4', '#F5B041', '#D98880', '#5DADE2'][0:len(mean_sal_by_title)])
ax.set_title("Mean Estimated Annual Salary by Top Job Titles", fontsize=14, weight='bold')
ax.set_xlabel("Mean Annual Salary (USD)")
ax.grid(axis='x', linestyle='--', alpha=0.4)
save_fig(fig, "02_mean_salary_by_title.png")
plt.close(fig)

# 3) Distribution histogram of estimated annual salary (capped)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.hist(cleaned['Salary_annual_capped'].dropna(), bins=30, edgecolor='black', color='#2C3E50', alpha=0.9)
ax.set_title("Distribution of Estimated Annual Salary (winsorized)", fontsize=14, weight='bold')
ax.set_xlabel("Annual Salary (USD)")
ax.set_ylabel("Count")
ax.grid(axis='y', linestyle='--', alpha=0.4)
save_fig(fig, "03_salary_distribution.png")
plt.close(fig)

# 4) Boxplot: Salary by company (top 6 companies)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
top6 = cleaned['Company_clean'].value_counts().nlargest(6).index.tolist()
data_to_plot = [cleaned.loc[cleaned['Company_clean']==c, 'Salary_annual_capped'].dropna() for c in top6]
ax.boxplot(data_to_plot, labels=top6, patch_artist=True,
           boxprops=dict(facecolor='#AED6F1'), medianprops=dict(color='#1B4F72'))
ax.set_title("Salary Distribution for Top 6 Companies (winsorized)", fontsize=14, weight='bold')
ax.set_ylabel("Annual Salary (USD)")
ax.tick_params(axis='x', rotation=30)
save_fig(fig, "04_boxplot_salary_by_company.png")
plt.close(fig)

# 5) Scatter: Company Score vs Avg Estimated Annual Salary
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
x = cleaned['Company Score']
y = cleaned['Salary_annual_est']
mask = (~x.isna()) & (~y.isna())
ax.scatter(x[mask], y[mask], s=40, alpha=0.8, c='#2E86C1', edgecolors='#154360')
ax.set_title("Company Score vs Estimated Annual Salary", fontsize=14, weight='bold')
ax.set_xlabel("Company Score")
ax.set_ylabel("Estimated Annual Salary (USD)")
ax.grid(True, linestyle='--', alpha=0.3)
save_fig(fig, "05_score_vs_salary.png")
plt.close(fig)

# 6) Pie chart: Top locations (by listing count)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
top_loc = cleaned['Location'].value_counts().nlargest(6)
colors = ['#1F618D', '#2E4053', '#27AE60', '#F39C12', '#AF7AC5', '#F1948A'][0:len(top_loc)]
ax.pie(top_loc.values, labels=top_loc.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(edgecolor='white'))
ax.set_title("Top Locations (by number of listings)", fontsize=14, weight='bold')
save_fig(fig, "06_top_locations_pie.png")
plt.close(fig)

# 7) Heatmap-like visualization of missingness (basic; without seaborn)
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
missing_matrix = cleaned.isnull().astype(int).T  # columns as rows for display
cax = ax.imshow(missing_matrix, aspect='auto', interpolation='nearest', cmap='Greys', vmin=0, vmax=1)
ax.set_yticks(range(len(missing_matrix.index)))
ax.set_yticklabels(missing_matrix.index)
ax.set_xticks([])
ax.set_title("Missingness Map (1 = missing, 0 = present)", fontsize=12, weight='bold')
save_fig(fig, "07_missingness_map.png")
plt.close(fig)

# 8) Bar: Count by salary unit (hour/year/unknown inferred)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
unit_counts = cleaned['Salary_unit'].fillna('unknown').value_counts()
bars = ax.bar(unit_counts.index, unit_counts.values, color=['#1B4F72', '#117A65', '#7D3C98'][0:len(unit_counts)])
ax.set_title("Counts by Salary Unit (hour / year / unknown-inferred)", fontsize=14, weight='bold')
ax.set_xlabel("Salary Unit")
ax.set_ylabel("Count")
ax.grid(axis='y', linestyle='--', alpha=0.4)
save_fig(fig, "08_salary_unit_counts.png")
plt.close(fig)

print("\nAll visualizations saved to:", OUT_DIR)


# Provide a small summary of cleaning actions
summary = {
    "initial_rows": df.shape[0],
    "final_rows": cleaned.shape[0],
    "missing_before": {"Company": df['Company'].isnull().sum(), "Company Score": df['Company Score'].isnull().sum() if 'Company Score' in df.columns else None, "Salary": df['Salary'].isnull().sum()},
    "missing_after": cleaned.isnull().sum().to_dict(),
    "salary_parsed_count": cleaned['Salary_annual_est'].notnull().sum()
}
print("\nCleaning summary:")
for k,v in summary.items():
    print(k, ":", v)



# End of script. The cleaned CSV is saved and 8 PNG visualizations are saved.








