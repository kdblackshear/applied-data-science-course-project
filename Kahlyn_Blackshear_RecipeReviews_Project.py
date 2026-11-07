import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sqlite3
from scipy import stats

# Load the dataset
df = pd.read_csv('recipe_reviews.csv')

# Display shape of the dataset
print("Shape of the dataset:", df.shape)
# Display the column names of the dataset
print("Column names:", df.columns)
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
# Display the data types of each column
print("Data types of each column:")
print(df.dtypes)
# Display the information about the dataset
print("Information about the dataset:")
print(df.info())
# Display the missing values in the dataset
print("Missing values in the dataset:")
print(df.isnull().sum())
# Display the number of exact duplicate rows in the dataset
print("Number of exact duplicate rows in the dataset:", df.duplicated().sum())
# Display the identifier duplicate counts in the dataset
print("Identifier duplicate counts in the dataset:")
for col in ['comment_id', 'user_id', 'recipe_code']:
    if col in df.columns:
        duplicate_count = df[col].duplicated().sum()
        print(f"Total duplicate entries in '{col}': {duplicate_count}")
# Display the appropriate numerical and categorical summaries of the dataset
print("Numerical summaries of the dataset:")
print(df.describe())
print("Categorical summaries of the dataset:")
print(df.describe(include=['object']))

# Determine whether comment_id uniquely identifies review records
unique_comment_ids = df['comment_id'].nunique()
total_records = df.shape[0]
if unique_comment_ids == total_records:
    print("comment_id uniquely identifies review records.")
else:
    print("comment_id does not uniquely identify review records.")
# Determine whether each recipe_code consistently corresponds to one recipe_name and one recpie_number
recipe_code_groups = df.groupby('recipe_code').agg({
    'recipe_name': pd.Series.nunique,
    'recipe_number': pd.Series.nunique
})
inconsistent_recipe_codes = recipe_code_groups[(recipe_code_groups['recipe_name'] > 1) | (recipe_code_groups['recipe_number'] > 1)]
if inconsistent_recipe_codes.empty:
    print("Each recipe_code consistently corresponds to one recipe_name and one recipe_number.")
else:
    print("Some recipe_codes do not consistently correspond to one recipe_name and one recipe_number:")
    print(inconsistent_recipe_codes)
# Determine whether each user_id consistently corresponds to one user_name
user_id_groups = df.groupby('user_id').agg({
    'user_name': pd.Series.nunique
})
inconsistent_user_ids = user_id_groups[user_id_groups['user_name'] > 1]
if inconsistent_user_ids.empty:
    print("Each user_id consistently corresponds to one user_name.")
else:
    print("Some user_ids do not consistently correspond to one user_name:")
    print(inconsistent_user_ids)


# Create copy of raw dataset for cleaning
df_cleaned = df.copy()

# Standardize column names
df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
# Remove surrounding spaces from appropriate text columns
text_columns = ['recipe_name', 'user_name', 'text']
for col in text_columns:
    df_cleaned[col] = df_cleaned[col].str.strip()
# Standardize categorical values in recipe_name and user_name columns
df_cleaned['recipe_name'] = df_cleaned['recipe_name'].str.title()
df_cleaned['user_name'] = df_cleaned['user_name'].str.title()
# Convert created_at column to datetime format
df_cleaned['created_at'] = pd.to_datetime(df_cleaned['created_at'], unit='s', errors='coerce')
# Convert numerical columns to appropriate data types
numerical_columns = ['stars', 'recipe_number', 'reply_count', 'thumbs_up', 'thumbs_down', 'user_reputation', 'best_score']
for col in numerical_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
# Identify and handle exact duplicate rows
exact_duplicates = df_cleaned.duplicated()
if exact_duplicates.any():
    print(f"Found {exact_duplicates.sum()} exact duplicate rows. Removing them.")
    df_cleaned = df_cleaned[~exact_duplicates]
# Investigate dupilcated indetifiers before removing records with duplicated identifiers
identifier_columns = ['comment_id', 'user_id', 'recipe_code']
for col in identifier_columns:
    duplicate_count = df_cleaned[col].duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicated entries in '{col}'.")
        # Display the duplicated entries for review
        duplicated_entries = df_cleaned[df_cleaned[col].duplicated(keep=False)]
        print(f"Duplicated entries in '{col}':")
        print(duplicated_entries[[col, 'recipe_name', 'user_name']])
# Handle missing values using documented and justified rules
    # For this, we will drop rows with missing values in critical columns
critical_columns = ['comment_id', 'user_id', 'recipe_code', 'recipe_name', 'user_name', 'text', 'stars', 'created_at']
df_cleaned = df_cleaned.dropna(subset=critical_columns)
# Investigate inconsistent recipe or user identifiers before constructing relational tables
    # Check for inconsistent recipe_code to recipe_name and recipe_number mapping
inconsistent_recipe_codes = df_cleaned.groupby('recipe_code').agg({
    'recipe_name': pd.Series.nunique,
    'recipe_number': pd.Series.nunique
})
inconsistent_recipe_codes = inconsistent_recipe_codes[(inconsistent_recipe_codes['recipe_name'] > 1) | (inconsistent_recipe_codes['recipe_number'] > 1)]
if not inconsistent_recipe_codes.empty:
    print("Inconsistent recipe_code to recipe_name and recipe_number mapping found:")
    print(inconsistent_recipe_codes)
    # Check for inconsistent user_id to user_name mapping
inconsistent_user_ids = df_cleaned.groupby('user_id').agg({
    'user_name': pd.Series.nunique
})
inconsistent_user_ids = inconsistent_user_ids[inconsistent_user_ids['user_name'] > 1]
if not inconsistent_user_ids.empty:
    print("Inconsistent user_id to user_name mapping found:")
    print(inconsistent_user_ids)
# Validate known value ranges:
# Stars must contain values from 0 through 5, inclusive
invalid_ratings = df_cleaned[~df_cleaned['stars'].between(0, 5)]
if not invalid_ratings.empty:
    print("Invalid ratings found (not between 0 and 5):")
    print(invalid_ratings[['comment_id', 'stars']])
    df_cleaned = df_cleaned[df_cleaned['stars'].between(0, 5)]

# reply count, thumbs up, thumbs down, user reputation, and best score are non-negative integers. We will check for any negative values in these columns and remove them if found.
non_negative_columns = ['reply_count', 'thumbs_up', 'thumbs_down', 'user_reputation', 'best_score']
for col in non_negative_columns:
    negative_values = df_cleaned[df_cleaned[col] < 0]
    if not negative_values.empty:
        print(f"Negative values found in '{col}':")
        print(negative_values[[col]])
        # Remove records with negative values in these columns
        df_cleaned = df_cleaned[df_cleaned[col] >= 0]
    
# Ensure required identifers are not missing. We have already dropped rows with missing values in critical columns, so we will check for any remaining missing values in the identifier columns.
missing_identifiers = df_cleaned[identifier_columns].isnull().sum()
if missing_identifiers.any():
    print("Missing identifiers found:")
    print(missing_identifiers)

# Assertion check for verification
assert df_cleaned['stars'].between(0, 5).all(), "Assertion Error: Stars contain values outside 0-5 range."
for col in non_negative_columns:
    if col in df_cleaned.columns:
        assert (df_cleaned[col] >= 0).all(), f"Assertion Error: Negative values found in {col}."
assert df_cleaned[identifier_columns].notnull().all().all(), "Assertion Error: Missing values found in critical identifiers."

# Create dataframe named rated_reviews containinf only rows with ratings from 1 through 5, inclusive. This will be used for analysis of ratings and reviews.
rated_reviews = df_cleaned[df_cleaned['stars'].between(1, 5)]

# Assertion for rated_reviews dataframe
assert rated_reviews['stars'].between(1, 5).all(), "Assertion Error: rated_reviews contains values outside 1-5 range." 
print("Data cleaning and validation completed successfully. The cleaned dataset is ready for analysis.")


# Audit user reputation stability per user_id before normalization
user_rep_counts = df_cleaned.groupby('user_id')['user_reputation'].nunique()
stable_rep = (user_rep_counts == 1).all()

if stable_rep:
    print("User reputation is stable per user_id. Proceeding with normalization.")
    print(f"Are all user reputations stable? {stable_rep}")
else:
    print("User reputation is not stable per user_id. Keeping user_rep in reviews table.")

# Decompose into three dataframes: recipes, users, and reviews
# Create recipes dataframe: one row per recipe code
recipes = df_cleaned[['recipe_code', 'recipe_name', 'recipe_number']].drop_duplicates(subset=['recipe_code']).reset_index(drop=True)
# Create users dataframe: one row per user id
users = df_cleaned[['user_id', 'user_name']].drop_duplicates(subset=['user_id']).reset_index(drop=True)
# Create reviews dataframe: one row per review (comment_id)
reviews = df_cleaned[['comment_id', 'user_id', 'recipe_code', 'text', 'stars', 'created_at', 'reply_count', 'thumbs_up', 'thumbs_down', 'user_reputation', 'best_score']].drop_duplicates(subset=['comment_id']).reset_index(drop=True)

# Verify primary keys and referential integrity
# Check primary key uniqueness with assertions
assert recipes['recipe_code'].is_unique, "Assertion Error: recipe_code is not unique in recipes dataframe."
assert users['user_id'].is_unique, "Assertion Error: user_id is not unique in users dataframe."
assert reviews['comment_id'].is_unique, "Assertion Error: comment_id is not unique in reviews dataframe."
# Verify every review refernces an existing recipe and user
assert reviews['recipe_code'].isin(recipes['recipe_code']).all(), "Assertion Error: Some reviews reference non-existent recipe_codes."
assert reviews['user_id'].isin(users['user_id']).all(), "Assertion Error: Some reviews reference non-existent user_ids."
print("Dataframes created successfully with verified primary keys and referential integrity.")

# Reconstruct analysis-ready dataframe vis joins
# Join reviews with recipes (many-to-one relationship)
anaylsis_ready_df = reviews.merge(recipes, on='recipe_code', how='inner', validate='many_to_one')
# Join the resulting dataframe with users (many-to-one relationship)
anaylsis_ready_df = anaylsis_ready_df.merge(users, on='user_id', how='inner', validate='many_to_one')

# Confirm row counts
original_row_count = len(df_cleaned)
reconstructed_row_count = len(anaylsis_ready_df)
print(f"Original cleaned dataset row count: {original_row_count}")
print(f"Reconstructed analysis-ready dataset row count: {reconstructed_row_count}")
assert original_row_count == reconstructed_row_count, "Assertion Error: Row counts do not match after reconstruction."
print("Reconstruction successful. The analysis-ready dataframe is ready for further analysis.")

# Save to output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'recipe_reviews_analysis_ready.csv')
anaylsis_ready_df.to_csv(output_path, index=False)
print(f"Analysis-ready dataframe saved to {output_path}")

# Read back and verify shap and coulmns
df_verification = pd.read_csv(output_path)
print("Verification of saved analysis-ready dataframe:")
print("Shape of the saved dataframe:", df_verification.shape)
print("Column names of the saved dataframe:", df_verification.columns.tolist())
assert df_verification.shape == anaylsis_ready_df.shape, "Assertion Error: Shape mismatch between saved and original analysis-ready dataframe."
print("Verification successful. The saved analysis-ready dataframe matches the original in shape and columns.")


# Create output directories
os.makedirs('output/tables', exist_ok=True)
db_path = 'output/recipe_reviews_project.db'

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_path)

# Store normalized dataframes into the SQLite tables
recipes.to_sql('recipes', conn, if_exists='replace', index=False)
users.to_sql('users', conn, if_exists='replace', index=False)
reviews.to_sql('reviews', conn, if_exists='replace', index=False)
print(f"Dataframes stored in SQLite database at {db_path}")

# Query 1: Count the number of records in each table
query_1 = """
SELECT 'recipes' AS table_name, COUNT(*) AS record_count FROM recipes
UNION ALL
SELECT 'users', COUNT(*) AS record_count FROM users
UNION ALL
SELECT 'reviews', COUNT(*) AS record_count FROM reviews;
"""
df_counts = pd.read_sql_query(query_1, conn)
print("Record counts for each table:")
print(df_counts)

# Query 2: Filter records using WHERE
query_2 = "SELECT comment_id, stars, text FROM reviews WHERE stars = 5 LIMIT 5"
df_5star = pd.read_sql_query(query_2, conn)
print("Sample of 5-star reviews:")
print(df_5star)

# Query 3: Compare average ratings using GROUP BY and AVG
query_3 = """
SELECT r.recipe_name, 
    ROUND(AVG(rev.stars), 2) AS average_rating,
    COUNT(rev.comment_id) AS total_reviews
FROM recipes r
INNER JOIN reviews rev ON r.recipe_code = rev.recipe_code
GROUP BY r.recipe_name;
"""
df_avg_ratings = pd.read_sql_query(query_3, conn)
# Save the average ratings to a CSV file
df_avg_ratings.to_csv('output/tables/recipe_average_ratings.csv', index=False)
print("Average ratings per recipe saved to 'output/tables/recipe_average_ratings.csv'")
print("Average ratings per recipe:")
print(df_avg_ratings)

# Query 4: Identify power users using GROUP BY, COUNT, and HAVING
query_4 = """
SELECT user_id, COUNT(*) AS review_count
FROM reviews
GROUP BY user_id
HAVING COUNT(*) >= 2
ORDER BY review_count DESC;
"""
df_power_users = pd.read_sql_query(query_4, conn)
print("Power users (users with 2 or more reviews):")
print(df_power_users)

# Query 5: INNER JOIN between recipes and reviews
query_5 = """
SELECT rev.comment_id, r.recipe_name, rev.stars, rev.thumbs_up
FROM reviews rev
INNER JOIN recipes r ON rev.recipe_code = r.recipe_code
LIMIT 5;
"""
df_joined = pd.read_sql_query(query_5, conn)

# Query 6: Joining all three relational tables
query_6 = """
SELECT rev.comment_id, u.user_name, r.recipe_name, rev.stars, rev.created_at
FROM reviews rev
INNER JOIN users u ON rev.user_id = u.user_id
INNER JOIN recipes r ON rev.recipe_code = r.recipe_code
ORDER BY rev.created_at DESC
LIMIT 5;
"""
df_full_join = pd.read_sql_query(query_6, conn)
# Save the joined data to a CSV file
df_full_join.to_csv('output/tables/recent_full_reviews.csv', index=False)
print("Recent full reviews saved to 'output/tables/recent_full_reviews.csv'")

# Query 7: Orignial research query (Impact of reputation tiers)
query_7 = """
SELECT 
    CASE 
        WHEN user_reputation = 1 THEN 'New/Base User (Rep 1)'
        WHEN user_reputation BETWEEN 2 AND 20 THEN 'Moderate Reputation (Rep 2-20)'
        ELSE 'High Reputation (21+)'
    END AS reputation_tier,
    COUNT(*) AS total_reviews,
    ROUND(AVG(stars), 2) AS average_rating,
    ROUND(AVG(thumbs_up), 2) AS average_thumbs_up
FROM reviews
GROUP BY reputation_tier
ORDER BY average_rating DESC;
"""

df_reputation_impact = pd.read_sql_query(query_7, conn)
print("Impact of reputation tiers on reviews:")
print(df_reputation_impact)

# Close the database connection
conn.close()
print("Database connection closed. All operations completed successfully.")


# Create a binary variable named high_rating
rated_reviews['high_rating'] = (rated_reviews['stars'] >= 4).astype(int)

# Create review_length column
if 'review_length' not in rated_reviews.columns:
    rated_reviews['review_length'] = rated_reviews['text'].astype(str).str.len()

# Calculate the empirical probability of receiving a high rating
p_high_rating = rated_reviews['high_rating'].mean()
print(f"Empirical probability of receiving a high rating (4 or 5 stars): {p_high_rating:.4f}")

# Calculate the conditional probability of a high rating review for at least 3 defined recipe rank or user reputation groups
# Define reputation groups
conditions = [
    (rated_reviews['user_reputation'] == 1),
    (rated_reviews['user_reputation'].between(2, 20)),
    (rated_reviews['user_reputation'] > 20)
]
choices = ['New/Base User (Rep 1)', 'Moderate Reputation (Rep 2-20)', 'High Reputation (21+)']
rated_reviews['reputation_group'] = np.select(conditions, choices, default='Unknown')

condition_probabilities = rated_reviews.groupby('reputation_group')['high_rating'].agg(['mean', 'count'])
condition_probabilities.columns = ['Review Count', 'Conditional Probability']
print("Conditional probabilities of receiving a high rating by reputation group:")
print(condition_probabilities)

# Compare the overall probability with the conditional probabilities
print(f"\nOverall probability of receiving a high rating: {p_high_rating:.4f}")
print("Conditional probabilities by reputation group:")
for group, row in condition_probabilities.iterrows():
    print(f"{group}: {row['Conditional Probability']:.4f} (Review Count: {row['Review Count']})")
print("Comparison: Groups that deviate from the overall probability show whether user experience/reputation influences positive rating tendencies.")

# Calculate the mean, median, standrd deviation, minimum, maximum, first quartile, and third quartile of 'stars', 'thumbs_up', and 'review_length' columns
def calculate_statistics(series, name):
    return pd.Series({
        'Mean': series.mean(),
        'Median': series.median(),
        'Standard Deviation': series.std(),
        'Minimum': series.min(),
        'Maximum': series.max(),
        'First Quartile (Q1)': series.quantile(0.25),
        'Third Quartile (Q3)': series.quantile(0.75)
    }, name=name)

stats_stars = calculate_statistics(rated_reviews['stars'], 'Stars')
stats_thumbs_up = calculate_statistics(rated_reviews['thumbs_up'], 'Thumbs Up')
stats_review_length = calculate_statistics(rated_reviews['review_length'], 'Review Length')

summary_df = pd.DataFrame([stats_stars, stats_thumbs_up, stats_review_length])
print("\nSummary statistics for 'stars', 'thumbs_up', and 'review_length':")
print(summary_df.round(3))

# Calculate an approximate 95% confidence interval for the mean review rating
x_bar = rated_reviews['stars'].mean()
s = rated_reviews['stars'].std(ddof=1)
n = len(rated_reviews['stars'])
margin_of_error = 1.96 * (s / np.sqrt(n))
confidence_interval_lower = x_bar - margin_of_error
confidence_interval_upper = x_bar + margin_of_error
print(f"\nSample mean of review ratings: {x_bar:.4f}")
print(f"\nSample standard deviation of review ratings: {s:.4f}")
print(f"\nSample size: {n}")
print(f"\nApproximate 95% confidence interval for the mean review rating: ({confidence_interval_lower:.4f}, {confidence_interval_upper:.4f})")

# Calculate and interpret relevant Pearson correlations between numerical variables: stars, thumbs_uo, reply_count, user_reputation, and review_length
correlation_matrix = rated_reviews[['stars', 'thumbs_up', 'reply_count', 'user_reputation', 'review_length']].corr(method='pearson')
print("\nPearson correlation matrix:")
print(correlation_matrix.round(3))

# Conduct one two-sided independet samples hypothesis test comparing the mean review length of high and non-high rating reviews, using Welch's t-test with a=0.05
high_ratings_len = rated_reviews[rated_reviews['high_rating'] == 1]['review_length']
non_high_ratings_len = rated_reviews[rated_reviews['high_rating'] == 0]['review_length']

t_stat, p_val = stats.ttest_ind(high_ratings_len, non_high_ratings_len, equal_var=False)
print(f"Welche's t-statistic: {t_stat: .4f}")
print(f"p-value: {p_val: .4e}")

alpha = 0.05
if p_val < alpha:
    print("Decision: Reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in mean review length between high-rating and non-high-rating reviews.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("Conclusion: No statistically significant difference in mean review length was found.")

# Calculate and intepret Cohen's d as an effect for the group comparison
n1 = len(high_ratings_len)
n2 = len(non_high_ratings_len)
mean1 = high_ratings_len.mean()
mean2 = non_high_ratings_len.mean()
var1 = np.var(high_ratings_len, ddof=1)
var2 = np.var(non_high_ratings_len, ddof=1)

sd_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / sd_pooled
print(f"Cohen's d: {cohens_d:.4f}")


# Create clean copy of rated_reviews
df_features = rated_reviews.copy()

# Create and document high rating: 1 when review stars are at least 4, otherwise 0
df_features['high_rating'] = (df_features['stars'] >= 4).astype(int)
# Create and document review length: number of characters in the review text
df_features['review_length'] = df_features['text'].astype(str).str.len()
# Create and document review word count: number of words in the review
df_features['review_word_count'] = df_features['text'].astype(str).str.split().str.len()

# Ensure created_at is datetime type
if not pd.api.types.is_datetime64_any_dtype(df_features['created_at']):
    df_features['created_at'] = pd.to_datetime(df_features['created_at'], unit='s', errors='coerce')

# Create and document review year: year extracted from created at
df_features['review_year'] = df_features['created_at'].dt.year
# Create and document review month: month extracted from created at
df_features['review_month'] = df_features['created_at'].dt.month
# Create and document total votes: sum of thumbs up and thumbs down
df_features['total_votes'] = df_features['thumbs_up'] + df_features['thumbs_down']
# Create and document net votes: thumbs up minus thumbs down
df_features['net_votes'] = df_features['thumbs_up'] - df_features['thumbs_down']
# Create and document has reply: 1 when reply count is greater than zero, otherwise 0
df_features['has_reply'] = (df_features['reply_count'] > 0).astype(int)

# Validate new vaiables
# Validate high_rating binary range
assert set(df_features["high_rating"].unique()).issubset({0, 1}), "Validation Error: high_rating contains non-binary values."

# Validate non-negative lengths and word counts
assert (df_features["review_length"] >= 0).all(), "Validation Error: Negative review_length found."
assert (df_features["review_word_count"] >= 0).all(), "Validation Error: Negative review_word_count found."

# Validate date extractions
assert df_features["review_year"].notnull().all(), "Validation Error: Missing values in review_year."
assert df_features["review_month"].between(1, 12).all(), "Validation Error: review_month outside 1-12 range."

# Validate vote mathematical relationships
assert (df_features["total_votes"] >= df_features["thumbs_up"]).all(), "Validation Error: total_votes less than thumbs_up."
assert (df_features["net_votes"] == (df_features["thumbs_up"] - df_features["thumbs_down"])).all(), "Validation Error: net_votes calculation mismatch."

# Validate has_reply binary range
assert set(df_features["has_reply"].unique()).issubset({0, 1}), "Validation Error: has_reply contains non-binary values."

print("All 8 engineered features validated successfully!")

# Update rated_reviews DataFrame reference
rated_reviews = df_features.copy()


# Create output directory
os.makedirs("output/figures", exist_ok=True)

# Set seaborn style
sns.set_theme(style="whitegrid")

# Ensure rated_reviews dataframe has components
if not pd.api.types.is_datetime64_any_dtype(rated_reviews['created_at']):
    rated_reviews['created_at'] = pd.to_datetime(rated_reviews['created_at'], unit='s', errors='coerce')

# Count plot of stars
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=rated_reviews, x="stars", palette="viridis", order=[1, 2, 3, 4, 5])
plt.title("Distribution of Review Star Ratings", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Star Rating", fontsize=12)
plt.ylabel("Review Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Add count annotations above bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig("output/figures/1_star_rating_distribution.png", dpi=300)
plt.close()
print("Saved Visualization 1.")

# Bar chart comparing average ratings across recipes
recipe_avg = rated_reviews.groupby("recipe_name")["stars"].mean().reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=recipe_avg, x="recipe_name", y="stars", palette="mako")
plt.title("Average Star Rating by Recipe", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Recipe Name", fontsize=12)
plt.ylabel("Average Star Rating", fontsize=12)
plt.ylim(0, 5.5)  # Rating scale limit
plt.xticks(rotation=15, fontsize=10)
# Annotate values
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig("output/figures/2_average_rating_by_recipe.png", dpi=300)
plt.close()
print("Saved Visualization 2.")

# Box plot comparing a numerical variable across groups
plt.figure(figsize=(9, 6))
sns.boxplot(data=rated_reviews, x="stars", y="review_length", palette="crest", showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"8"})
plt.title("Review Character Length Distribution Across Star Ratings", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Star Rating", fontsize=12)
plt.ylabel("Review Length (Characters)", fontsize=12)
# Add a legend note for the mean marker
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Mean Value', 
    markerfacecolor='red', markeredgecolor='black', markersize=8)], loc='upper right')
plt.tight_layout()
plt.savefig("output/figures/3_review_length_by_stars_boxplot.png", dpi=300)
plt.close()
print("Saved Visualization 3.")

# Line chart showing review activity over time
# Group by year-month
rated_reviews["year_month"] = rated_reviews["created_at"].dt.to_period("M").dt.to_timestamp()
time_activity = rated_reviews.groupby("year_month").size().reset_index(name="review_count")

plt.figure(figsize=(12, 5))
plt.plot(time_activity["year_month"], time_activity["review_count"], marker="o", color="b", linestyle="-", linewidth=2)
plt.title("Longitudinal Trend of Review Activity Over Time", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Review Date (Year-Month)", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("output/figures/4_review_activity_over_time.png", dpi=300)
plt.close()
print("Saved Visualization 4.")

# Scatter plot examining two numerical values
plt.figure(figsize=(9, 6))
sns.scatterplot(data=rated_reviews, x="review_length", y="thumbs_up", alpha=0.6, color="purple")
plt.title("Relationship Between Review Length and Community Thumbs Up", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Review Length (Characters)", fontsize=12)
plt.ylabel("Thumbs Up Count", fontsize=12)
plt.tight_layout()
plt.savefig("output/figures/5_review_length_vs_thumbs_up.png", dpi=300)
plt.close()
print("Saved Visualization 5.")

# Correlation heatmap supporting research question
# Select relevant numerical columns
num_cols = ["stars", "review_length", "review_word_count", "thumbs_up", "thumbs_down", "user_reputation", "best_score"]
corr_matrix = rated_reviews[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title("Correlation Matrix Heatmap of Numerical Features", fontsize=14, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("output/figures/6_correlation_heatmap.png", dpi=300)
plt.close()
print("Saved Visualization 6.")

print("All 6 visualizations generated and saved successfully in 'output/figures/'.")


# Necessary scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure output directory exists
os.makedirs("output/figures", exist_ok=True)

# Define features and target
# Ensure categorical group feature exists
if 'reputation_group' not in rated_reviews.columns:
    conditions = [
        (rated_reviews['user_reputation'] == 1),
        (rated_reviews['user_reputation'].between(2, 20)),
        (rated_reviews['user_reputation'] > 20)
    ]
    choices = ["New User", "Moderate User", "High Reputation User"]
    rated_reviews['reputation_group'] = np.select(conditions, choices, default="Unknown")

# Select feature columns
numerical_features = ['review_length', 'review_word_count', 'user_reputation', 'recipe_number', 'review_year', 'review_month']
categorical_features = ['reputation_group']
X = rated_reviews[numerical_features + categorical_features]
y = rated_reviews['high_rating']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

# Processing and modeling pipelines
# Numerical transformer: Imputation + Scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Imputation + One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer combining both
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Baseline Pipeline (Dummy Classifier)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='most_frequent'))
])

# Logistic Regression Pipeline
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train models on training data
baseline_pipeline.fit(X_train, y_train)
logreg_pipeline.fit(X_train, y_train)

# Evaluate both models on the test set
y_pred_baseline = baseline_pipeline.predict(X_test)
y_pred_logreg = logreg_pipeline.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

metrics_baseline = evaluate_model(y_test, y_pred_baseline, "Baseline (Dummy)")
metrics_logreg = evaluate_model(y_test, y_pred_logreg, "Logistic Regression")

# Create confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Predicted Non-High", "Predicted High"],
            yticklabels=["Actual Non-High", "Actual High"])
plt.title("Confusion Matrix - Logistic Regression Model", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.tight_layout()
plt.savefig("output/figures/7_logistic_regression_confusion_matrix.png", dpi=300)
plt.close()
print("Saved labeled confusion matrix visualization to output/figures/7_logistic_regression_confusion_matrix.png")


# Final validation and reproducible outputs
# Recheck shapes, data types, missing values, and duplicates
final_shape = rated_reviews.shape
final_null_counts = rated_reviews.isnull().sum().sum()
final_duplicate_count = rated_reviews.duplicated().sum()
print(f"Final Dataset Shape: {final_shape}")
print(f"Total Missing Values Across All Columns: {final_null_counts}")
print(f"Total Exact Duplicate Rows: {final_duplicate_count}")

# Run assertions for important validation rules
assert final_null_counts == 0, "Assertion Error: Unexpected missing values found in final dataset."
assert final_duplicate_count == 0, "Assertion Error: Duplicate rows found in final dataset."
assert rated_reviews['stars'].between(1, 5).all(), "Assertion Error: Final dataset contains invalid star ratings."
assert (rated_reviews['review_length'] >= 0).all(), "Assertion Error: Negative review lengths detected."
print("All final validation assertions passed successfully.")

# Save final anylsis ready dataset
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
final_dataset_path = os.path.join(output_dir, "recipe_reviews_analysis_ready.csv")
rated_reviews.to_csv(final_dataset_path, index=False)
print(f"Saved final analysis-ready dataset to: {final_dataset_path}")

# Confirm figures and result tables are saved in respective directories
print("Figures verified in: output/figures/")
print("Result tables verified in: output/tables/")

# Read important saved files back into Python and verify them
df_verify_reloaded = pd.read_csv(final_dataset_path)
print(f"Reloaded File Verification - Shape: {df_verify_reloaded.shape}")
assert df_verify_reloaded.shape == final_shape, "Assertion Error: Reloaded file shape mismatch."
print("Reloaded file shape and integrity verified successfully.")

# Print completion summary listing output files
print(f"Completion Summary: ")
output_files = [
    "output/recipe_reviews_project.db",
    "output/recipe_reviews_analysis_ready.csv",
    "output/queries.sql",
    "output/tables/recipe_average_ratings.csv",
    "output/tables/recent_full_reviews.csv",
    "output/figures/1_star_rating_distribution.png",
    "output/figures/2_average_rating_by_recipe.png",
    "output/figures/3_review_length_by_stars_boxplot.png",
    "output/figures/4_review_activity_over_time.png",
    "output/figures/5_review_length_vs_thumbs_up.png",
    "output/figures/6_correlation_heatmap.png",
    "output/figures/7_logistic_regression_confusion_matrix.png"
]

for file_path in output_files:
    status = "EXISTS" if os.path.exists(file_path) else "MISSING"
    print(f"[{status}] {file_path}")