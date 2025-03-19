import pandas as pd
import csv
# load the Excel file
file_path = "ResearchOutputs.xlsx"
df = pd.read_excel(file_path)

# Extract required columns and rename them according to project requirements
df_filtered = df[['OutputTitle', 'OutputYear', 'OutputVenue']].copy()
df_filtered.columns = ['Title', 'Year', 'Agency']

# Handle missing values by filling with default values
df_filtered['Year'] = df_filtered['Year'].fillna(0).astype(int) # Ensure 'Year' is an integer
df_filtered['Agency'] = df_filtered['Agency'].fillna("Unknown") # Fill missing agency names

# NLP-based keyword extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure text data is not missing
df['ProjectTitle'].fillna("", inplace=True)
df['OutputBiblio'].fillna("", inplace=True)

# Combine text fields for keyword extraction
df['combined_text'] = df['ProjectTitle'] + " " + df['OutputBiblio']

# Apply TF-IDF to extract unique keywords
vectorizer = TfidfVectorizer(stop_words="english", max_features=25)  # Expanding max features
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
feature_names = vectorizer.get_feature_names_out()

# Retrieve top 5 keywords for each document
keywords_list = []
for row in tfidf_matrix.toarray():
    top_keywords = [feature_names[i] for i in row.argsort()[-10:]]
    keywords_list.append(";".join(top_keywords))

# Store keywords in the dataframe 
df_filtered['Keywords'] = keywords_list

# Data validation function to check completeness and correctness
def validate_data(df):
    errors = []
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        errors.append("Missing values detected. Please check data integrity.")
    
    # Validate 'Year' column format
    if not df['Year'].apply(lambda x: isinstance(x, int) and x >= 0).all():
        errors.append("The 'Year' column must contain non-negative integers.")

    # Ensure 'Title' column is not empty
    if df['Title'].str.strip().eq("").any():
        errors.append("Empty values detected in the 'Title' column.")

    # Return validation result
    return "Data validation passed" if not errors else "\n".join(errors)

# Run data validation
validationn_result =  validate_data(df_filtered)

# Save processed data to CSV file
output_csv = "fsrdc_outputs.csv"
df_filtered.to_csv(output_csv, index=False, encoding="utf-8")

if __name__ == "__main__":
    data = pd.read_csv('fsrdc_outputs.csv')
    print(data)