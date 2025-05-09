import pandas as pd

df = pd.read_csv("CSV_FILE")

# assume df is loaded
df.columns = [c.replace('\ufeff','').strip() for c in df.columns]

def make_resume_text(r):
    return (
      f"Career Objective: {r['career_objective']}\n"
      f"Skills: {r['skills']}\n"
      f"Education: {r['degree_names']} at {r['educational_institution_name']} ({r['passing_years']})\n"
      f"Experience: {r['positions']} at {r['professional_company_names']} — {r['responsibilities']}\n"
      f"Certifications: {r.get('certification_skills','')} by {r.get('certification_providers','')}\n"
    )

def make_jd_text(r):
    return (
      f"Job Position: {r['job_position_name']}\n"
      f"Required Skills: {r['skills_required']}\n"
      f"Responsibilities: {r['responsibilities.1']}\n"
      f"Education Requirements: {r['educationaL_requirements']}\n"
    )

# drop rows missing any of these core columns:
core = ['career_objective','skills','degree_names','positions','responsibilities',
        'job_position_name','skills_required','responsibilities.1','educationaL_requirements','matched_score']
df = df.dropna(subset=core)

# build new columns
df['resume_text'] = df.apply(make_resume_text, axis=1)
df['jd_text']     = df.apply(make_jd_text, axis=1)
df['label']       = (df['matched_score'] >= 0.7).astype(int)

# save this cleaned table for reference
df.to_csv("prepared_data.csv", index=False)

print("Step 0 done: prepared_data.csv with resume_text, jd_text, label")

print(df.columns.tolist())
print(df.shape)
