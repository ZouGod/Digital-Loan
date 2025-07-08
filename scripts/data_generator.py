import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()

# ========== UPDATED CONFIGURABLE PARAMETERS (USD) - CAMBODIA 2025 ==========
# Source: National Institute of Statistics (NIS) 2023 Report
AGE_RANGE = (18, 65)
GENDER_PROBS = [0.48, 0.52]  # Male, Female (NIS 2023)
MARITAL_STATUS = ['Single', 'Married', 'Divorced', 'Widowed']
MARITAL_PROBS = [0.32, 0.58, 0.07, 0.03]  # Updated per NIS
DEPENDENTS_RANGE = (0, 5)  # Increased for rural households member in house 

# Cambodia provinces (all 25 provinces with equal weights)
PROVINCES = [
    'Phnom Penh', 'Kandal', 'Preah Sihanouk', 'Battambang', 
    'Siem Reap', 'Kampong Cham', 'Takeo', 'Kampong Speu',
    'Kampong Thom', 'Kampot', 'Svay Rieng', 'Tboung Khmum',
    'Pursat', 'Prey Veng', 'Koh Kong', 'Kep',
    'Ratanakiri', 'Mondulkiri', 'Stung Treng', 'Pailin',
    'Banteay Meanchey', 'Kampong Chhnang', 'Kratie', 
    'Oddar Meanchey', 'Preah Vihear'
]

_raw_probs = [1/25] * 25
PROVINCE_PROBS = [p/sum(_raw_probs) for p in _raw_probs]
EDUCATION_JOB_MAPPING = {
    'None': {
        'jobs': ['Farmer', 'Fishery Worker', 'Construction Laborer', 'Domestic Worker'],
        'employment': ['Self-Employed'],
        'prob': 0.18
    },
    'Primary': {
        'jobs': ['Farmer', 'Tuk-Tuk Driver', 'Construction Worker', 'Garment Worker', 'Market Vendor'],
        'employment': ['Self-Employed'],
        'prob': 0.42
    },
    'Secondary': {
        'jobs': ['Garment Worker', 'Shop Assistant', 'Waitstaff', 'Security Guard', 'Taxi Driver'],
        'employment': ['Employed', 'Self-Employed'],
        'prob': 0.25
    },
    'High School': {
        'jobs': ['Bank Staff', 'Receptionist', 'Sales Staff', 'Hotel Staff', 'Factory Supervisor'],
        'employment': ['Employed'],
        'prob': 0.08
    },
    'Vocational': {
        'jobs': ['Electrician', 'Plumber', 'Auto Mechanic', 'Beautician', 'Nurse'],
        'employment': ['Employed'],
        'prob': 0.04
    },
    'Bachelor': {
        'jobs': ['Teacher', 'Accountant', 'Engineer', 'Government Officer', 'NGO Worker'],
        'employment': ['Employed'],
        'prob': 0.025
    },
    'Master+': {
        'jobs': ['Doctor', 'Lawyer', 'University Lecturer', 'Bank Manager', 'Tech Specialist'],
        'employment': ['Employed'],
        'prob': 0.005
    }
}

# Updated income ranges (USD/month) with 2025 projections (NBC Inflation Report) - Narrowed ranges
INCOME_RANGES_USD = {
    # Informal Sector
    'Farmer': (85, 150), 
    'Fishery Worker': (90, 140),
    'Construction Laborer': (120, 180),
    'Construction Worker': (120, 180),
    'Domestic Worker': (100, 140),
    'Tuk-Tuk Driver': (150, 250),
    'Market Vendor': (130, 200),
    'Garment Worker': (160, 220),
    
    # Formal Sector
    'Shop Assistant': (180, 280),
    'Waitstaff': (180, 300),
    'Security Guard': (190, 300),
    'Taxi Driver': (200, 320),
    'Bank Staff': (300, 500),
    'Receptionist': (250, 400),
    'Sales Staff': (280, 450),
    'Hotel Staff': (250, 420),
    'Factory Supervisor': (350, 600),
    'Electrician': (300, 550),
    'Plumber': (300, 500),
    'Auto Mechanic': (320, 600),
    'Beautician': (250, 500),
    'Nurse': (350, 650),
    'Teacher': (380, 750),
    'Accountant': (500, 1000),
    'Engineer': (600, 1500),
    'Government Officer': (400, 800),
    'NGO Worker': (450, 1200),
    'Doctor': (900, 2500),
    'Lawyer': (800, 2000),
    'University Lecturer': (700, 1800),
    'Bank Manager': (1500, 4000),
    'Tech Specialist': (800, 2000)
}

LOAN_PURPOSES = [
    'Agriculture', 'Home Renovation', 'Medical', 'Debt Consolidation', 
    'Education', 'Business Expansion', 'Vehicle', 'Consumer Goods'
]

def _normalize_probs(probs):
    total = sum(probs)
    return [p / total for p in probs]
# Based on MLVT 2023 Report
JOB_LOAN_PURPOSE_MAP = {
    # Informal Sector
    'Farmer': {'purposes': LOAN_PURPOSES, 'probs': [0.4, 0.15, 0.15, 0.05, 0.05, 0.10, 0.05, 0.05]},
    'Fishery Worker': {'purposes': LOAN_PURPOSES, 'probs': [0.35, 0.15, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05]},
    'Construction Laborer': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.20, 0.15, 0.10, 0.05, 0.10, 0.10]},
    'Construction Worker': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.20, 0.15, 0.10, 0.05, 0.10, 0.10]},
    'Domestic Worker': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.25, 0.15, 0.15, 0.05, 0.05, 0.10]},
    'Tuk-Tuk Driver': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.15, 0.10, 0.15, 0.05, 0.15, 0.30, 0.05]},
    'Market Vendor': {'purposes': LOAN_PURPOSES, 'probs': [0.10, 0.15, 0.10, 0.10, 0.05, 0.35, 0.05, 0.10]},
    'Garment Worker': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.20, 0.10, 0.20, 0.05, 0.10, 0.10]},
    
    # Formal Sector
    'Shop Assistant': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05]},
    'Waitstaff': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05]},
    'Security Guard': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.20, 0.15, 0.15, 0.05, 0.10, 0.05]},
    'Taxi Driver': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.15, 0.15, 0.15, 0.05, 0.10, 0.30, 0.05]},
    'Bank Staff': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.10, 0.15, 0.15, 0.10, 0.20, 0.05]},
    'Receptionist': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.15, 0.20, 0.05, 0.15, 0.05]},
    'Sales Staff': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.10, 0.15, 0.15, 0.15, 0.05]},
    'Hotel Staff': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05]},
    'Factory Supervisor': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.15, 0.10, 0.15, 0.15, 0.10, 0.05]},
    'Electrician': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.10, 0.10, 0.25, 0.10, 0.05]},
    'Plumber': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.10, 0.10, 0.25, 0.10, 0.05]},
    'Auto Mechanic': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.15, 0.15, 0.10, 0.10, 0.30, 0.10, 0.05]},
    'Beautician': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.15, 0.10, 0.15, 0.15, 0.25, 0.10, 0.05]},
    'Nurse': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.15, 0.15, 0.20, 0.10, 0.10, 0.05]},
    'Teacher': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.20, 0.10, 0.15, 0.30, 0.05, 0.10, 0.05]},
    'Accountant': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.10, 0.15, 0.15, 0.10, 0.15, 0.05]},
    'Engineer': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.10, 0.10, 0.15, 0.15, 0.15, 0.05]},
    'Government Officer': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.15, 0.10, 0.15, 0.10, 0.15, 0.05]},
    'NGO Worker': {'purposes': LOAN_PURPOSES, 'probs': [0.10, 0.20, 0.15, 0.10, 0.20, 0.10, 0.10, 0.05]},
    'Doctor': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.30, 0.10, 0.05, 0.15, 0.15, 0.15, 0.05]},
    'Lawyer': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.30, 0.10, 0.05, 0.15, 0.15, 0.15, 0.05]},
    'University Lecturer': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.10, 0.10, 0.25, 0.10, 0.10, 0.05]},
    'Bank Manager': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.30, 0.10, 0.05, 0.15, 0.15, 0.15, 0.05]},
    'Tech Specialist': {'purposes': LOAN_PURPOSES, 'probs': [0.05, 0.25, 0.10, 0.10, 0.20, 0.15, 0.10, 0.05]},
    'default': {'purposes': LOAN_PURPOSES, 'probs': [0.15, 0.20, 0.10, 0.15, 0.10, 0.15, 0.10, 0.05]}
}

LOAN_AMOUNT_RANGE_USD = (50, 22000)  # Reduced upper limit
INTEREST_RATE_RANGE = (10, 18)
LOAN_TERM_RANGE_MONTHS = (1, 38)

def generate_cambodian_name(gender):
    male_first_names = [
        "Acharya", "Akra", "Amara", "Anchaly", "Arun", "Atith", "Bona", "Bora", "Boran", "Borey",
        "Bourey", "Bunroeun", "Chakara", "Chakra", "Chamroeun", "Chankrisna", "Chann", "Chanthou",
        "Chantrea", "Chanvatey", "Charya", "Chea", "Chhay", "Chhaya", "Dara", "Darany", "Davuth", "Devi",
        "Manson", "Virakbott", "Menghour", "Theasov", "Sopharypanavath", "Linton", "Yousin", "Nestar",
        "Chendara", "Ramom", "Saihong", "Chhimheng", "Sopanha", "Leab", "Sereybothtepy", "Tivoen",
        "Visal", "Satya", "Honghav", "Huot", "Penghuot", "Chanpanhavath", "Thornpunleu", "Mengthong",
        "Ehak", "Panhar", "Pich", "Sorany", "Uteytithya", "Pimol", "Sinh", "Lyhov", "Sengly", "Venhav",
        "Kotthrayothe", "Panha", "Phuravong", "Sithol", "Sovatanak", "Soksan", "Punloue", "Veasna",
        "Madarine", "Raksmey", "Pichphyrom", "Dolla", "Kokhout", "Sopereakline", "Sakphearith",
        "Sokheng", "Panharong", "Phourivath", "Seth", "Makara", "Sothy", "Srorn", "Sakura", "Pisey",
        "Sovichet", "Hoklin", "Seaklim"
    ]
    female_first_names = [
        "Achariya", "Akara", "Anchali", "Arunny", "Ary", "Bopha", "Botum", "Boupha", "Champei", "Champey",
        "Chamroeun", "Chan", "Chankrisna", "Chanlina", "Chanmony", "Channary", "Chanthavy", "Chantou", "Chantrea",
        "Chanvatey", "Chariya", "Chavy", "Chaya", "Chea", "Chenda", "Chhaiya", "Chhean", "Chhorvin", "Chhorvon",
        "Chivy", "Choum", "Da", "Daevy", "Dara", "Darareaksmey", "Davi", "Sophanny", "Kimsray", "Marina",
        "Livhoung", "Kimsoun", "Gechleang", "Sreysor", "Povrajana", "Phinnaroth", "Sreypov", "Thida",
        "Chanda", "Dalin", "Livita", "Mengly"
    ]
    last_names = [
        "Chan", "Kong", "Ly", "Heng", "Kim", "Seng", "Phan", "Vann", "Sok", "Rath",
        "Chea", "Chhun", "Chhim", "Chum", "Chhay", "Chhorn", "Chhoeun", "Chhunly", "Chenda", "Chanty",
        "Chhunhak", "Chhunny", "Chhaya", "Chhoun", "Chhoy", "Chhorvy", "Pheng", "Phirun", "Phalla", "Phirum",
        "Phors", "Phorn", "Chao", "Cheng", "Chhin", "Chork", "Choun", "Din", "Ear", "Hong", "Hou",
        "Hour", "Kao", "Keo", "Kho", "Khov", "Korn", "Lay", "May", "Long", "Lorn", "Lysan",
        "Mat", "Nangdy", "Narin", "Non", "Nuon", "Oem", "Oeurn", "Ornchann", "Ou", "Ouk",
        "Pa", "Pay", "Pech", "Pen", "Phoeun", "Phon", "Phork", "Pich", "Pov", "Ratana",
        "Ret", "Rin", "Ry", "Sa", "Sambo", "Saphorn", "Say", "Seang", "Sem", "Sin",
        "Srun", "Te", "Tep", "Thy", "Tip", "Touch", "Uch", "Vannak", "Chork", "Hing", "Rith", "Morm", "Mom"
    ]
    middle_names = [
        "Sokha", "Vanna", "Rithy", "Sophea", "Vuthy", "Sovann", "Srey", "Ratha", "Vichea", "Sok"
    ]

    if gender == "Male":
        first_name = random.choice(male_first_names)
    else:
        first_name = random.choice(female_first_names)
    last_name = random.choice(last_names)
    if random.random() < 0.3:
        middle_name = random.choice(middle_names)
        return f"{last_name} {middle_name} {first_name}"
    return f"{last_name} {first_name}"

def generate_demographics():
    gender = np.random.choice(['Male', 'Female'], p=GENDER_PROBS)
    age = int(np.random.normal(loc=35, scale=10))
    age = max(AGE_RANGE[0], min(AGE_RANGE[1], age))
    
    if age < 30:
        education_probs = [0.15, 0.30, 0.35, 0.15, 0.04, 0.015, 0.005]
    elif age < 50:
        education_probs = [0.25, 0.35, 0.25, 0.10, 0.03, 0.015, 0.005]
    else:
        education_probs = [0.30, 0.40, 0.20, 0.07, 0.02, 0.008, 0.002]
    total_prob = sum(education_probs)
    education_probs = [p / total_prob for p in education_probs]

    education = np.random.choice(
        list(EDUCATION_JOB_MAPPING.keys()),
        p=education_probs
    )
    province = np.random.choice(PROVINCES, p=PROVINCE_PROBS)
    
    if age < 25:
        marital_probs = [0.70, 0.25, 0.04, 0.01]
    elif age < 40:
        marital_probs = [0.10, 0.80, 0.07, 0.03]
    else:
        marital_probs = [0.05, 0.70, 0.15, 0.10]
    
    if np.random.choice(MARITAL_STATUS, p=marital_probs) == 'Married':
        dependents = min(DEPENDENTS_RANGE[1], max(DEPENDENTS_RANGE[0], np.random.poisson(2.0)))
    else:
        dependents = min(DEPENDENTS_RANGE[1], max(DEPENDENTS_RANGE[0], np.random.poisson(0.5)))

    return {
        'Name': generate_cambodian_name(gender),
        'Age': age,
        'Gender': gender,
        'Marital_Status': np.random.choice(MARITAL_STATUS, p=marital_probs),
        'Dependents': dependents,
        'Education_Level': education,
        'Province': province
    }

FORMAL_JOBS = [
    'Bank Staff', 'Receptionist', 'Sales Staff', 'Hotel Staff', 'Factory Supervisor',
    'Electrician', 'Plumber', 'Auto Mechanic', 'Beautician', 'Nurse', 'Teacher',
    'Accountant', 'Engineer', 'Government Officer', 'NGO Worker', 'Doctor', 'Lawyer',
    'University Lecturer', 'Bank Manager', 'Tech Specialist', 'Security Guard', 'Shop Assistant', 'Waitstaff', 'Taxi Driver'
]

def generate_employment(education_level, province, age):
    job_info = EDUCATION_JOB_MAPPING[education_level]
    job_title = random.choice(job_info['jobs'])
    
    if job_title in FORMAL_JOBS:
        employment_type = 'Employed'
    else:
        if province == 'Phnom Penh':
            employment_probs = [0.40, 0.60]
        else:
            employment_probs = [0.70, 0.30]
        available_types = job_info['employment']
        type_probs = [employment_probs[['Employed', 'Self-Employed'].index(t)] if t in available_types else 0 for t in ['Employed', 'Self-Employed']]
        type_probs = [p/sum(type_probs) for p in type_probs if p > 0]
        employment_type = np.random.choice([t for t in ['Employed', 'Self-Employed'] if t in available_types], p=type_probs)

    working_years = max(0, age - 18)
    if employment_type == 'Self-Employed':
        tenure = min(working_years, np.random.randint(3, 10))
    else:
        tenure = min(working_years, np.random.randint(5, 20))
    
    if job_title in ['Doctor', 'Lawyer', 'Engineer', 'Government Officer', 'Bank Manager']:
        tenure = min(working_years, max(5, np.random.randint(5, 15)))

    return {
        'Job_Title': job_title,
        'Employment_Type': employment_type,
        'Job_Tenure': max(1, tenure)
    }

def generate_financials(job_title, province, age, existing_loans):
    income_range = INCOME_RANGES_USD[job_title]
    median_income = (income_range[0] + income_range[1]) / 2
    base_income = np.random.lognormal(mean=np.log(median_income), sigma=0.15)
    base_income = np.clip(base_income, income_range[0], income_range[1])

    location_factor = 1.15 if province == 'Phnom Penh' else 1.0
    experience = max(0, age - 22)
    experience_factor = 1 + (experience * 0.01)
    monthly_income = base_income * location_factor * experience_factor
    monthly_income = np.clip(monthly_income, income_range[0], income_range[1] * 1.2)

    has_coapplicant = np.random.random() < 0.4
    coapplicant_income = np.random.uniform(0, monthly_income * 0.4) if has_coapplicant else 0

    savings_factor = np.random.uniform(0.05, 0.12)
    savings = monthly_income * 12 * savings_factor
    savings = np.clip(savings, 0, 10000)

    # Calculate credit score (300-850 range)
    base_score = 650
    late_payments = np.random.randint(0, 4) if existing_loans > 0 else 0
    late_payment_impact = late_payments * -75
    income_component = min(80, monthly_income * 0.08)
    age_component = min(70, age * 0.5)
    savings_component = min(50, savings * 0.001)
    credit_score = base_score + late_payment_impact + income_component + age_component + savings_component
    credit_score = int(np.clip(credit_score, 300, 850))

    # Calculate interest rate based on credit score
    if credit_score >= 750:
        interest_rate = np.random.uniform(10, 12)
    elif credit_score >= 650:
        interest_rate = np.random.uniform(12, 14)
    elif credit_score >= 550:
        interest_rate = np.random.uniform(14, 16)
    else:
        interest_rate = np.random.uniform(16, 18)

    debt_ratio = np.random.uniform(1.0, 2.0)
    loan_amount = (monthly_income + coapplicant_income) * debt_ratio * 12
    loan_amount = np.clip(loan_amount, LOAN_AMOUNT_RANGE_USD[0], LOAN_AMOUNT_RANGE_USD[1])

    loan_term = int(np.clip(np.random.weibull(1.5) * 10, LOAN_TERM_RANGE_MONTHS[0], LOAN_TERM_RANGE_MONTHS[1]))

    return {
        'Monthly_Income_USD': round(monthly_income, 2),
        'Coapplicant_Income_USD': round(coapplicant_income, 2),
        'Savings_Balance_USD': round(savings, 2),
        'Credit_Score': credit_score,
        'Loan_Amount_USD': round(loan_amount, 2),
        'Interest_Rate': round(interest_rate, 2),
        'Loan_Term_Months': loan_term
    }

def get_realistic_loan_purpose(job_title):
    purpose_info = JOB_LOAN_PURPOSE_MAP.get(job_title, JOB_LOAN_PURPOSE_MAP['default'])
    return np.random.choice(purpose_info['purposes'], p=_normalize_probs(purpose_info['probs']))

def generate_loan_applications(num_records=1000):
    records = []
    
    for _ in range(num_records):
        demo = generate_demographics()
        existing_loans = np.random.poisson(0.7)
        emp = generate_employment(demo['Education_Level'], demo['Province'], demo['Age'])
        financials = generate_financials(emp['Job_Title'], demo['Province'], demo['Age'], existing_loans)
        
        application_date = fake.date_between(start_date='-2y', end_date='today')
        
        record = {
            **demo,
            **emp,
            **financials,
            'Application_Date': application_date,
            'Loan_Purpose': get_realistic_loan_purpose(emp['Job_Title']),
            'Existing_Loans': existing_loans,
            'Late_Payments': np.random.poisson(0.3 * existing_loans),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    df['Total_Income_USD'] = df['Monthly_Income_USD'] + df['Coapplicant_Income_USD']
    df['Loan_to_Income_Ratio'] = df['Loan_Amount_USD'] / (df['Total_Income_USD'] * 12)
    
    return df

def get_ml_input_features():
    return [
        'Name', 'Age', 'Gender', 'Marital_Status', 'Dependents', 'Education_Level', 
        'Province', 'Job_Title', 'Employment_Type', 'Job_Tenure', 
        'Monthly_Income_USD', 'Coapplicant_Income_USD', 'Savings_Balance_USD', 
        'Credit_Score', 'Existing_Loans', 'Late_Payments',
        'Loan_Purpose', 'Loan_Term_Months'
    ]

if __name__ == "__main__":
    df = generate_loan_applications(100000)
    
    ml_features = get_ml_input_features() + ['Loan_Amount_USD', 'Interest_Rate']
    ml_df = df[ml_features]
    
    ml_df.to_csv('D:\CADT\InternshipI\Project\Digital-Loan\data\loan_data_100K.csv', index=False)
    print(f"Generated {len(ml_df)} records with {len(ml_features)} features")
    print("ML Features:", ml_features)