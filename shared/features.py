"""
shared/features.py
===================
SINGLE SOURCE OF TRUTH for every feature in the Workplace Stress Prediction
System.  Every column name, allowed value, display label, question text, and
modifiable flag lives here.  Other modules IMPORT from this file — never
re-type category lists elsewhere.

Category orders match docs/Architecture.md section 4 exactly.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# 1.  Feature definitions
#     Each entry is a dict with:
#       name          – snake_case column name used internally
#       display_label – short human-readable label for charts / UI
#       question_text – full question shown in the form
#       categories    – allowed values in LOW → HIGH order (ordinal) or
#                       in listing order (nominal)
#       encoding      – "ordinal" or "onehot"
#       modifiable    – True if the user can change it in the What-If tool
# ---------------------------------------------------------------------------

FEATURES: List[Dict] = [
    # ── About you ──────────────────────────────────────────────────────
    {
        "name": "age_group",
        "display_label": "Age group",
        "question_text": "What is your age group?",
        "categories": ["22-28", "29-34", "35-40"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "job_level",
        "display_label": "Job level",
        "question_text": "What is your current job level?",
        "categories": ["Junior Level", "Mid-Manager Level", "Senior Level"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "gender",
        "display_label": "Gender",
        "question_text": "What is your gender?",
        "categories": ["Male", "Female", "Other", "Prefer not to say"],
        "encoding": "onehot",
        "modifiable": False,
    },
    {
        "name": "marital_status",
        "display_label": "Marital status",
        "question_text": "What is your marital status?",
        "categories": ["Single", "Married", "Divorced/Separated"],
        "encoding": "onehot",
        "modifiable": False,
    },
    {
        "name": "num_children",
        "display_label": "Number of children",
        "question_text": "How many children do you have?",
        "categories": ["0", "1", "2", "3+"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "num_dependents",
        "display_label": "Number of dependents",
        "question_text": "How many dependents do you have?",
        "categories": ["0", "1", "2", "3+"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    # ── Work ───────────────────────────────────────────────────────────
    {
        "name": "working_hours",
        "display_label": "Working hours",
        "question_text": "How many hours do you work per day?",
        "categories": ["Less than 8", "8-9", "10-11", "More than 11"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "overtime_frequency",
        "display_label": "Overtime frequency",
        "question_text": "How often do you work overtime?",
        "categories": ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "workload_level",
        "display_label": "Workload level",
        "question_text": "How would you rate your workload level?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "job_satisfaction",
        "display_label": "Job satisfaction",
        "question_text": "How satisfied are you with your job?",
        "categories": [
            "Very Dissatisfied",
            "Dissatisfied",
            "Neutral",
            "Satisfied",
            "Very Satisfied",
        ],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "career_growth_pressure",
        "display_label": "Career growth pressure",
        "question_text": "How much career growth pressure do you feel?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "job_security_concern",
        "display_label": "Job security concern",
        "question_text": "How concerned are you about job security?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "after_hours_communication",
        "display_label": "After-hours communication",
        "question_text": "How often do you receive work-related communication after office hours?",
        "categories": ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "thinks_about_work_outside",
        "display_label": "Thinks about work outside",
        "question_text": "How often do you think about work outside office hours?",
        "categories": ["Never", "Rarely", "Sometimes", "Often", "Always"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    # ── Family and money ───────────────────────────────────────────────
    {
        "name": "childcare_responsibility",
        "display_label": "Childcare responsibility",
        "question_text": "What is your level of childcare responsibility?",
        "categories": ["None", "Low", "Moderate", "High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "elderly_care_responsibility",
        "display_label": "Elderly care responsibility",
        "question_text": "What is your level of elderly care responsibility?",
        "categories": ["None", "Low", "Moderate", "High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "family_responsibility_pressure",
        "display_label": "Family responsibility pressure",
        "question_text": "How much pressure do you feel from family responsibilities?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "financial_pressure",
        "display_label": "Financial pressure",
        "question_text": "How much financial pressure do you feel?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "relationship_pressure",
        "display_label": "Relationship pressure",
        "question_text": "How much pressure do you feel from your marriage/relationship?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    {
        "name": "social_pressure",
        "display_label": "Social pressure",
        "question_text": "How much pressure do you feel from relatives/social circle?",
        "categories": ["Very Low", "Low", "Moderate", "High", "Very High"],
        "encoding": "ordinal",
        "modifiable": False,
    },
    # ── Lifestyle ──────────────────────────────────────────────────────
    {
        "name": "sleep_duration",
        "display_label": "Sleep duration",
        "question_text": "What is your average sleep duration per day?",
        "categories": [
            "Less than 5 hours",
            "5-6 hours",
            "6-7 hours",
            "7-8 hours",
            "More than 8 hours",
        ],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "sleep_quality",
        "display_label": "Sleep quality",
        "question_text": "How would you rate your sleep quality?",
        "categories": ["Very Poor", "Poor", "Average", "Good", "Excellent"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "exercise_frequency",
        "display_label": "Exercise frequency",
        "question_text": "How often do you exercise or do physical activity?",
        "categories": ["Never", "1-2 days/week", "3-4 days/week", "5+ days/week"],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "screen_time",
        "display_label": "Screen time",
        "question_text": "How much non-work screen time do you have daily?",
        "categories": [
            "Less than 1 hour",
            "1-3 hours",
            "3-5 hours",
            "More than 5 hours",
        ],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "leisure_time",
        "display_label": "Leisure time",
        "question_text": "How much leisure/relaxation time do you have daily?",
        "categories": [
            "Less than 1 hour/day",
            "1-2 hours/day",
            "2-3 hours/day",
            "More than 3 hours/day",
        ],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "commute_time",
        "display_label": "Commute time",
        "question_text": "What is your daily commute time?",
        "categories": [
            "Less than 30 minutes",
            "30-60 minutes",
            "1-2 hours",
            "More than 2 hours",
        ],
        "encoding": "ordinal",
        "modifiable": True,
    },
    {
        "name": "work_life_balance",
        "display_label": "Work-life balance",
        "question_text": "How would you rate your work-life balance?",
        "categories": ["Very Poor", "Poor", "Average", "Good", "Excellent"],
        "encoding": "ordinal",
        "modifiable": True,
    },
]


# ---------------------------------------------------------------------------
# 2.  Target variable
# ---------------------------------------------------------------------------

TARGET = {
    "name": "stress_level",
    "display_label": "Overall stress level",
    "question_text": "What is your overall stress level?",
    "categories": ["Low", "Medium", "High"],  # label-encode → 0, 1, 2
}


# ---------------------------------------------------------------------------
# 3.  Convenience look-ups (derived from the lists above, never edit directly)
# ---------------------------------------------------------------------------

# Column name mapping:  original Google Forms header (stripped) → snake_case
# See docs/DATA_NOTES.md section 3 for the full mapping.
COLUMN_RENAME_MAP: Dict[str, str] = {
    "Age Group": "age_group",
    "Current Job Level": "job_level",
    "Gender": "gender",
    "Marital Status": "marital_status",
    "Number of Children": "num_children",
    "Number of Dependents": "num_dependents",
    "Working Hours Per Day": "working_hours",
    "Overtime Frequency": "overtime_frequency",
    "Workload Level": "workload_level",
    "Job Satisfaction": "job_satisfaction",
    "Career Growth Pressure": "career_growth_pressure",
    "Job Security Concern": "job_security_concern",
    "Work-related Communication After Office Hours": "after_hours_communication",
    "How often do you think about work outside office hours?": "thinks_about_work_outside",
    "Childcare Responsibilities": "childcare_responsibility",
    "Elderly Care Responsibilities": "elderly_care_responsibility",
    "Family Responsibility Pressure": "family_responsibility_pressure",
    "Financial Pressure": "financial_pressure",
    "Marriage/Relationship Pressure": "relationship_pressure",
    "Relative/Social Pressure": "social_pressure",
    "Average Sleep Duration Per Day": "sleep_duration",
    "Sleep Quality": "sleep_quality",
    "Exercise/Physical Activity": "exercise_frequency",
    "Daily Non-Work Screen Time": "screen_time",
    "Leisure/Relaxation Time": "leisure_time",
    "Daily Commute Time": "commute_time",
    "Work-Life Balance": "work_life_balance",
    "Overall Stress Level": "stress_level",
}

# Ordered list of feature names (for DataFrame column ordering)
FEATURE_NAMES: List[str] = [f["name"] for f in FEATURES]

# Quick access: feature name → its definition dict
FEATURE_MAP: Dict[str, Dict] = {f["name"]: f for f in FEATURES}

# Allowed values per feature (for validation)
ALLOWED_VALUES: Dict[str, List[str]] = {f["name"]: f["categories"] for f in FEATURES}
ALLOWED_VALUES[TARGET["name"]] = TARGET["categories"]

# Ordinal features and their category orders (for OrdinalEncoder)
ORDINAL_FEATURES: List[str] = [f["name"] for f in FEATURES if f["encoding"] == "ordinal"]
ORDINAL_CATEGORIES: List[List[str]] = [
    f["categories"] for f in FEATURES if f["encoding"] == "ordinal"
]

# Nominal features (for OneHotEncoder)
NOMINAL_FEATURES: List[str] = [f["name"] for f in FEATURES if f["encoding"] == "onehot"]

# Modifiable features (for the What-If simulator)
MODIFIABLE_FEATURES: List[str] = [f["name"] for f in FEATURES if f["modifiable"]]

# Non-modifiable features
NON_MODIFIABLE_FEATURES: List[str] = [f["name"] for f in FEATURES if not f["modifiable"]]
