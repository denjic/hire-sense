# Canonical categories for the Resume Analyzer
from __future__ import annotations

CANONICAL_JOB_CATEGORIES: tuple[str, ...] = (
    "Advocate",
    "Arts",
    "Automation Testing",
    "Blockchain",
    "Business Analyst",
    "Civil Engineer",
    "Data Science",
    "Database",
    "DevOps Engineer",
    "DotNet Developer",
    "ETL Developer",
    "Electrical Engineering",
    "Hadoop",
    "Health and fitness",
    "HR",
    "Java Developer",
    "Mechanical Engineer",
    "Network Security Engineer",
    "Operations Manager",
    "PMO",
    "Python Developer",
    "SAP Developer",
    "Sales",
    "Testing",
    "Web Designing",
)

# Map noisy labels from CSV variants to canonical names
CATEGORY_ALIASES: dict[str, str] = {
    "advocate": "Advocate",
    "arts": "Arts",
    "automation testing": "Automation Testing",
    "blockchain": "Blockchain",
    "business analyst": "Business Analyst",
    "civil engineer": "Civil Engineer",
    "data science": "Data Science",
    "database": "Database",
    "devops engineer": "DevOps Engineer",
    "dotnet developer": "DotNet Developer",
    "dot net developer": "DotNet Developer",
    ".net developer": "DotNet Developer",
    "etl developer": "ETL Developer",
    "electrical engineering": "Electrical Engineering",
    "electrical engineer": "Electrical Engineering",
    "hadoop": "Hadoop",
    "health and fitness": "Health and fitness",
    "health & fitness": "Health and fitness",
    "hr": "HR",
    "human resources": "HR",
    "java developer": "Java Developer",
    "mechanical engineer": "Mechanical Engineer",
    "network security engineer": "Network Security Engineer",
    "operations manager": "Operations Manager",
    "pmo": "PMO",
    "python developer": "Python Developer",
    "sap developer": "SAP Developer",
    "sales": "Sales",
    "testing": "Testing",
    "web designing": "Web Designing",
    "web design": "Web Designing",
}


def canonicalize_category(raw: str) -> str | None:
    """Return the canonical category name, or None if unknown."""
    key = " ".join(str(raw).strip().lower().split())
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    for canonical in CANONICAL_JOB_CATEGORIES:
        if canonical.lower() == key:
            return canonical
    return None
