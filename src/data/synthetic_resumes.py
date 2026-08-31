"""Synthetic resume fallback when the dataset is unavailable."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.constants import CANONICAL_JOB_CATEGORIES

logger = logging.getLogger(__name__)

SKILLS: dict[str, list[str]] = {
    "Advocate": [
        "civil procedure", "legal drafting", "client counseling", "case research", "court appearances",
    ],
    "Arts": [
        "drawing", "color theory", "exhibition planning", "studio teaching", "portfolio development",
    ],
    "Automation Testing": [
        "Selenium", "Playwright", "API testing", "CI pipelines", "test reporting",
    ],
    "Blockchain": [
        "Solidity", "Hardhat", "EVM", "wallet UX", "smart-contract testing",
    ],
    "Business Analyst": [
        "user stories", "process mapping", "SQL validation", "UAT", "stakeholder workshops",
    ],
    "Civil Engineer": [
        "site supervision", "AutoCAD", "quantity takeoff", "IS codes", "quality checklists",
    ],
    "Data Science": [
        "Python", "pandas", "scikit-learn", "SQL", "experiment design",
    ],
    "Database": [
        "PostgreSQL", "query tuning", "backups", "indexing", "schema design",
    ],
    "DevOps Engineer": [
        "CI/CD", "Docker", "Kubernetes", "Terraform", "observability",
    ],
    "DotNet Developer": [
        "C#", "ASP.NET Core", "Entity Framework", "SQL Server", "REST APIs",
    ],
    "ETL Developer": [
        "SQL", "Informatica", "data quality", "scheduling", "source-to-target mapping",
    ],
    "Electrical Engineering": [
        "SLDs", "protection", "commissioning", "IS/IEC standards", "panel design",
    ],
    "Hadoop": [
        "Spark", "Hive", "HDFS", "PySpark", "partitioning",
    ],
    "Health and fitness": [
        "program design", "movement coaching", "client assessment", "group classes", "CPR/AED",
    ],
    "HR": [
        "recruiting", "onboarding", "employee relations", "HRIS", "policy communication",
    ],
    "Java Developer": [
        "Java", "Spring Boot", "JPA", "REST", "JUnit",
    ],
    "Mechanical Engineer": [
        "CAD", "GD&T", "DFM", "prototyping", "manufacturing processes",
    ],
    "Network Security Engineer": [
        "firewalls", "VPN", "TCP/IP", "segmentation", "incident response",
    ],
    "Operations Manager": [
        "SLA ownership", "vendor management", "capacity planning", "incident command", "process improvement",
    ],
    "PMO": [
        "RAID logs", "status reporting", "stakeholder management", "change control", "portfolio tracking",
    ],
    "Python Developer": [
        "Python", "FastAPI", "pytest", "PostgreSQL", "Redis",
    ],
    "SAP Developer": [
        "ABAP", "RICEF", "S/4HANA", "interfaces", "transport management",
    ],
    "Sales": [
        "discovery", "pipeline management", "CRM", "closing", "account expansion",
    ],
    "Testing": [
        "test cases", "exploratory testing", "defect reporting", "regression", "API checks",
    ],
    "Web Designing": [
        "Figma", "responsive layout", "HTML/CSS", "accessibility", "design systems",
    ],
}

ROLES: dict[str, list[str]] = {
    "Advocate": ["Litigation Associate", "Corporate Counsel", "Legal Advisor"],
    "Arts": ["Arts Instructor", "Exhibition Coordinator", "Creative Producer"],
    "Automation Testing": ["SDET", "QA Automation Engineer", "Mobile Automation Tester"],
    "Blockchain": ["Smart Contract Engineer", "Blockchain Backend Developer", "Web3 Engineer"],
    "Business Analyst": ["IT Business Analyst", "Finance BA", "Agile Business Analyst"],
    "Civil Engineer": ["Site Engineer", "Structural Designer", "Estimation Engineer"],
    "Data Science": ["Data Scientist", "ML Engineer", "Analytics Scientist"],
    "Database": ["Database Administrator", "SQL Developer", "Data Platform Engineer"],
    "DevOps Engineer": ["DevOps Engineer", "SRE", "Platform Engineer"],
    "DotNet Developer": ["ASP.NET Developer", "Full-stack .NET Engineer", "C# Engineer"],
    "ETL Developer": ["ETL Developer", "Data Engineer", "Warehouse Developer"],
    "Electrical Engineering": ["Electrical Designer", "Field Electrical Engineer", "Controls Engineer"],
    "Hadoop": ["Hadoop Developer", "Big Data Engineer", "Spark Engineer"],
    "Health and fitness": ["Personal Trainer", "Wellness Coordinator", "S&C Coach"],
    "HR": ["HR Generalist", "Talent Acquisition Specialist", "HR Operations Associate"],
    "Java Developer": ["Java Backend Developer", "Spring Boot Engineer", "Microservices Engineer"],
    "Mechanical Engineer": ["Mechanical Designer", "Manufacturing Engineer", "HVAC Engineer"],
    "Network Security Engineer": ["Network Security Engineer", "SOC Engineer", "Security Analyst"],
    "Operations Manager": ["Operations Manager", "Facility Operations Lead", "Business Operations Manager"],
    "PMO": ["PMO Analyst", "Project Manager", "PMO Coordinator"],
    "Python Developer": ["Python Backend Developer", "Platform Engineer", "Full-stack Python Developer"],
    "SAP Developer": ["ABAP Developer", "SAP Consultant", "SAP Integration Developer"],
    "Sales": ["B2B Sales Executive", "SDR", "Account Manager"],
    "Testing": ["Manual QA Engineer", "QA Analyst", "Domain Tester"],
    "Web Designing": ["UI Designer", "Web Designer", "UX/UI Designer"],
}

EXPERIENCE_SNIPPETS: tuple[str, ...] = (
    "Led a cross-functional effort that reduced cycle time by documenting handoffs and adding a weekly risk review.",
    "Owned a production workflow end-to-end, including monitoring, incident notes, and a written runbook for the next person.",
    "Partnered with stakeholders to turn ambiguous requests into testable acceptance criteria and a sequenced delivery plan.",
    "Improved quality by adding checks before release and coaching peers on how to reproduce failures with evidence.",
    "Delivered a scoped project on a deadline by cutting non-essential work and keeping a transparent status log.",
    "Mentored a junior teammate through code/work reviews focused on correctness, not style nitpicks.",
)


def _resume_text(category: str, index: int, years: int) -> str:
    role = ROLES[category][index % len(ROLES[category])]
    skills = SKILLS[category]
    rotated = skills[index % len(skills):] + skills[: index % len(skills)]
    skill_line = ", ".join(rotated)
    snippet = EXPERIENCE_SNIPPETS[index % len(EXPERIENCE_SNIPPETS)]
    education = (
        "B.Tech" if index % 3 else "M.Sc" if index % 3 == 1 else "Bachelor's"
    )
    return (
        f"{role} with {years} years of experience in {category}. "
        f"Core skills: {skill_line}. {snippet} "
        f"Comfortable writing clear documentation, working with stakeholders, and owning outcomes. "
        f"Education: {education} relevant to {category}. "
        f"Seeking a {role} role where {category.lower()} depth is used daily, not as a keyword list."
    )


def generate_synthetic_resumes(resumes_per_category: int) -> pd.DataFrame:
    if resumes_per_category < 3:
        raise ValueError(
            "Need at least 3 synthetic resumes per category for an 80/10/10 split")

    rows: list[dict[str, str]] = []
    seq = 1
    for category in CANONICAL_JOB_CATEGORIES:
        for i in range(resumes_per_category):
            years = 2 + (i % 8)
            rows.append(
                {
                    "resume_id": f"R{seq:05d}",
                    "category": category,
                    "resume_text": _resume_text(category, i, years),
                }
            )
            seq += 1

    df = pd.DataFrame(rows)
    logger.warning(
        "Using synthetic fallback: %d resumes (%d per category). "
        "Place the Kaggle CSV in data/raw/ for the real dataset.",
        len(df),
        resumes_per_category,
    )
    return df
