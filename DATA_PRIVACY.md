Data Privacy & Anonymization

This repository contains a sanitized version of client data used for a Proof of Concept (POC) sales analytics dashboard.

To protect client confidentiality and comply with responsible data handling practices, all datasets have been anonymized before being uploaded.

**What Was Masked**

The following fields were transformed:

1. Personal Identifiable Information (PII)

Contact names → Replaced with synthetic labels (e.g., Contact 0001)

Sales consultants / agents → Replaced with generic IDs (e.g., Agent 01)

Emails → Replaced with synthetic email addresses

Phone numbers → Replaced with synthetic phone numbers

Addresses / suburbs → Replaced with generated addresses

Internal record IDs → Hashed values

2. Free-Text Fields

Notes, comments, and descriptions → Replaced with redacted placeholders

3. Financial Values (SPV / Revenue)

Sales Purchase Values were numerically obfuscated using deterministic scaling.

Relative performance structure is preserved.

Absolute financial values are not real.

4. Geographic Data

Postcodes were generalized while preserving state-level grouping.

**What Was Preserved**

To ensure the dashboard remains functionally accurate:

Row counts

Data structure

Column names

Consultant distribution

Conversion ratios

Revenue ordering patterns

This allows the analytics logic, KPIs, and visualizations to function correctly without exposing real business data.

**Purpose**

The anonymized datasets are provided solely for:

Demonstrating technical implementation

Showcasing data cleaning logic

Displaying dashboard functionality

Portfolio and educational purposes

They do not reflect actual client data.

**Responsible Data Handling**

No confidential or commercially sensitive information from the client is publicly shared in this repository.
