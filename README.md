# SM-Dashboard
Client: iBuild Solutions
Tech Stack: Python, Streamlit, Pandas, Plotly
Type: Internal Analytics Tool (POC)
Link: https://sm-dashboard-ibuild.streamlit.app/

1. Project Background
iBuild Solutions was tracking their sales and marketing performance using Google Sheets. The process was manual and time-consuming. Each report required:
•	Cleaning CSV exports from CRM
•	Combining multiple sheets
•	Manually calculating conversions
•	Creating summary tables
•	Comparing consultant performance
There was no centralized dashboard and no automated performance view.
The company also did not want to invest in enterprise BI tools like Power BI.
I built a lightweight, automated dashboard using Python and Streamlit to solve this problem.
 
2. Problem Statement
The business needed answers to key operational questions:
•	How many leads are assigned to each consultant?
•	How many leads convert to contacts?
•	How many contacts convert to purchase agreements (PAs)?
•	Who generates the highest revenue?
•	Which states perform best?
•	Where is the sales funnel breaking?
These insights were previously calculated manually.
 
3. Solution Overview
I developed a Sales & Marketing Dashboard that:
•	Accepts CRM CSV exports (Leads, Contacts, PA files)
•	Automatically cleans and standardizes data
•	Calculates performance metrics
•	Visualizes consultant-level and team-level performance
•	Allows downloadable reports
The dashboard works with inconsistent column names and messy inputs, reducing dependency on perfect data formatting.
 
4. Key Features
1. Consultant Performance Summary
The dashboard calculates for each consultant:
•	Leads Allocated
•	Contact Conversions
•	PA Conversions
•	Total SPV (Sales Purchase Value)
Consultants are automatically ranked by performance.
This allows management to quickly identify top performers and underperformers.
 
2. Conversion Funnel Analysis
The system tracks full funnel performance:
•	Leads → Contacts
•	Contacts → PAs
•	Leads → PAs
Features include:
•	Individual consultant funnel view
•	Team-level aggregate funnel
•	Conversion rate comparison across consultants
This helps identify where conversions drop and where coaching may be needed.
 
3. Revenue Performance (SPV Tracking)
Revenue is calculated using SPV values from PA files.
The system:
•	Automatically detects SPV columns
•	Cleans currency formatting
•	Handles GST variations
•	Aggregates total SPV per consultant
Visualizations include:
•	Revenue ranking
•	Bubble chart (Volume vs Conversion vs Revenue)
•	Toggle between conversion-based and revenue-based views
This shifts focus from activity metrics to revenue impact.
 
4. Geographic Performance
Using postcode mapping, the dashboard:
•	Automatically assigns Australian states
•	Shows Leads by State
•	Shows PA Conversions by State
This supports regional strategy and marketing allocation decisions.
 
5. Data Standardization & Automation
One of the biggest challenges was inconsistent CRM exports.
To solve this, I implemented:
•	Automatic column detection
•	Name normalization (e.g., first name → full name)
•	Manual alias mapping for duplicates
•	Currency parsing and cleaning
•	File validation checks
•	Downloadable cleaned outputs
This reduced manual correction work significantly.
 
5. Business Impact
The dashboard replaced manual spreadsheet reporting and provided:
•	Real-time consultant performance tracking
•	Clear funnel visibility
•	Revenue-focused performance evaluation
•	Regional performance insights
•	Standardized reporting format
Management can now:
•	Compare consultants objectively
•	Identify conversion gaps
•	Track revenue contribution
•	Make faster operational decisions
All without requiring enterprise BI tools.
 
6. Technical Implementation
Core components include:
•	Pandas for cleaning, transformation, aggregation
•	Streamlit for interactive UI
•	Plotly for dynamic visualizations
•	Conversion rate calculations using vectorized operations
•	Robust merging logic to handle missing values
•	Flexible schema detection for CRM exports
The application is lightweight, deployable, and does not require a database.
 
7. Challenges & Solutions
Challenge 1: Inconsistent Column Names
Solution: Built dynamic column detection using candidate matching.
Challenge 2: Name Variations
Solution: Created a name canonicalization system with manual alias overrides.
Challenge 3: Dirty Currency Data
Solution: Implemented regex-based SPV cleaning and numeric coercion.
Challenge 4: Missing State Data
Solution: Inferred state from postcode ranges automatically.


