# Tokyo Housing Database & Price Forecasting
This project analyzes Tokyo’s rental housing landscape to support strategic decisions at Student Mobilization, Inc. The process involves scraping over 1,200 listings from *SUUMO.jp*, storing and organizing the data in an SQLite database, and applying regression models to forecast rental prices based on features such as floor plan, area, and building size. The goal is to streamline housing logistics for new field staff by simplifying the search for affordable and well-located housing options.

## Key Questions
- Can we predict rental prices based on key features?
- How does proximity to train stations influence prices? 
- What numerical features are most correlated with rent?
- What is the 95% confidence interval for the mean rental price?
- Are new buildings consistently priced higher than old ones? 

## Methods Used
- Python (pandas, matplotlib, seaborn, OOP)
- Web scraping with BeautifulSoup (*SUUMO.jp*)
- SQLite for data warehousing, querying, cleaning, and feature engineering
- Exploratory Data Analysis (EDA)
- Multivariate linear regression modeling (sklearn)

## Key Insights
- The 95% confidence interval for the mean rent is ¥101,696 ± 2,408.
- Floor area shows the strongest correlation with rent, whereas station proximity exhibits little to no correlation.
- Building size is positively correlated with rent, while building age shows a negative correlation.
- A linear regression model using floor plan, area, building age, building size, and floor level explains 86.8% of the variance in rent (R² = 0.868).
- With the possibility of nonlinearity or missing features, a more robust model should be employed for future analyses. 


## Data Source Disclosure
All rental listing data was collected from *SUUMO.jp* via web scraping. The dataset includes publicly available information such as rental prices, fees, floor plans, apartment sizes, and train stations. This project respects the site’s terms of use, and the data is used solely for analysis and operational purposes.

## Files
- `tokyo_housing_analysis.ipynb`: Full Jupyter notebook with code, visualizations, and insights
- `tokyo_housing.csv`: Processed dataset of Tokyo housing metrics and features
