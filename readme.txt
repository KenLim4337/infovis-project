------------------------------------------
Minimum Required File Structure
------------------------------------------
- base-folder
    - assets
        - style.css
    - processed_data
        - business_category_counts.gzip
        - business_review_fused.gzip
    - yelp_data
        - yelp_academic_dataset_business.json
        - yelp_academic_dataset_checkin.json
        - yelp_academic_dataset_review.json
    - dashboard.py
    - data_processing.ipynb
    - data_processing.py
    - data_analysis.ipynb
    - data_analysis.py
    - data_visualization.ipynb
    - data_visualization.py
    - requirements.txt

------------------------------------------
File Descriptions & Instructions
------------------------------------------
- ../assets/style.css
    - Custom css styling for dashboard
    - Note: As custom CSS is delivered via the assets folder, it will only be applied when hosted live. Check the hosted link: https://finalproject-kmkthsxnqa-nn.a.run.app to confirm final styling.

- ../processed_data/
    - Files in this folder are outputs of data_processing.py or data_processing.ipynb
    - Data can be found here:
        - https://drive.google.com/file/d/1aHxtagTriFiu7t736Ied--JzehpMwPhl/view?usp=share_link
        - https://drive.google.com/file/d/1iBFOFKdMv2X_Pm8rOGyaMwgCw5scSM4n/view?usp=share_link

- ../yelp_data/
    - Raw Yelp data as acquired from https://www.yelp.com/dataset/download
    - Required to run data_processing.py and data_processing.ipynb

- ../dashboard.py
    - Code for dashboard
    - Make sure to adjust all directories to the relevant ones.

- ../data_processing.ipynb and ../data_processing.py
    - Code to process raw Yelp data.
    - Outputs 2 csv gzips, business_category_counts.gzip and business_review_fused.gzip
        - business_category_counts.gzip stores the counts of different categories
        - business_review_fused.gzip stores data for all 50k businesses as well as related data such as all reviews, checkins, star ratings, business hours etc.
    - Make sure to adjust all directories such as in the read_csv functions and to_csv functions to be correct and pointing towards existing folders.

- ../data_analysis.ipynb and ../data_analysis.py
    - Handles the analysis and visualization for items related to sections 6 through 13 of the dataset.
    - Make sure to adjust all directories such as read_csv functions to point towards the correct files/directories.

- ../data_visualization.ipynb and ../data_visualization.py
    - Visualizes the data visualization portion of the project (sections 14 and 15)
    - Make sure to adjust all directories such as read_csv functions to point towards the correct files/directories.
