# writing-features-AI-papers
Code for "What do writing features tell us about AI papers?"

## Feature Extractor
**Requirements**  
- For the grammar error extractor (GECTOR) part: `transformers 2.x`  
- For the RST extractor part: the `feng-hirst-rst-parser`  
- For the remaining part: `transformers 3.x`  

**To extract the features**  
Please refer to the scripts in `scripts/20200830_feature_extractor/script_extract_features.sh`.  
The feature names by category is at [url](https://warm-snowball-public-datasets.s3.amazonaws.com/writing_features_ai_papers/df_ai_labeled.csv) (df_ai_labeled.csv, 618.4KB).  
A downloadable version of the extracted features is at [url](https://warm-snowball-public-datasets.s3.amazonaws.com/writing_features_ai_papers/features_v2_with_venue.csv) (features_v2_with_venue.csv, 1.0GB).  


## Venue Labels
- First use the regex matcher in `notebooks/20200826_journals_count.ipynb`. This exports to `df_ai.csv`  
- For each venue name, give a human label as either C or W. This results in `df_ai_labeled.csv`. We include this file in `data/df_ai_labeled.csv`  