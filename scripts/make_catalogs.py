import pandas as pd
from requests import get
import sys
sys.path.append("../")
from src.classification import get_match_label_advanced
import os
# variables
INPUT_FULL = "nway_csc21_gaia3_full_neg_study_dis_niter200.parquet"     # your master list
ML_THRESH    = 0.466          # p_match_ind threshold
NWAY_THRESH  = 0.5            # p_any threshold
OUTPUT_FULL  = "csc_gaia_full_with_flags.csv"
OUTPUT_PRIMARY = "csc_gaia_primary_counterparts.csv"
OUTPUT_COLS = [
    'csc21_name', 'csc21_ra', 'csc21_dec', 'min_theta_mean',
    'gaia3_source_id', 'gaia3_ra', 'gaia3_dec', 'p_i', 'p_any', 
    'p_match_ind', 'separation', 'match_flag', 
    'flag_nway_confident', 'flag_ml_confident', 'flag_sep_ok'
    ]

path = "../catalogs/"
# check if path exists
if not os.path.exists(path):
    # if not, create it
    os.makedirs(path)
output_full = path + OUTPUT_FULL
output_primary = path + OUTPUT_PRIMARY

# load full candidate catalog
full = pd.read_parquet(INPUT_FULL)

# compute boolean flags
full['flag_nway_confident'] = full['p_any'] >= NWAY_THRESH
full['flag_ml_confident']   = full['p_match_ind'] >= ML_THRESH
full['flag_sep_ok']         = get_match_label_advanced(full, p_threshold=ML_THRESH)['flag_sep_ok']

# save full catalog with flags
full.to_csv(output_full, index=False, columns=OUTPUT_COLS)
print(f"Wrote full catalog with flags to {output_full}")

# quality_label:
#    0 = unlikely match
#    1 = ambiguous
#    2 = likely a match
#    3 = confident match
full['quality_label'] = 0
# ambiguous
mask = full['flag_nway_confident'] & ~(full['flag_ml_confident'] & full['flag_sep_ok'])
full.loc[mask, 'quality_label'] = 1
# likely (ml + sep, but not nway)
mask = full['flag_ml_confident'] & full['flag_sep_ok'] & ~full['flag_nway_confident']
full.loc[mask, 'quality_label'] = 2
# confident (all three)
mask = full['flag_nway_confident'] & full['flag_ml_confident'] & full['flag_sep_ok']
full.loc[mask, 'quality_label'] = 3

# select primary candidates
primary = (
    full
    .sort_values(['csc21_name', 'quality_label', 'p_i'], ascending=[True, False, False])
    .groupby('csc21_name', as_index=False)
    .first()   # picks the top row per group
)

# Save primary catalog
primary.to_csv(output_primary, index=False, columns=OUTPUT_COLS + ['quality_label'])
print(f"Wrote primary catalog to {output_primary}")
