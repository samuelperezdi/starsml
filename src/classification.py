from math import sqrt, ceil

def get_match_label_simple(df, p_threshold=0.35):
    """get match labels for sources meeting position and probability criteria"""

    df_in_crit = df.copy()
    
    # position criteria mask
    pos_mask = (
        ((df_in_crit['min_theta_mean'] < 3) & (df_in_crit['separation'] <= 1.3)) |
        ((df_in_crit['min_theta_mean'].between(3, 6)) & (df_in_crit['separation'] <= 1.3)) |
        ((df_in_crit['min_theta_mean'] > 6) & (df_in_crit['separation'] <= 2.2))
    )
    
    # combined criteria mask
    mask = pos_mask
    
    # set labels 
    df_in_crit.loc[:, 'label'] = 0
    df_in_crit.loc[mask & (df_in_crit['p_match_ind'] > p_threshold), 'label'] = 1
    
    return df_in_crit

def get_match_label_advanced(df, p_threshold=0.35):
    """
    label matches using dynamic separation thresholds computed from 
    off-axis angle and error components, plus a minimum ML match score.
    df has columns:
      - 'min_theta_mean': off-axis angle in arcmin
      - 'separation': candidate separation in arcsec
      - 'p_match_ind': ML match score
    """

    # sigma_psf from off-axis angle (theta in arcmin)
    def compute_psf_sigma(theta):
        # 90% ECF radius parameterization (from pog fig 4.12)
        R_ecf90 = 1.1 + 0.05 * theta + 0.1 * (theta ** 2)
        return R_ecf90 / 2.15

    # total expected positional error from all contributions (arcsec)
    def compute_total_error(theta):
        sigma_psf = compute_psf_sigma(theta)
        return sqrt(0.1**2 + 0.5**2 + (sigma_psf / 3)**2 + sigma_psf**2)
    
    # dynamic threshold separation (arcsec)
    def compute_threshold(theta):
        tot_err = compute_total_error(theta)
        th = ceil(tot_err + 0.5)
        # for theta<3, enforce 1.5 arcsec threshold
        if theta < 3:
            th = max(th, 1.5)
        # cap the threshold at 10 arcsec
        th = min(th, 10)
        return th
    
    # Work on a copy of the DataFrame
    df_new = df.copy()
    
    # Compute the threshold separation for each source based on its off-axis angle
    df_new['threshold_sep'] = df_new['min_theta_mean'].apply(compute_threshold)
    
    # Build a mask that requires candidate separation to be within threshold
    pos_mask = df_new['separation'] <= df_new['threshold_sep']
    
    # A source is labeled as a match (label = 1) if both the position and ML score criteria are met.
    df_new['label'] = 0
    df_new.loc[pos_mask & (df_new['p_match_ind'] > p_threshold), 'label'] = 1
    
    return df_new
