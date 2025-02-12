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