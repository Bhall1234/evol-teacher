import scipy.stats as stats

# data
group_a_interactive_scores = [4, 4] 
group_b_interactive_scores = [2, 2]  

# Perform the test for comparing interactive system scores between Group A and Group B
stat, p_value = stats.mannwhitneyu(group_a_interactive_scores, group_b_interactive_scores)
print('Mann-Whitney U Test (Interactive System): Stat=%.3f, p=%.3f' % (stat, p_value))


# non-interactive system scores
group_a_non_interactive_scores = [4, 3]  
group_b_non_interactive_scores = [4, 4] 

# Perform the test for comparing non-interactive system scores between Group A and Group B
stat_non, p_value_non = stats.mannwhitneyu(group_a_non_interactive_scores, group_b_non_interactive_scores)
print('Mann-Whitney U Test (Non-Interactive System): Stat=%.3f, p=%.3f' % (stat_non, p_value_non))

# Mann-Whitney U Test (Interactive System): Stat=4.000, p=0.194
# Mann-Whitney U Test (Non-Interactive System): Stat=1.000, p=0.617
