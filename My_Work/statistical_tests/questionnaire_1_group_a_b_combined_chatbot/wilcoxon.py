import scipy.stats as stats

# Data from Group A
group_a_interactive_scores = [4, 4]  # scores for Group A after interactive system
group_a_non_interactive_scores = [4, 3]  # scores for Group A after non-interactive system

# Perform the test for Group A
stat_a, p_value_a = stats.wilcoxon(group_a_interactive_scores, group_a_non_interactive_scores)
print('Group A Wilcoxon Test: Stat=%.3f, p=%.3f' % (stat_a, p_value_a))

# Group B with their respective scores
group_b_interactive_scores = [2, 2]  # scores for Group A after interactive system
group_b_non_interactive_scores = [4, 4]  # scores for Group A after non-interactive system

# Perform the test for Group B
stat_b, p_value_b = stats.wilcoxon(group_b_interactive_scores, group_b_non_interactive_scores)
print('Group B Wilcoxon Test: Stat=%.3f, p=%.3f' % (stat_b, p_value_b))
