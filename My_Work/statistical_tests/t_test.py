from scipy import stats

non_interactive_scores = [4, 4, 3, 4]
interactive_scores = [4, 2, 4, 2]

t_statistic, p_value = stats.ttest_rel(non_interactive_scores, interactive_scores)
print(f"T-statistic: {t_statistic}, P-value: {p_value}")

# Output:
# T-statistic: 1.0, P-value: 0.39100221895577053
