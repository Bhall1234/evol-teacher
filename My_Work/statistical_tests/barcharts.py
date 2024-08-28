import matplotlib.pyplot as plt
import os

def create_bar_chart(participants, perceived_difficulty, colors, title, output_file_path, y_max, y_label):
    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(participants, perceived_difficulty, color=colors)

    # Add labels and title
    plt.xlabel('Participant')
    plt.ylabel(y_label)  # Correctly set the y-axis label
    plt.title(title)
    plt.ylim(0, y_max)  # Set y-axis limit based on the scale

    # Save the chart
    plt.savefig(output_file_path)

    # Optionally display the chart
    plt.show()

def non_chatbot_data():
    # Data for existing questions
    participants = ['1', '2', '3', '4']
    perceived_difficulty = [2, 4, 4, 5]
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
    title = 'Perceived Task Difficulty'
    output_dir = 'My_Work/statistical_tests/questionnaire_1_group_a_b_combined_no_chatbot'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'perceived_task_difficulty.png')

    # Generate bar chart for existing data
    create_bar_chart(participants, perceived_difficulty, colors, title, output_file_path, 6, 'Perceived Task Difficulty')

    # Data for new questions
    q2_likert = [1, 4, 1, 2]
    q2_title = 'Perceived Satisfaction using non-interactive system'
    q2_output_file_path = os.path.join(output_dir, 'perceived_satisfaction.png')
    create_bar_chart(participants, q2_likert, colors, q2_title, q2_output_file_path, 6, 'Perceived Satisfaction')

    q3_likert = [4, 5, 3, 4]
    q3_title = 'Perceived Learning Effectiveness using non-interactive system'
    q3_output_file_path = os.path.join(output_dir, 'perceived_learning.png')
    create_bar_chart(participants, q3_likert, colors, q3_title, q3_output_file_path, 6, 'Perceived Learning Effectiveness')

    q4_likert = [3, 4, 4, 4]
    q4_title = 'Perceived Confidence Increase using non-interactive system'
    q4_output_file_path = os.path.join(output_dir, 'perceived_confidence.png')
    create_bar_chart(participants, q4_likert, colors, q4_title, q4_output_file_path, 6, 'Perceived Confidence Increase')

    q5_likert = [2, 3, 3, 3]
    q5_title = 'Perceived Task Challenge in isolation'
    q5_output_file_path = os.path.join(output_dir, 'perceived_challenge.png')
    create_bar_chart(participants, q5_likert, colors, q5_title, q5_output_file_path, 3.5, 'Perceived Task Challenge')

def chatbot_data():
    # Data for existing questions
    participants = ['1', '2', '3', '4']
    perceived_difficulty = [1, 1, 3, 3]
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
    title = 'Perceived Task Difficulty'
    output_dir = 'My_Work/statistical_tests/questionnaire_1_group_a_b_combined_chatbot'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'perceived_task_difficulty.png')

    # Generate bar chart for existing data
    create_bar_chart(participants, perceived_difficulty, colors, title, output_file_path, 6, 'Perceived Task Difficulty')

    # Data for new questions
    q2_likert = [5, 1, 5, 4]
    q2_title = 'Perceived Satisfaction using non-interactive system'
    q2_output_file_path = os.path.join(output_dir, 'perceived_satisfaction.png')
    create_bar_chart(participants, q2_likert, colors, q2_title, q2_output_file_path, 6, 'Perceived Satisfaction')

    q3_likert = [5, 5, 4, 4]
    q3_title = 'Perceived Learning Effectiveness using non-interactive system'
    q3_output_file_path = os.path.join(output_dir, 'perceived_learning.png')
    create_bar_chart(participants, q3_likert, colors, q3_title, q3_output_file_path, 6, 'Perceived Learning Effectiveness')

    q4_likert = [4, 4, 4, 4]
    q4_title = 'Perceived Confidence Increase using non-interactive system'
    q4_output_file_path = os.path.join(output_dir, 'perceived_confidence.png')
    create_bar_chart(participants, q4_likert, colors, q4_title, q4_output_file_path, 6, 'Perceived Confidence Increase')

    q5_likert = [1, 2, 3, 3]
    q5_title = 'Perceived Task Challenge in isolation'
    q5_output_file_path = os.path.join(output_dir, 'perceived_challenge.png')
    create_bar_chart(participants, q5_likert, colors, q5_title, q5_output_file_path, 3.5, 'Perceived Task Challenge')

    # chatbot assistance rating

chatbot_data()
