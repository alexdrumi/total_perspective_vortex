# Define the range of subjects and the specific runs you want to include
subjects = range(60, 80)  # S001 to S050
runs = ['06', '10', '14']

# Initialize an empty list to hold the file paths
file_list = []

# Loop through each subject and each run to create the file paths
for s in subjects:
    subject_id = f"S{s:03}"  # Formats the subject number with leading zeros, e.g., S001
    for r in runs:
        file_path = f'"../data/{subject_id}/{subject_id}R{r}.edf",'
        file_list.append(file_path)

# Optionally, print the list in a formatted way
for file in file_list:
    print(f"\t{file}")

