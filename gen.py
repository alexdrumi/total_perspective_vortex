# Define the range of subjects and recordings
start_subject = 1
end_subject = 109
recordings = ["04", "08", "12"]

# Initialize a list to store the file paths
file_paths = []

# Iterate through each subject
for s in range(start_subject, end_subject + 1):
	subject_str = f"S{s:03}"  # Formats the subject number with leading zeros (e.g., S001)
	# Iterate through each recording for the current subject
	for r in recordings:
		# Construct the file path
		file_path = f'"../data/{subject_str}/{subject_str}R{r}.edf",'
		# Append the formatted file path to the list
		file_paths.append(f'\t{file_path}')

# Optionally, print all file paths
for path in file_paths:
	print(path)

# If you want to store them as a Python list, you can do the following:
# print("[")
# for path in file_paths:
#     print(path)
# print("]")

