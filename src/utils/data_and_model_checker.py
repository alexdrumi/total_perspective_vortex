import os

def check_data():
	"""
	Checks that all EDF files for subjects S001-S109 and runs 1-14 exist under the expected directory structure.

	Example structure:
		- ../../data/S022/S022R06.edf

	Returns:
		None
	"""

	base_path = "../../data"
	missing_files = []
	total_files = 0

	for subj_id in range(1, 110):  #S001 to S109
		#Format subject string (eg: S001, ..., S109)
		subj_str = f"S{subj_id:03d}"
		
		for run in range(1, 15):   #runs 1-14
			run_str = f"{run:02d}"   #zero padded run 01, 02...
			edf_filename = f"{subj_str}R{run_str}.edf"
			edf_path = os.path.join(base_path, subj_str, edf_filename)
			
			total_files += 1
			#check if file exists
			if not os.path.isfile(edf_path):
				missing_files.append(edf_path)

	if missing_files:
		print("\nMissing EDF files:\n")
		print(f"\nTotal missing: {len(missing_files)} out of {total_files}")
		raise ValueError(f"Data files are missing.")

def check_models():
	"""
	Checks that all model .joblib files for 6 experiments exist under the expected directory structure.

	Example structure:
		- ../../models/pipe_runs_6_10_14.joblib

	Returns:
		None
	"""

	base_path = "../../models"
	missing_files = []
	total_files = 0
	experiment_run_groups = [
			[1],
			[2],
			[3, 7, 11],
			[4, 8, 12],
			[5, 9, 13],
			[6, 10, 14]
		]

	missing_files = []
	total_files = 0

	for runs in experiment_run_groups:
		#key for the filename, eg:. "runs_3_7_11"
		group_key = f"runs_{'_'.join(map(str, runs))}"
		model_filename = f"pipe_{group_key}.joblib"
		model_path = os.path.join(base_path, model_filename)
		total_files += 1
		#check if file exists
		if not os.path.isfile(model_path):
			missing_files.append(model_path)

	if missing_files:
		print("\nMissing model files:\n")
		print(f"\nTotal missing: {len(missing_files)} out of {total_files}")
		raise ValueError("Model files are missing. Please ensure all 6 .joblib models exist.") #why is it printing in 2 places?
