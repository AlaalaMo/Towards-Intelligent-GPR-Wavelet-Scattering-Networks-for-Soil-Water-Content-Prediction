import subprocess
import time
import os
import argparse


def has_room_for_jobs(username, max_job_count=100):
    time.sleep(.1)
    squeue_output = subprocess.run(["squeue", "-u", username], capture_output=True, text=True)
    output_lines = squeue_output.stdout.strip().split("\n")
    job_count = len(output_lines) - 1  # Exclude header line
    resource_limit_reached = False
    for line in output_lines[1:]:
        if 'Resources' in line:
            resource_limit_reached = False
            break
        elif 'QOSMAX' in line:
            resource_limit_reached = True
            break

    can_submit_more_jobs = (job_count < max_job_count) and not resource_limit_reached
    print('we submitting?',can_submit_more_jobs)
    return can_submit_more_jobs
def submit_jobs(model_type, debug=False):
    my_username = "vfilardi"
    current_directory = os.getcwd()
    i = 0
    start_time = time.time()

    tracking_file = f"{model_type}_generated_sh_files.txt"

    if not os.path.exists(tracking_file):
        print(f"Tracking file '{tracking_file}' not found. Please run the generation script first.")
        return

    with open(tracking_file, 'r') as tf:
        sh_file_list = [line.strip() for line in tf if line.strip().endswith(".sh")]

    sh_file_count = len(sh_file_list)
    print(f"{sh_file_count} .sh files found in '{tracking_file}'.")

    while i < sh_file_count:
        room = has_room_for_jobs(my_username)
        if room and sh_file_count > 0:
            print(f"Submitting job {i + 1}/{sh_file_count}: {sh_file_list[i]}", flush=True)
            subprocess.run(["sbatch", sh_file_list[i]])
            if debug:
                time.sleep(5)
                user_input = input("Type 'exit' to end job; Type 'runout' to turn off debug mode and run the loop; Type anything else to continue a loop: ")
                if user_input.lower() == 'exit':
                    print("Exiting the loop.")
                    break
                elif user_input.lower() == 'runout':
                    print('Debug mode turned off. Continuing submission.')
                    debug = False
                else:
                    print("Continuing in debug mode.")
            if i<5:
                time.sleep(5)
            i += 1
        else:
            time.sleep(30)
            end_time = time.time()
            elapsed_time_seconds = end_time - start_time
            hours, remainder = divmod(elapsed_time_seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"{int(hours)}:{int(minutes)} elapsed --- Waiting to submit job {i + 1}/{sh_file_count}", flush=True)

    print("All jobs have been submitted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs to the HPC cluster based on a tracking file.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type corresponding to the tracking file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (pauses after each submission).")
    args = parser.parse_args()

    submit_jobs(model_type=args.model_type, debug=args.debug)

