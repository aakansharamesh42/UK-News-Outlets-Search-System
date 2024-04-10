import subprocess
import os, sys
import warnings


FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

sys.path.append(UTILPATH)

from common import Logger

logpath = os.path.join(UTILPATH, 'pipeline.log')
logger = Logger(logpath)

logger.log_event('info', f'{FILENAME} - Start script')

def execute_script(script_path):
    try:
        logger.log_event('info', f'{FILENAME} - Executing {script_path}')
        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        return stdout, stderr
    except Exception as e:
        message = f"Error executing {script_path}: {str(e)}"
        logger.log_event('error', f'{FILENAME} - {message}')
        warnings.warn(message)
        quit()

def execute_pipeline(script_paths):
    errors = {}
    for script_path in script_paths:
        stdout, stderr = execute_script(script_path)
        if stderr:
            errors[script_path] = stderr
    return errors


if __name__ == "__main__":
    # List of script paths to execute

    path_module_new_data = os.path.join(BASEPATH, "module_new_data.py")
    path_run_daily_index = os.path.join(BASEPATH, "run_daily_index.py")
    path_run_daily_sentiment = os.path.join(BASEPATH, "run_daily_sentiment.py")
    path_run_summarizer = os.path.join(BASEPATH, "run_summarizer.py")

    script_paths = [
        path_module_new_data,
        path_run_daily_index,
        path_run_daily_sentiment,
        path_run_summarizer
    ]  # Add more scripts as needed
    
    # Execute the pipeline
    errors = execute_pipeline(script_paths)

    if errors:
        print("Errors occurred:")
        for script_path, error in errors.items():
            print(f"{script_path}: {error}")
    else:
        print("Pipeline executed successfully")