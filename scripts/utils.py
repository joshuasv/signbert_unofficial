import os
import json
import shutil

def create_or_clean_directory(path, assume_yes=False):
    toret = False
    # Check if the directory exists
    if os.path.exists(path):
        # Assume yes if assume_yes is True, otherwise ask the user
        if assume_yes or input(f"The directory {path} already exists. Do you want to remove its contents? (y/n): ").strip().lower() == 'y':
            # Remove the contents of the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"All contents of {path} have been removed.")
            toret = True
        else:
            print("The contents of the directory will remain unchanged.")
    else:
        # Create the directory if it does not exist
        os.makedirs(path)
        print(f"The directory {path} has been created.")
        toret = True
    
    return toret

def read_json(json_fpath):

    with open(json_fpath, 'r') as fid: return json.load(fid)
