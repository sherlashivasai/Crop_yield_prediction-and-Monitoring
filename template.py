import os 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    'src/__init__.py',
    '.env',
    'app1.py',
    'requirements.txt',
    'work.ipynb'

]
# Iterate over the list of files and check if they exist in the current directory

for file in list_of_files:
    file_path = Path(file)
    if file_path.is_file():
        logging.info(f'{file} exists')
    else:
        logging.info(f'{file} does not exist')
        os.system(f'touch {file}')
        logging.info(f'{file} created')

# Check if the files have been created
for file in list_of_files:
    file_path = Path(file)
    if file_path.is_file():
        logging.info(f'{file} exists')
    else:
        logging.info(f'{file} does not exist')
        logging.info(f'{file} was not created')

