import os
import subprocess

# Replace this with the path to your Dropbox directory
dropbox_dir = '/home/mp74207/dropboxpartition/Dropbox'

for root, dirs, files in os.walk(dropbox_dir):
    if 'venv' in dirs:
        venv_dir = os.path.join(root, 'venv')
        print(f'Ignoring {venv_dir}...')
        subprocess.run(['attr', '-s', 'com.dropbox.ignored', '-V', '1', venv_dir])
