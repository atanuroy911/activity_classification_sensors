import py7zr

# Path to the .7z file
input_file = 'pre_processed_datasets.7z'

# Directory to extract the contents
output_dir = 'out/'

# Open the .7z file and extract its contents
with py7zr.SevenZipFile(input_file, mode='r') as archive:
    archive.extractall(output_dir)

print(f'Unzipped contents to: {output_dir}')
