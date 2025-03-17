# Import packages

# These packages allow the opening and saving off Excel files
from openpyxl import load_workbook
import csv
from pyxlsb import open_workbook
from openpyxl import Workbook

# Pandas allows the manipulation of data in dataframes (like tables)
import pandas as pd

# OS allow the looping through each file directory path
from os import listdir
from os.path import isfile, join
import os

# Allows the manipulation of the dataframes by using their dates
from datetime import datetime
from datetime import timedelta

# This code starts with the most recent spreadsheet and imports that as a dataframe.
# It then loops through each of the spreadsheets within the ECIST data folder and appends data that is not present in that original dataframe to a list.
# It then adds that list to the original dataframe and then saves the resultant dataframe as a csv.

# The below function is to import data sets.
# As the data required is in hidden sheets which have different variations of the same 'Raw Data' naming, there are a few alternatives included.

def read_workbook(file_path):
    sheets = pd.ExcelFile(file_path, engine='pyxlsb').sheet_names
    if 'New Raw Data' in sheets:
        data = pd.read_excel(file_path, engine='pyxlsb', sheet_name='New Raw Data')
    elif 'Raw Data' in sheets:
        data = pd.read_excel(file_path, engine='pyxlsb', sheet_name='Raw Data')
    elif 'Raw data' in sheets:
        data = pd.read_excel(file_path, engine='pyxlsb', sheet_name='Raw data')
    return(data)

# The below uses this function to import the latest data set and saves it as a dataframe.
file_path_new = (r"file path")
data = read_workbook(file_path_new)
df1 = pd.DataFrame(data)
df1 = df1.dropna(subset = ['Period'])

# The below function is to convert serial date to datetime - Excel gives wierd dates when imported and we need the date as a datetime later on.
excel_origin = datetime(1900, 1, 1)
def excel_serial_to_datetime(serial):
    return excel_origin + timedelta(days=serial - 2)  # Adjust for the leap year bug

# Now we can apply that function to create a new data column with datetime in the latest dataset.
df1['date'] = df1['Period'].apply(excel_serial_to_datetime)

# Now the big bit, iterate through the file structure and take the following steps:
# - skip the file if it doesn't end in xlsb as there are other random files included in the folders
# - combine the root and file name to create a file path that we can then use to load in a dataframe
# - create a date column from the period column
# - remove the rows with dates in df2 that are present in df1 
# - add the remaining rows to a list
# - repeat the process for the following files

# Set the 'root folder' (the directory folder we want the code to loop through)
root_folder = (r'file path')

# Initialize a list as an empty DataFrame outside the loop
df2_list = []

# Loop through the file structure
for root, dirs, files in os.walk(root_folder):
    # For each file in the files within the folders
    for file in files:
        # Check if the file name ends with xlsb (check if the file is an xlsb file), if yes:
        if file.lower().endswith('.xlsb'):
            # Create a file path for that file
            file_path_old = os.path.join(root, file)
            # Import data using that file path
            data = read_workbook(file_path_old)
            # Convert that imported data into a dataframe
            df2 = pd.DataFrame(data)
            # Drop any null values where no date is recorded
            df2 = df2.dropna(subset = ['Period'])
            # Convert any unusual date naming conventions to date
            df2['Period'] = pd.to_numeric(df2['Period'], errors='coerce')
            # Drop nulls again in this data set
            df2.dropna(subset=['Period'], inplace=True)
            # Convert the excel date to a datetime object
            df2['date'] = df2['Period'].apply(excel_serial_to_datetime)
            # Create a new dataframe where any date NOT in the first dateframe we imported gets added to the list (df2_list)
            df2_filtered = df2[~df2['date'].isin(df1['date'])]
            df2_list.append(df2_filtered)
        # If the file is not an xlsb file, then skip that file and go to the next one
        else:
            continue 
        
# Stick every value in the list together with the first dataframe       
df_merged = pd.concat(df2_list, ignore_index=True)

# Drop any duplicate rows from the data
df_final = df_merged.drop_duplicates()
        
# create a file path to save down the new extended csv
csv_file_path = r"file path to save"

# Save the extended CSV
if not df_final.empty:
    df_final.to_csv(csv_file_path, index=False)
else:
    print("No data to save.")
    
