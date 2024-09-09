# this program creates graphs from Monics csv output files


import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv
    
# Function to create graphs
def create_graphs(file_name, column_name_list=None, observed=None):

    with open(file_name, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(4096))
        csvfile.seek(0)
        if dialect.delimiter != ',':
            dialect.delimiter = ','
        
        # check if the file is a csv file
        csv_reader = csv.reader(csvfile, dialect)
        
        # skip first row
        _ = next(csv_reader) # daily
        header = next(csv_reader) # header
        _ = next(csv_reader) # units
        # check if the header is correct, and contains the column names + Date
        if header[0] != 'Date':
            print(f"File {file_name} does not contain a Date column")
            return
        if column_name_list is not None:
            for name in column_name_list:
                if name not in header:
                    print(f"Column {name} not found in file {file_name}")
                    return
                
        # construct the dataframe
        df = pd.DataFrame(columns=header)      
         # find the end of daily data, search for the first empty row  
        for row in csv_reader:
            if not any(row):
                break
            df.loc[len(df.index)] = row

        # filter columns for column_name_list
        if column_name_list is not None:
            df = df[['Date']+ column_name_list ]

        # Accumulate the data
        for column_name in column_name_list:
            df = accumulate_data(df, column_name)

        # Get the column names
        columns = df.columns

        # Get the number of columns
        num_columns = len(columns)
        # Get the number of graphs
        num_graphs = num_columns - 1
        # Get the x axis
        x = df[columns[0]] # Date
        # drop header row
        x = x.drop([0])
        # convert x to datetime
        x = pd.to_datetime(x)
        # Get the y axis
        y = df[columns[1:num_columns]]
        # drop header row
        y = y.drop([0])
        # convert y to float
        y = y.astype(float)
        # create name for the output folder
        output_folder = file_name.split('.')[0]
        # create the output folder
        os.makedirs(output_folder, exist_ok=True)
        pathToOutput = os.path.join(os.getcwd(), output_folder)
        # Create the graphs
        for i in range(num_graphs):
            currentColumn = columns[i+1]
            print("current:", currentColumn)
            plt.plot(x, y[y.columns[i]], label=y.columns[i])
            # reduce the number of ticks for dates on x axis
            plt.xticks(rotation=35)
            plt.xlabel(columns[0])
            # reduce the number of ticks on y axis to 15 ticks between min and max
            min = y[y.columns[i]].min()
            max = y[y.columns[i]].max()
            title = currentColumn
            if observed is not None and title.endswith('_Accumulated'):
                title = title[:-12] # remove _Accumulated
                if title in observed:
                    observed_df = observed[title][0]
                    units = observed[title][1]
                    # get column name that is not Date
                    len_Col = len(observed_df.columns)
                    idxUnits = 0
                    for i in range(1, len_Col):
                        if observed_df.columns[i] != 'Date': 
                            label = 'Observed'
                            if len(units) > 0:
                                label = label + ' ' + units[idxUnits]  
                                idxUnits += 1
                                                        
                            column_name = observed_df.columns[i]
                            # Get column at index i
                            obs_y = observed_df.iloc[:, i]
                            # get min and max of observed data
                            minY = obs_y.min()
                            maxY = obs_y.max()
                            if minY < min:
                                min = minY
                            if maxY > max:
                                max = maxY
                            obs_x = observed_df['Date'] # Date
                            plt.plot(obs_x, obs_y, label=label)
                    
            plt.yticks(np.linspace(min, max, 15))
            plt.ylabel(currentColumn)
            plt.title(currentColumn)
            plt.legend()
            # Save the graph
            plt.savefig(f"{pathToOutput}/{currentColumn}.png")
            print(f"Graph {pathToOutput}/{currentColumn}.png created")
            plt.close()

# Function to accumulate the data day by day and add it as a column to the dataframe
def accumulate_data(df, column_name):
    # Get the column
    column = df[column_name]
    # Get the length of the column
    length = len(column)
    # make a copy of the column
    new_column = column.copy()
    # change tpye to float
    new_column = new_column.astype(float)

    # Accumulate the data
    prev = 0

    for i in range(length):
        col = float(new_column[i])
        prev += col
        new_column[i] = prev
    # Add the new column to the dataframe
    df[column_name + '_Accumulated'] = new_column
    return df

# Function read observed data from excel file
def read_observed_data(file_name, sheet_name, colunm_name):
    # get all sheets
    sheets = pd.ExcelFile(file_name).sheet_names
    if sheet_name not in sheets:
        print(f"Sheet {sheet_name} not found in file {file_name}")
        return
    # read the sheet
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    units = []
    # check if df has column date
    if 'Date' not in df.columns:
        # get column names that don't start with Unnamed as Units
        for column in df.columns:
            if not column.startswith('Unnamed'):
                units.append(column)
        df.columns = df.iloc[0]


    #df.columns = df.iloc[0]
    # # drop rows with all NaN
    # df = df.dropna(how='all')
    # # drop columns with all NaN
    # df = df.dropna(axis=1, how='all')
    # drop all columns except the one with the date and the column_name
    df = df[['Date', colunm_name]]
    # remove header row
    df = df.drop([0])
    
    # change Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # change column_name to float
    df[colunm_name] = df[colunm_name].astype(float)
    return df,units

# Function to get the csv files
def get_csv_files():
    # Get the current working directory
    cwd = os.getcwd()
    # Get the files in the current working directory
    files = os.listdir(cwd)
    # Get the csv files
    csv_files = []
    for file in files:
        if re.search('.csv', file):
            csv_files.append(file)
    return csv_files

# Main function
def main():

    # define where the observed data is stored and to which coulmn it corresponds
    observed = {
        'Recharge': os.path.abspath('./observed_data/GMN_waterleaching.xlsx'),
        'NLeach': os.path.abspath('./observed_data/GMN_Nleaching.xlsx')
    }
    

    sim_folder = './sim_data'
    os.chdir(sim_folder)
    # Get the csv files
    csv_files = get_csv_files()
    # get keys from csv file names BR1_result.csv or BR1-out.csv -> BR1
    keys = []
    for file in csv_files:
        tokens = file.split('_')
        if len(tokens) == 1:
            tokens = file.split('-')
        keys.append( tokens[0])
    
    # create a dictionary with the keys and the csv files
    csv_files = dict(zip(keys, csv_files))

    # list of columns to be plotted
    column_name_list = ["Recharge","NLeach"]
    # Create the graphs
    for key in csv_files:
        file = csv_files[key]
        # drop the number from the key
        sheet_name = key[:-1]
        if key == 'GE1':
            sheet_name = 'GE1'
        elif key.startswith('GE'):
            sheet_name = 'GE2_7'
        observed_df = dict()
        for keyRef, obsfile in observed.items():
            # example: IT1_Cum
            column = key + '_Cum'
            observed_df[keyRef] = read_observed_data(obsfile, sheet_name, column)

        create_graphs(file, column_name_list, observed_df)

# Call the main function
if __name__ == '__main__':
    main()