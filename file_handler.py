import pandas as pd
import pathlib as pl
import os
import re
import json



main_path = pl.Path('')


jsons_list = ['rct_data_format.json','recog_F1_format.json']
rct_file_form_list = ['recog_F1_format.json']


jsons_paths = []
for i in jsons_list:
    jsons_paths.append(str(str(main_path) + '/' + i))


jsons_dic = dict()

i = 0 
for x in jsons_list:
    f = open(jsons_paths[i])
    jsons_dic[x] = json.load(f)
    i = i+1






class rct_dataframe:
    def __init__(self, dataframe=pd.DataFrame(), proj_id="NONE", rct_ID='NONE', rct_type='NONE', ads=[],formatted=False,inp_file_ty='csv'):
        self.dataframe = dataframe
        self.proj_id = proj_id
        self.rct_ID = rct_ID
        self.inp_file_ty = inp_file_ty
        self.rct_type = rct_type
        self.ads = ads
        self.formatted = formatted

        return super().__init__()
    
    
    def format(self):
        reactor_type = self.rct_type
        if reactor_type == "F1": # Benchtops    
            pass


        if reactor_type == "F3": # 50L
            pass


        # self.dataframe["Airflow(slpm)"] = 

    ### later this class should probably incorporate functions within itself to do things like OUR or other
    # internal-dataframe calculations.


#### file formats
valid_formats = re.compile('[A-F]{1}[A-L]{1}[0-3]{1}[0-9]{1}MC{1}F{1}(1|50){1}([A-D][1-2]*?){1}(\.txt)', re.IGNORECASE)
    # re.compile('')
### IMPORTANT! - CHANGE .txt to .xlxs | .csv TO ONLY ALLOW FORMATTED TABLES
## .txt was only added for bug fixing!



#### GENERAL FUNCTIONAL OUTLINE
# a file parser will take local files of any kind, however submitted, and re-configured into a proper dataframe instance. 
# Formatting can be accounted for within the function. Types of inputs include
# - F1 raw data, F3 raw data, ignition downloaded data, saved xlxs OR .csv of the previous,
# there needs to be verification of any numeric values being saved as strings (this is an issue)

# og_file parser will do the same as above, but for off-gas data.
# Query parser will reformat queries into an identical table as the rct_parser

d_csv_form = ['.',',',';','ansi',0,'skip']
d_xlsx_form = []

f1_rct_ids = ['F1A1','F1A2', 'F1B1', 'F1B2', 'F1C1', 'F1C2', 'F1D1', 'F1D2']


### TO DO:
# Need to add a REFORMATING
def rct_fp_F1raw(file_path):
    for i in f1_rct_ids:
        if i in file_path:
            id = i
        else:
            pass
    print(id)
    dataframe = []

    reformat = jsons_dic['rct_data_format.json']
    
    for file in [file_path]:
        try:
        # Read each CSV file using custom separator and decimal
            df = pd.read_csv(file, thousands=d_csv_form[0], decimal=d_csv_form[1], sep=d_csv_form[2] , encoding=d_csv_form[3], header=d_csv_form[4], on_bad_lines=d_csv_form[5])
        except:
            print("File not csv, trying xlsx")
            try:
                df = pd.read_excel(file)
            except:
                raise KeyError("File specified neither csv or xlsx")
            
        try:
            df.rename(columns={'Temperature(ºC)':"Temperature(C)"})
        except:
            print("no temp column to rename")
            
        try:
            for x in df.columns:
                # print("x is " + x)
                try:
                    z = rct_file_form_list[0]
                    # print(z)
                    # print(jsons_dic[z])
                    attempt = jsons_dic[z]
                    # print('Attempt is:')
                    # print(attempt)
                    see = attempt[str(x)]
                    # print(see)
                    # print(reformat[attempt[str(x)]])
                    df.rename(columns={str(x): reformat[attempt[str(x)]]},inplace=True)
                except:
                    df.__delitem__(x)
                    print("Error, column " + str(x) + "is an unwanted column or is incorrectly formatted") 
        except:
            raise KeyError("An error occured while trying to check \nfile dataframe's header formats")
                

        # Replace commas with periods in numeric columns
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        df[numeric_cols] = df[numeric_cols].replace({',': '.'}, regex=True)

        dataframe.append(df)
    df_extract = pd.concat(dataframe, ignore_index=True)

    #fix all the damn temperature titles
    
    # df_extract['Temperature(C)'] = df_extract['Temperature(ºC)']

    # df_extract.rename(columns={'DateTime(UTC)': reformat['date_time']},inplace=True)
    # Lastly, just fix the DATETIME to not have the UTC modifier
    df_extract[reformat['date_time']] = df_extract[reformat['date_time']].map(lambda x: x.lstrip('+-').rstrip(' -0600'))
    df_extract[reformat['date_time']] = df_extract[reformat['date_time']].map(lambda x: x.lstrip('+-').rstrip(' +0100'))

    # datetimeformat = '%Y/%m/%d %H:%M:%S %p'
    try:
        df_extract[reformat['date_time']] = pd.to_datetime(df_extract[reformat['date_time']], format='%Y-%m-%d %H:%M:%S %p')
    except:
        df_extract[reformat['date_time']] = pd.to_datetime(df_extract[reformat['date_time']], format='%Y/%m/%d %H:%M:%S %p')
    
    dataframe = df_extract
    
    new_rct_dataframe = rct_dataframe(dataframe,rct_ID=id,rct_type='F1',ads='_b')
    return new_rct_dataframe



### file parsers for other types of rct data extraction

def rct_fp_F3raw():
    pass

def rct_fp_F1ignition():
    pass

def rct_fp_F3ignition():
    pass


def rct_file_parser(input_path: pl.Path):
    ##first, check if submitted file is formatted
    if valid_formats.match(str(input_path))==None:
        return KeyError("Invalid reactor file name format submitted")
    else:
        #check type of file
        # if v
        pass

        



def og_file_parser(input_path: pl.Path):
    pass



def og_add():
    pass

def batch_file_parser(input_path: pl.Path,search_key: str):

    dir = os.listdir(input_path)
    batch_list = [x for x in dir if valid_formats.match(x)]

    for i in dir:
        print(i)
        print(valid_formats.match(i))
    return batch_list


def batch_frame_mergers():
    pass


##### TESET PROCESSING RAW FILES
# ex_path = pl.Path(r'D:/Batch_BCU-M1_submerged_DI19MCF1D1Q1012_11.csv')

# example_rct_file = rct_fp_F1raw(str(ex_path))
# print(example_rct_file.dataframe.columns)
#########


