import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.pyplot import cm
import matplotlib as mpl
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.dates as mdates
import shutil
import json
import math

### Definitions:
# A rct or reactor will be an individual fermentation reactor and it's directly associated data. 
# These will have names like "F1A1" "F1C2" or "F3_50A" and will be stored as rct_dataframes.
# There will be two types of reactors (for now), just F3 and F1
#
# A project or proj will be an array of reactors or runs in the form of rct_dataframes. 
# These are (almost always) runs that were 
# performed at the same time on the same day. 
# These will have names like "DG20", "DF29" or "DG12" similar to the pershing Yearnum:monthnum:day
# format for naming.


### The following class will contain information for individual reactors as a part of wider projects. This
# allows pulling from individual reactors and from individual projects, or from a wider range of reactors
# across multiple projects.

class rct_dataframe:
    def __init__(self, dataframe=pd.DataFrame(), proj_id="NONE", rct_ID='NONE', rctr_type='NONE', ads=[]):
        self.dataframe = dataframe
        self.proj_id = proj_id
        self.rct_ID = rct_ID
        self.rctr_type = rctr_type
        self.ads = ads
        return super().__init__()
    ### later this class should probably incorporate functions within itself to do things like OUR or other
    # internal-dataframe calculations.


class work_space:
    def __init__(self, projects: list, styles: list):
        self.projects = [projects]
        self.styles = styles




def r_rct_file(file_path, rtcr_type, ads=[]):
    # needs to take in any bioreactor data file. This can be either a saved version from this program, or a raw euro-csv.
    # This function simply needs to read the path, be told which file type it is, and turn it into a dataframe.

    # ads are just a descriptor to inform what information has been merged with the base data set.
    # _b is the base file, saying basic reactor data has been populated. All data will be merged according to this.
    # _og is off gas data, saying off gas data has been merged
    # _r is a deprecated addition. 
    
    rctr_types = ['F1', 'F3']
    adds_options = ['_b','_og', '_r']
    rct_idents = ['F1A1','F1A2','F1B1','F1B2','F1C1','F1C2','F1D1','F1D2','50A','50B', "DO"]
    if rtcr_type not in rctr_types:
        raise ValueError("Not a valid reactor type. Valid reactor types are F1 and F3")
    
    checkval=0
    for i in rct_idents:
        if i in str(file_path):
            checkval = 1
            pass

    if checkval==0:
        raise ValueError("No reactor ID detected in file_path")

    for i in ads:
        if i not in adds_options:
            raise ValueError("Not a valid file-type modifier. Valid inputs are string values _b, _og, _dw, _lc, and _r")


    # for fp in file_path:
    # # Split the extension from the path and normalise it to lowercase.
    #     ext = os.path.splitext(fp)[-1].lower()


    if rtcr_type=='F1':
        df_type = []
        if '_b' in ads:
            df_type.append('_b')
            if '_og' in ads:
                df_type.append('_og')
            if '_r' in ads:
                df_type.append('_r')
            dataframe = pd.read_csv(file_path)
        else:
            dataframe = []
            for file in [file_path]:
                # Read each CSV file using custom separator and decimal
                df = pd.read_csv(file, thousands='.', decimal=',', sep=';' , encoding='ansi', header=0, on_bad_lines='skip')

                # Replace commas with periods in numeric columns
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns
                df[numeric_cols] = df[numeric_cols].replace({',': '.'}, regex=True)

                dataframe.append(df)
            combined_df = pd.concat(dataframe, ignore_index=True)

            #fix all the damn temperature titles
            combined_df['Temperature(C)'] = combined_df['Temperature(ºC)']

            # Lastly, just fix the DATETIME to not have the UTC modifier
            combined_df['DateTime(UTC)'] = combined_df['DateTime(UTC)'].map(lambda x: x.lstrip('+-').rstrip(' -0500'))
            combined_df['DateTime(UTC)'] = combined_df['DateTime(UTC)'].map(lambda x: x.lstrip('+-').rstrip(' +0100'))

            # datetimeformat = '%Y/%m/%d %H:%M:%S %p'
            try:
                combined_df['DateTime(UTC)'] = pd.to_datetime(combined_df['DateTime(UTC)'], format='%Y-%m-%d %H:%M:%S %p')
            except:
                combined_df['DateTime(UTC)'] = pd.to_datetime(combined_df['DateTime(UTC)'], format='%Y-%m-%d %H:%M:%S %p')
            
            dataframe = combined_df
            df_type.append('_b')

    
    if rtcr_type=="F3":
        dataframe = []
        df_type = []
        for file in [file_path]:
            combined_df = pd.read_excel(file)
        


        combined_df = combined_df.rename(columns={'gas1_flow':'Air_Flow(slpm)','tt_38':'Temperature(ºC)', 't_stamp':'DateTime(UTC)', 'at_41_do':'DO(%)'})

        combined_df['DateTime(UTC)'] = pd.to_datetime(combined_df['DateTime(UTC)'], format='%Y-%m-%d %H:%M:%S %p')
        # print(combined_df['DateTime(UTC)'][1:100])

        dataframe = combined_df
        df_type.append('_b')

        #### NEED TO DO THIS LATER!!!!@
    
    # Check which rctr ID this one is.
    for i in rct_idents:
        if i in str(file_path):
            id=i
    
    if id == None:
        raise ValueError("Could not find reactor ID")


    new_rct_dataframe = rct_dataframe(dataframe,rct_ID=id,rctr_type=rtcr_type,ads=df_type)
    return new_rct_dataframe
### returns a single rct_dataframe object with some blank attributes




### Input (should be a saved file that contains a stored version of the tables contained within a
# batch project file. Should require both the path and an array of project name(s), but as separate entities to 
# make the function iterable as a batch process if needed.)
# def open_project():
### returns a series of project variables according to the number of input projects
# actually implementing this will be tricky... Either I should reconsider the approach
# or try to create iterable variables. Or the answer is probably somethingsomething dictionaries.

### Input requires the path of the batch files, the type of reactors it's pulling, and the project id.
def batch_open_proj(folder_path,rct_type,project_id) -> list:
    dir = os.listdir(folder_path)

    batch_List = []
    for i in dir:
        if project_id in i:
            batch_List.append(i)
    # print(batch_List)
    Proj_rct_dfs = []
    for i in batch_List: 
        print(Path(folder_path +'//' + i))
        k = r_rct_file(Path(folder_path +'//' + i),rct_type)
        Proj_rct_dfs.append(k)
    
    for i in Proj_rct_dfs:
        i.proj_id = project_id

    Proj_rct_dfs.sort(key=lambda x: x.rct_ID, reverse=False)
    sorted_rct_dfs = sorted(Proj_rct_dfs, key=lambda x: x.rct_ID, reverse=False)


    return sorted_rct_dfs
### returns a table containing multiple rct_dataframe objects for that project.

### Input requires a project file array containing the individual dataframes for a project.
def batch_open_merg_og(batch_proj_list, og_path) -> list:
    ### takes the input of a project list from batch open proj and accesses each rct_dataframe objct
    ### finds each off gas according to the specified structure
    ### merges them all and re-stores the dataframe into the rct_dataframe object
    # print("the length of batch_projclist is " + str(len(batch_proj_list)))
    for i in batch_proj_list:
        df = i.dataframe
        id = i.rct_ID
        proj = i.proj_id
        og = find_read_og(og_path,proj_id=proj,rct_id=id)
        # print("length of the read find-read og array is " + str(len(og)))
        dfs = [df]
        for j in og:
            dfs.append(j)
        # print("THE LENGTH OF THE DFS IS " + str(len(dfs)))
        output=merge_dfs(dfs,merge_column='DateTime(UTC)')
        i.dataframe = output
        i.ads.append('_og')

    return batch_proj_list
### returns an array of rct_dataframes with offgas merged and the _og atribute added.
        
    

### Input requires the file_path of the off gas data. The exact pathing needs to be iterated in a 
# better way. (This is used heavily in the batch_og function).
def read_OGdat(og_data_file) -> pd.DataFrame:
    # We need to read and reformat the OG data so that it's datetime matches the F1 data
    try:
        og_data = pd.read_csv(og_data_file)
    except:
        og_data = pd.read_excel(og_data_file)
    og_data.rename(columns={og_data.columns[0]: "DateTime(UTC)"}, inplace = True)
    # print(og_dat['DateTime(UTC)'])
    og_data['DateTime(UTC)'] = pd.to_datetime(og_data['DateTime(UTC)'], format = '%Y/%m/%d %H:%M:%S.%f')

    return og_data
### returns a pd.dataframe containing the offgas data with reformatted DateTime column



### Input requires dataframes of datetime format with a name according to the merge_column input
# perhaps it's not necessary to specify this merge_column if we always will use datetime?...
def merge_dfs(dataframes, merge_column) -> pd.DataFrame:
    ### reads dataframes that are properly formatted with Datetime[0] columns
    ### 'dataframes' input should be a tuple of dataframes of the format [reactor data, og_dat1, og_dat2, ...og_dat#]
    merged_df = dataframes[0]

    by_list = ['N2', 'O2', 'Ar', 'CO2', 'C2H5OH', 'CH3OH', 'Ammonia','int', 'CDC', 'OXQ', 'RQ', 'RMS Flow', 'mass17', 'mass18','mass28', 'mass31', 'mass32', 'mass40', 'mass44', 'mass46']
    # print(dataframes)
    j=1
    # print("the length of givin dataframes for merging is "+str(len(dataframes)))
    merging=[]
    for i in range(1,len(dataframes),1):
        # print("i is " + str(i))
        merging.append(dataframes[i])
    merging_df = pd.concat(merging)
    # print ("here")
    merged_df = pd.merge_asof(merged_df, merging_df, on=merge_column, direction='nearest',tolerance=pd.Timedelta('0.00833333333333333 hour'), allow_exact_matches=True)

    ### This dataframe can be exported as a csv that is easily readable as well
    return merged_df
### returns a dataframe of merged bioreactor to off-gas data



### Input requires the destination of the specified offgas data, project_id and rct_id.
### TO DO: add a "folder structure" type function to change how batch_og gets its data.
# Currently, the folder structure this function seeks is simply how the offgas analyzer at MC saves things. 
# This is not ideal.
def find_read_og(folder_name,proj_id,rct_id) -> list:
    og_container = []
    month = str(ord(proj_id[1])-64)
    if len(month)<2:
        month = "0"+month
    # print("month is " + str(month))
    day = proj_id[2:4]
    # print("day is " + str(day))
    og_directory = os.listdir(Path(folder_name+"/"+str(rct_id)+"/analysis/"))
    # for b in og_directory:
        # print(b)
        # print("here")
    prefixed = [filename for filename in og_directory if filename.startswith(month+str(day)) or filename.startswith(month+str(int(day)+1))]
    # print(prefixed)
    for c in prefixed:
        # print(c)
        og_container.append(read_OGdat(Path(folder_name + "/"+str(rct_id)+"/analysis/"+str(c))))
        # print("length of the og container is " + str(len(og_container)))
    return og_container
### returns an array containing dataframes of offgas data with reformatted Datetime column



### Input *important* requires merged off-gas data to function at all. 
# (should add feature to check for off-gas data [ads="_og"] and return an error if the dataframe doesnt
# have the attribute)
def oxygen_uptake_rate(rct_dataframe: rct_dataframe,pressure, vol) -> rct_dataframe:
    df = rct_dataframe.dataframe

    r_const = 0.0820574 ### L*atm/mol/K
    r_air_const = 0.28705 ### J/kg/K
    d_air = 1.2928 #g/l at standard conditions (0C and 1tm)
    mm_air = 28.97 #g/mol
    R = r_air_const/mm_air

    d_o2 = 1.4289
    mm_o2 = 32
    o2_mol_per = 0.2095 #percent mol in air


    air_dat = df["Air_Flow(slpm)"]
    o2_dat = df["O2_Fow(slpm)"]
    o2_out = df["O2"]
    temp_dat = df['Temperature(ºC)']
    temp_k = []
    for i in temp_dat: ## corrects for C to K
        temp_k.append(i+273.15)

    gasflow = df["Air_Flow(slpm)"].add(df["O2_Fow(slpm)"])
    print(gasflow[0:100])

    ### calculate molair/min
    mol_air = []
    j = 0
    # # Calculate based on n = PV/RT
    # for i in air_dat:
    #     mol_air.append(pressure*i/r_const/temp_k[j])
    #     j=j+1
    ### Calculate based on density
    for i in air_dat:
        mol_air.append(1/mm_air*d_air*i)

    ### calculate mol/o2/min IN
    mol_o2_in = []
    ### based on density
    for i in o2_dat:
        mol_o2_in.append(1/mm_o2*d_o2*i)
    print("MOL O2 IN")
    print(mol_o2_in[0:100])

        ### TO DO ADD BASED ON n = PV/RT

    air_in_molO2 = []
    for i in mol_air:
        air_in_molO2.append(i*o2_mol_per)
    print(air_in_molO2[0:100])

    mol_o2_intotal = []
    k = 0
    for i in mol_o2_in:
        mol_o2_intotal.append(i + air_in_molO2[k])
        k = k+1
    print('IN TOTAL')
    print(mol_o2_intotal[0:100])

    mol_o2_out = []
    l = 0
    for i in o2_out:
        mol_o2_out.append((i/100)*gasflow[l])
        l = l+1
    print(mol_o2_out[0:100])

    OUR = []
    ## ASSUMING: Oxygen in - oxygen uptaken = oxygen out (ALL MOLS)
    x = 0
    # for i in mol_o2_out:
    #     if math.isnan(i)==True:
    #         OUR.append(None)
    #     else:
    #         OUR.append((i-mol_o2_intotal[x])*60/vol*1000)
    #     x = x+1

    for i in o2_out:
        if math.isnan(i)==True:
            OUR.append(None)
        else:
            OUR.append(((0.21-(i/100))*mol_air[x])*60/vol*1000)
        x = x+1


    df["OUR"] = OUR
    rct_dataframe.dataframe=df
    output = rct_dataframe
    return output
### Returns the input rct_dataframe with the OUR added
### NEEDS to add the attribute _our to the ads attribute array


def OUR2(rct_datf, pres, vol):
    df = rct_datf.dataframe

    air_dat = df["Air_Flow(slpm)"]
    o2_out = df["O2"]
    temp_dat = df['Temperature(ºC)']

    # n = PV/RT
    mol_air = []
    j = 0
    for i in air_dat:
        mol_air.append(i/22.4) #mol air/min
    
    o2_dif = []
    for i in o2_out:
        o2_dif.append(0.21-i/100)
    
    OUR=[]
    for i in o2_dif:
        OUR.append(mol_air[j]*i*60/vol*1000)
        j = j+1

    df["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]=OUR
    rct_datf.dataframe=df
    output = rct_datf
    return output

def OUR3(rct_frame,pres,vol):
    df = rct_frame.dataframe
    o2_in = .2098
    co2_in = 0.0394/100

    air_dat = df["Air_Flow(slpm)"]
    o2_out = df["O2"]
    temp_dat = df['Temperature(ºC)']
    co2_out = df["CO2"]
    amm_out = df["Ammonia"]
    R = 0.0831410

    j = 0
    first_val = []
    for i in air_dat:
        first_val.append((i*pres)/(vol*R*(273.15)))
        j = j+1

    second_val = []
    k = 0
    for i in o2_out:
        second_val.append(o2_in-(
            (1-o2_in-co2_in)/
            (1-i/100-(co2_out[k])/100)
            )*(i/100))
        k = k+1
    
    OUR = []
    l = 0
    for i in second_val:
        OUR.append(first_val[l]*i*1000*60)
        l = l+1

    df["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]=OUR
    rct_frame.dataframe = df
    return rct_frame

def CER(rct_frame,pres,vol):
    df = rct_frame.dataframe
    o2_in = .2098
    co2_in = 0.0394/100

    air_dat = df["Air_Flow(slpm)"]
    o2_out = df["O2"]
    temp_dat = df['Temperature(ºC)']
    co2_out = df["CO2"]
    amm_out = df["Ammonia"]
    R = 0.0831410

    j = 0
    first_val = []
    for i in air_dat:
        first_val.append((i*pres)/(vol*R*(273.15)))
        j = j+1

    second_val = []
    k = 0
    for i in co2_out:
        second_val.append((
            (1-o2_in-co2_in)/
            (1-i/100-(o2_out[k])/100)
            )*(i/100)-co2_in)
        k = k+1
    
    CER = []
    l = 0
    for i in second_val:
        CER.append(first_val[l]*i*1000*60)
        l = l+1

    df["CER(mmol⋅L⁻¹⋅hr⁻¹)"]=CER
    rct_frame.dataframe = df
    return rct_frame

def RQ_true(rct_df):
    df = rct_df.dataframe

    CER = df["CER(mmol⋅L⁻¹⋅hr⁻¹)"]
    OUR = df["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]
    RQ = []
    j=0
    for i in CER:
        try:
            RQ.append(OUR[j]/i)
        except:
            pass
        j = j+1
    df["RQ(recal)"] = RQ
    rct_df.dataframe = df
    return rct_df

### Input requires an array of properly formatted dataframes and other info.
## Perhaps better to force input of the custom rct_dataframe object?
## All the info required is pulled from attributes of an object like that anyway.
def graphing(df_array,Info_req,names,run_name,plot_t="line",subplots=False):
    ###GRAPHING FUNCTION TO MAKE TESTING EASIER

    ###Section 1: formatting the x and y data into an indexable array.
    xydats=[]
    num_graphs = len(df_array)
    num_rows = int(round(num_graphs/2,0))
    # print(num_graphs)
    # print(num_rows)

    if type(Info_req)==list:
        for i in df_array:
            if graphing_settings['Whole plot']['EFT or Datetime?']=="Datetime":
                x = i['DateTime(UTC)']
            else:
                x_a = mdates.date2num(i['DateTime(UTC)'])
                x = x_a - x_a[0]
            y1 = i[Info_req[0]]
            y2 = i[Info_req[1]]
            xydats.append([x,y1,y2])

    else:
        for i in df_array:
            x_a = mdates.date2num(i['DateTime(UTC)'])
            x = x_a - x_a[0]
            y = i[Info_req]
            xydats.append([x,y])

    
    ### Section 1.1: setting up for actually graphing
    n = len(xydats)
    color = iter(cm.tab20c(np.linspace(0, 1,10)))
    color2 = iter(cm.tab20(np.linspace(0,1,10)))


    ii=0
    # plots = []
    if graphing_settings['Whole plot']['Reformat X']=='True':
        xformatter = mdates.DateFormatter('%H')


    ### Section 2: Graphing based FIRSTLY on number of datatypes, SECONDLY on subplotting or overlay

    # If accessing two datatypes per graph...
    if type(Info_req)==list:
        if subplots==True:
            fig, axs = plt.subplots(num_rows,2)
            print(num_rows)
            row_num = 0
            for row in axs:
                col_num = 0
                for col in row:
                    numm = ii+col_num+row_num
                    try:
                        id = names[numm]
                    except:
                        break
                    c=next(color)
                    label = str(" ") + str(id)

                    ax1=col
                    ax2 = ax1.twinx()
                    ax1.set_prop_cycle(c=[(0.0, 0.119, 0.255), (0.0, 0.199, 0.255), (0.0, 0.199, 0.185),(0.0, 0.199, 0.133)])
                    ax2.set_prop_cycle(c=[(0.0, 0.119, 0.255), (0.0, 0.199, 0.255), (0.0, 0.199, 0.185),(0.0, 0.199, 0.133)])

                    ax1mask = np.isfinite(xydats[numm][1])
                    ax2mask = np.isfinite(xydats[numm][2])

                    # c=next(color)
                    label1 = str(Info_req[0]) + str(" ") + str(id)
                    label2 = str(Info_req[1]) + str(" ") + str(id)
                    ax1.plot(xydats[numm][0][ax1mask],xydats[numm][1][ax1mask],ls="-",c=c,linewidth=1, label=label1)
                    ax1.set_ylabel(Info_req[0],fontdict=font_dicc)
                    c=next(color)
                    ax2.plot(xydats[numm][0][ax2mask],xydats[numm][2][ax2mask],ls='--',c=c,linewidth=1.5, label=label2)
                    ax2.set_ylabel(Info_req[1],fontdict=font_dicc)
                    
                    # col.plot(xydats[numm][0],xydats[numm][1], label=label)
                    col.set_title(label)
                    col.xaxis.set_major_formatter(xformatter)
                    col.xaxis.set_ticks(np.arange(min(xydats[numm][0]),max(xydats[numm][0]),4/24))
                    col.set(xlabel="EFT (hr)")
                    ticklabels = [item.get_text() for item in col.get_xticklabels()]
                    try:
                        ticklabels[6]=24
                        col.set_xticklabels(ticklabels)
                    except:
                        pass
                    # col_num = col_num+1
                    col_num=col_num+1
                    xlim = graphing_settings[Info_req[0]]['xlim']
                    if not xlim == []:
                        ax1.set_xlim(xlim)
                    y1lim = graphing_settings[Info_req[0]]['ylim']
                    if not y1lim == []:
                        ax1.set_ylim(y1lim)
                    y2lim = graphing_settings[Info_req[1]]['ylim']
                    if not y2lim == []:
                        ax2.set_ylim(y2lim)
                ii=ii+1
                row_num = row_num+1

        legend_elements = [Line2D([0], [0], color='black', lw=4,ls='-', label=Info_req[0]),Line2D([0], [0], ls='--', color='black', label=Info_req[1])]
        # plt.legend(handles=legend_elements, labels=[Info_req[0], Info_req[1]],loc='best',bbox_to_anchor=(1.5, 0.5))
                    
        if subplots==False:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.set_prop_cycle(c=[(0.0, 0.119, 0.255), (0.0, 0.199, 0.255), (0.0, 0.199, 0.185),(0.0, 0.199, 0.133)])
            ax2.set_prop_cycle(c=[(0.0, 0.119, 0.255), (0.0, 0.199, 0.255), (0.0, 0.199, 0.185),(0.0, 0.199, 0.133)])
            for i in xydats:
                ax1mask = np.isfinite(xydats[ii][1])
                ax2mask = np.isfinite(xydats[ii][2])
                id = names[ii]
                try:
                    c=next(color)
                except:
                    c=next(color2)
                label1 = str(Info_req[0]) + str(" ") + str(id)
                label2 = str(Info_req[1]) + str(" ") + str(id)
                ax1.plot(xydats[ii][0][ax1mask],xydats[ii][1][ax1mask],c=c,ls="-",linewidth=1, label=label1)
                ax1.set_ylabel(Info_req[0],fontdict=font_dicc)
                try:
                    c=next(color)
                except:
                    c=next(color2)
                ax2.plot(xydats[ii][0][ax2mask],xydats[ii][2][ax2mask],c=c,ls='-',linewidth=1.5, label=label2)
                ax2.set_ylabel(Info_req[1],fontdict=font_dicc)
                ax1.set_xlabel("EFT (hrs)")
                ii=ii+1
            ax2.legend(loc="best",prop=font)
            ax1.legend(loc="center left",prop=font)
            all_runs = ""
            k = 1
            print(run_name)
            for i in run_name:
                all_runs.join(i)
                if k > len(run_name):
                    all_runs.join(', ')
                k = k+1   
            if graphing_settings['Whole plot']['Reformat X']=='True':
                plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
                plt.xticks(np.arange(min(xydats[0][0]),max(xydats[0][0]),2/24))#,labels=ticklabels)
                try:
                    ticklabels = [item.get_text() for item in plt.gcf().axes[0].get_xticklabels()]
                    ticklabels[12]=24
                except:
                    pass
                plt.xticks(np.arange(min(xydats[0][0]),max(xydats[0][0]),2/24),labels=ticklabels)
                plt.xlabel("EFT (hr)")
            plt.title(all_runs,fontdict=font_dicc)

            
            xlim = graphing_settings[Info_req[0]]['xlim']
            if not xlim == []:
                ax1.set_xlim(xlim)
            y1lim = graphing_settings[Info_req[0]]['ylim']
            if not y1lim == []:
                ax1.set_ylim(y1lim)
            y2lim = graphing_settings[Info_req[1]]['ylim']
            if not y2lim == []:
                ax2.set_ylim(y2lim)
    # If only accessing one datatype
    else:
        if subplots==True:
            fig, axs = plt.subplots(num_rows,2)
            row_num = 0
            for row in axs:
                col_num = 0
                for col in row:
                    numm = ii+col_num+row_num
                    try:
                        id = names[numm]
                    except:
                        break
                    c=next(color)
                    label = str(" ") + str(id)
                    col.plot(xydats[numm][0],xydats[numm][1],c=c, label=label)
                    col.set_title(label)
                    if graphing_settings['Whole plot']['Reformat X']=='True':
                        col.xaxis.set_major_formatter(xformatter)
                        col.xaxis.set_ticks(np.arange(min(xydats[numm][0]),max(xydats[numm][0]),4/24))
                        col.set(xlabel="EFT (hr)",ylabel=Info_req)
                    if graphing_settings['Whole plot']['Change ticks'] == "True":
                        ticklabels = [item.get_text() for item in col.get_xticklabels()]
                        try:
                            ticklabels[6]=24
                            col.set_xticklabels(ticklabels)
                        except:
                            print("could not find 24hr timepoint")
                    col_num = col_num+1
                    xlim = graphing_settings[Info_req]['xlim']
                    ylim = graphing_settings[Info_req]['ylim']
                    if not xlim == []:
                        col.set_xlim(xlim)
                    if not ylim == []:
                        col.set_ylim(ylim)

                ii=ii+1
                row_num = row_num+1
            
            
        else:            
            for i in xydats:
                
                id = names[ii]
                print(id)
                try:
                    c=next(color)
                except:
                    c=next(color2)
                label = str(" ") + str(id)
                x_mask = np.isfinite(xydats[ii][1])
                if plot_t=="line":
                    plt.plot(xydats[ii][0][x_mask],xydats[ii][1][x_mask],ls="-",c=c, linewidth="1", label=label)
                if plot_t=="dot":
                    plt.scatter(xydats[ii][0],xydats[ii][1],c=c, label=label, s=2)
                plt.ylabel(Info_req,fontdict=font_dicc)
                ii=ii+1
            plt.legend(loc="best",prop=font)
            if graphing_settings['Whole plot']['Reformat X']=='True':
                plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
                plt.xlabel("EFT (hr)",fontdict=font_dicc)
                plt.xticks(np.arange(min(xydats[0][0]),max(xydats[0][0]),2/24))#,labels=ticklabels)
                try:
                    ticklabels = [item.get_text() for item in plt.gcf().axes[0].get_xticklabels()]
                    ticklabels[6]=24
                except:
                    pass
            if len(run_name) <= 1:
                plt.title(Info_req+' ('+run_name[0]+')',fontdict=font_dicc)
            else:
                all_runs = ""
                k = 1
                for i in run_name:
                    all_runs.join(i)
                    if k > len(run_name):
                        all_runs.join(', ')
                    k = k+1   
                plt.title(all_runs,fontdict=font_dicc)
            
            xlim = graphing_settings[Info_req]['xlim']
            ylim = graphing_settings[Info_req]['ylim']
            if not xlim == []:
                plt.xlim(xlim)
            if not ylim == []:
                plt.ylim(ylim)
    ### section 3: Final formatting and such.


    
    sub_text=""
    if subplots==True:
        sub_text = "split"
    plt.tight_layout(pad=graphing_settings['Whole plot']['padding'])
    plt.savefig('PNGs/'+str(run_name)+str(Info_req)+str(sub_text))
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
### opens a matplotlib instance/figure with specified arguments as an OVERLAYED graph.



### Input requires an array of rct_dataframes for a specified project
# I believe the forced class here doesnt actually work, since the batch_proj_list it's asking 
# for is actually an array *CONTAINING* the rct_dataframes.
def proj_graphing(batch_proj_list:rct_dataframe,Info_req,plot_t="line",subplots=False):
    dfs = []
    ids = []
    run_name=[]
    for i in range(len(batch_proj_list)):
        if batch_proj_list[i].proj_id not in run_name:
            run_name.append(batch_proj_list[i].proj_id)
    for i in batch_proj_list:
        dfs.append(i.dataframe)
        if len(run_name) <= 1:
            ids.append(i.rct_ID)
        else:
            ids.append((i.proj_id+" "+i.rct_ID))
    print(ids)
    
    graphing(dfs,Info_req,ids,run_name=run_name,plot_t=plot_t,subplots=subplots)
### opens sequential reactor data into graphs of requested info



def biomass_import(Biomass_path):
    df = pd.read_csv(Biomass_path)
    
    df['DateTime(UTC)'] = pd.to_datetime(df['DateTime(UTC)'], format = 'mixed')

    return df
### imports biomass data for a whole project as a formatted dataframe



def add_biomass(proj_list,biomass_df):
    fixed_df = pd.DataFrame()    
    fixed_df["DateTime(UTC)"] = biomass_df["DateTime(UTC)"]
    new_proj_list = []
    for i in proj_list:
        df = i.dataframe
        id = i.rct_ID
        fixed_df["Biomass(g_L)"] = biomass_df[id]
        dataframes = [df,fixed_df]
        # print(dataframes)
        added_df = merge_dfs(dataframes=dataframes,merge_column="DateTime(UTC)")
        i.dataframe = added_df

    return proj_list
### merges biomass data into rct_dataframes of a project. 


def OUR_biomass(proj_list,vol):
    dfs = []
    for i in proj_list:
        dfs.append(i.dataframe)
    
    ii = 0
    for i in dfs:
        our_df = i["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]
        biomass_df = i["Biomass(g_L)"]*vol
        i["OUR(mmol⋅L⁻¹⋅gX⁻¹⋅hr⁻¹)"] = our_df/biomass_df
        proj_list[ii].dataframe = i
        ii = 1+ ii
    
    return proj_list


def save_project(proj_list,folder_path):
    proj_name = proj_list[0].proj_id
    path = os.path.join(folder_path, proj_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    proj_dic = dict()
    for i in proj_list:
        i_name = i.rct_ID
        i_dic = dict()
        i_dic["Reactor Type"] = i.rctr_type
        i_dic["Reactor ID"] = i.rct_ID    
        i_dic["Additions"] = i.ads
        proj_dic[i_name] = i_dic
        df = i.dataframe
        df.to_csv(path + "/" + proj_name + "_" + i_name  + '.csv',index=False)
    path = Path(path)
    json_path = path / (proj_name + ".json")
    json_path.write_text(json.dumps(proj_dic))
### Writes the selected project list into a set of csvs and an associated json file to describe the project



def open_project(proj_name,folder_path):
    json_path = Path(folder_path + "/" + proj_name + "/" + proj_name + ".json")
    directory = str(json_path)
    with open(directory) as json_file:
        proj_dic = json.load(json_file)
    
    project_list =[]
    for i in proj_dic:
        df_path = Path(folder_path + "/" + proj_name + "/" + proj_name + "_" + str(proj_dic[i]["Reactor ID"] + ".csv"))
        df = pd.read_csv(df_path)
        a = rct_dataframe(df,proj_name,proj_dic[i]["Reactor ID"],proj_dic[i]["Reactor Type"],proj_dic[i]["Additions"])
        project_list.append(a)
    return project_list
### Returns the project list associated with the requested project




def multi_proj_graphing(work_set: work_space, Info_req, plot_t = "line", subplots=False):

    dfs = []
    ids = []
    run_name = []

    
    # since a work_set will contain a projects list, and each
    # project will itself be a list, we have to double index
    projs = work_set.projects

    projs.sort(key=lambda x : x[0].rct_ID, reverse=False)

    projs = sorted(projs, key=lambda x : x[0].rct_ID, reverse=False)

    
    b = 0
    for i in projs:
        for j in range(len(projs)):
            print("j is " + str(j))
            print("checking for " + str(projs[b][0].proj_id))
            if projs[b][0].proj_id not in run_name:
                print("run name list is " + str(run_name))
                run_name.append(projs[b][0].proj_id)
        for j in i:
            dfs.append(j.dataframe)
            ids.append((j.proj_id+" "+j.rct_ID))
        b = b+1
    graphing(dfs,Info_req,ids,run_name=run_name,plot_t=plot_t,subplots=subplots)
### Opens graphs across a whole set of projects


def multi_proj_graphing2(work_set: work_space, Info_req, exclusions, plot_t = "line", subplots=False):

    dfs = [] #dataframes for each line
    ids = [] #the names of line
    run_name = [] #project names

    
    # since a work_set will contain a projects list, and each
    # project will itself be a list, we have to double index
    projs = work_set.projects

    projs.sort(key=lambda x : x[0].rct_ID, reverse=False)

    projs = sorted(projs, key=lambda x : x[0].rct_ID, reverse=False)

    b = 0 

    for i in projs:
        #for every list of reactors within each project...
        cur_proj = i[0].proj_id
        run_name.append(cur_proj)
        try:
            ex = exclusions[cur_proj]
        except:
            ex = []
        
        for j in i:
            if j.rct_ID not in ex:
                #for every reactor within each project's list
                dfs.append(j.dataframe)
                if b == 0:
                    ids.append(j.rct_ID+ " [" + j.proj_id + "]")
                else:
                    ids.append(j.rct_ID)
                b=b+1

        b=0
    graphing(dfs,Info_req,ids,run_name=run_name,plot_t=plot_t,subplots=subplots)

    




########################
### END OF FUNCTIONS ###
########################



# testing batch_og merging


batch_file_path = r'C:\Users\nmaher\F1workspace\BATCH FILES'
og_path = r'C:\Users\nmaher\F1workspace\OGdat'

# batch_file_path = r'C:\Users\nicol\Documents\F1_dat_processing\BATCHfiles'
# og_path = r'C:\Users\nicol\Documents\F1_dat_processing\OGdat'


# DF14 = batch_open_proj(batch_file_path,'F1','DE14')

# DF29 = batch_open_proj(batch_file_path,'F1','DF29')

# # print(path_test)




# RD6 = batch_open_proj(batch_file_path,'F1','rd6')




# print(DG12_og)

##################
### test just graphing
# sample = DG12_og[0]
# graphing(sample.dataframe,Info_req="DO(%)",names=sample.rct_ID,run_name=sample.proj_id)

##################
## test batch graphing
# testing_b_grap = proj_graphing(DG12_og, "DO(%)")
##################

# testing out OUR calculation!


# dg_og_our =[]
# for i in DG20_og:
#     our_test = oxygen_uptake_rate(i, pressure=0.8, vol=2.5)
#     dg_og_our.append(our_test)




# j=0
# for i in DG12_og:
#     new_our = oxygen_uptake_rate(i,1)
#     DG12_og[j]=new_our
#     j = 1+j

# print(test_our["OUR"])

##################
## test batch graphing
# testing_b_grap = proj_graphing(DG12_og, "DO(%)")
# testing_b_grap = proj_graphing(DG20_og, ["DO(%)","Agitation(RPM)"])

# # TESTING sending whole project to csv
# for i in DG12_og:
#     i.dataframe.to_csv(str(i.rct_ID+'.csv'))

##################
### test biomass add


# bm_df = biomass_import(r"C:\Users\nmaher\F1workspace\Biomass\DG20-MCF1-E5 - Data export.csv")
# new_df = add_biomass(DG20_og,bm_df)


font_dicc = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }


font = font_manager.FontProperties(weight='normal',
                                   style='normal', size=8)


graphing_settings = { 
    "DO(%)": {
        'xlim' : [0,1],
        'ylim' : [0,110]
    },
    "Agitation(RPM)": {
        'xlim' : [0,1],
        'ylim' : [0,1500]
    },
    "pH": {
        'xlim' : [0,1],
        'ylim' : [2,8]
    },
    "O2": {
        'xlim' : [0,1],
        'ylim' : [19.25,21.1]
    },
    "CO2": {
        'xlim' : [],
        'ylim' : [0,2.4]
    },
    "Temperature(C)": {
        'xlim' : [0,1],
        'ylim' : []
    },
    "OUR(mmol⋅L⁻¹⋅hr⁻¹)": {
        'xlim': [0,1],
        'ylim' : []
    },
    "CER(mmol⋅L⁻¹⋅hr⁻¹)": {
        'xlim': [0,1],
        'ylim' : [0,70]
    },
    "RQ(recal)": {
        'xlim': [0,1],
        'ylim' : [0.7,1.3]
    },
    "Biomass(g_L)": {
        'xlim': [0,1],
        'ylim' : []    
    },
    "OUR(mmol⋅L⁻¹⋅gX⁻¹⋅hr⁻¹)": {
        'xlim': [0,1],
        'ylim' : []      
    },
    "RQ": {
        'xlim': [0,1],
        'ylim' : [0.7,1.7]      
    },
    "Whole plot": {
        'padding': 1,
        'Reformat X': 'True',
        'EFT or Datetime?': 'EFT',
        'Change ticks': 'True'
    }
}


# multi_proj_graphing2(og_projs,"OUR(mmol⋅L⁻¹⋅hr⁻¹)",exclusions={"DG12":["F1A1","F1A2","F1B1","F1B2"]})

# multi_proj_graphing2(og_projs,"DO(%)",exclusions={"DG20":["F1A1","F1A2","F1B1","F1B2"],"DG12":["F1A2","F1B1","F1B2"]})
# proj_graphing(DG12_og_OUR,["O2","OUR(mmol⋅L⁻¹⋅hr⁻¹)"])
# for i in all_projects:
# # # #     # # try:
#     proj_graphing(i,Info_req=["DO(%)","Agitation(RPM)"],subplots=False)

#     # proj_graphing(i,Info_req=["DO(%)","CO2"],subplots=True)

#     proj_graphing(i,Info_req="pH",subplots=True)
    
    # proj_graphing(i,Info_req="O2",subplots=False)
    
    # proj_graphing(i,Info_req=["DO(%)","Agitation(RPM)"],subplots=True)
    # proj_graphing(i,Info_req=["O2","CO2"],subplots=False)

    
    
    # proj_graphing(i,Info_req=["DO(%)","Agitation(RPM)"],subplots=False)
    # except:
    #     print(KeyError)
    #     pass

# testing_opened_proj_GRAPHING = proj_graphing(test_open_proj,["O2","CO2"],subplots=True)
# testing_opened_proj_GRAPHING = proj_graphing(test_open_proj,"DO(%)",subplots=False)

###############
# Workspace test

# test_workspace = work_space(DF14,1)
# test_workspace.projects.append(DF29)
# test_workspace.projects.append(DG12_og)
# # # print(test_workspace.projects[0][0].proj_id)
# test_workspace.projects.append(DG20_og)


# # test_workspace.projects.append(DF14)
# for i in test_workspace.projects:
#     print(i[0].proj_id)
# # print(test_workspace.projects[1][0].proj_id)

# exclu_ex = {
#     "DG12" : ["F1A1","F1A2"]
# }
# # exclu_ex = {}
# multi_proj_graphing2(test_workspace,"DO(%)",subplots=False,exclusions=exclu_ex)

# # multi_proj_graphing(test_workspace,"Agitation(RPM)",subplots=False)
# multi_proj_graphing(test_workspace,["DO(%)","Agitation(RPM)"],subplots=True)

# multi_proj_graphing(test_workspace,["DO(%)","Agitation(RPM)"],subplots=False)

# graphing([DG12_og[0].dataframe],["DO(%)","pH"],"F1A1","DG12")

# fp1 = r'C:\Users\nicol\Documents\F1_dat_processing\BATCHfiles\Batch_BCU-M2DE14F1A2e2_6.csv'

# fp2 = r'C:\Users\nicol\Documents\F1_dat_processing\BATCHfiles\Batch_BCU-M1_n_DE14F1B2e2_8.csv'

# A2 = r_rct_file(fp1, "F1")
# A2.proj_id = "DF14"
# A2.rct_ID = "F1A2"

# B1 = r_rct_file(file_path=fp2,rtcr_type='F1')
# B1.proj_id = "DF14"
# B1.rct_ID = "F1B1"

# print(B1.rct_ID)

# df14proj = [A2,B1]
# savepath = r'C:\Users\nicol\Documents\F1_dat_processing'

# # save_project(df14proj,savepath)
# proj_graphing(df14proj,["DO(%)",'Agitation(RPM)'],subplots=False)

# proj_graphing(df14proj,'pH',subplots=False)

# # proj_graphing(df14proj,'Temperature(C)',subplots=False)



# proj_graphing(dg_og_our,"OUR")

batch_file_path = r'E:\BATCHfiles'
og_path = r'E:\Offgasdata'

DH23 = batch_open_proj(batch_file_path,"F1","DH23")
DH23 = batch_open_merg_og(DH23,og_path)


DG20 = batch_open_proj(batch_file_path,"F1","DG20")
DG20 = batch_open_merg_og(DG20,og_path)

DH29 = batch_open_proj(batch_file_path,"F1","DH29")
DH29 = batch_open_merg_og(DH29,og_path)

DO1 = batch_open_proj(batch_file_path,"F1","DO1")
print(DO1[1].dataframe["DO(%)"])
save_project(DO1,"")


pressure = 1.0133
volume = 4.1
proj_out = []
proj = [DH23,DG20]
for i in proj:
    for j in i:
        b = OUR3(j,pressure,volume)
        b = CER(b,pressure,volume)
        proj_out.append(b)



# proj_graphing(DH23_our,["O2","CO2"])



proj_graphing(DO1,["DO(%)","Agitation(RPM)"])

# proj_graphing(DH23_our,"CER(mmol⋅L⁻¹⋅hr⁻¹)")
# proj_graphing(proj_out,"OUR(mmol⋅L⁻¹⋅hr⁻¹)")
# xydat = [(0,12,15,18,20,21,22,24),(0.75,6.13,8.2,15.66,19.00,21.8,23.2,25.2)]

# proj_graphing(DH23_our,"RQ")

# proj_graphing(DH23_our,"CO2")

# proj_graphing(DH23_our,"pH")


# proj_graphing(DH23_our,["CER(mmol⋅L⁻¹⋅hr⁻¹)","OUR(mmol⋅L⁻¹⋅hr⁻¹)"])

# proj_graphing(DH23_our,["RQ","OUR(mmol⋅L⁻¹⋅hr⁻¹)"])


# print(max(DH23_our[0].dataframe["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]))
# print(max(DH23_our[1].dataframe["OUR(mmol⋅L⁻¹⋅hr⁻¹)"]))
