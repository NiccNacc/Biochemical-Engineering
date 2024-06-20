import pandas as pd
from google.cloud import bigquery
import graphing


### this is the query shortener dictionary
### this is for... shortening queries
qs = {
    "db" : "production-data-infra",
    'ig' : 'ignition_indexed',
    "mc_f1_ds" : 'production-data-infra.ignition_indexed.mc_f1_indexed_downsampled',
    "mc_f50_ds" : 'production-data-infra.ignition_indexed.mc_f50_indexed_downsampled',
    "mc_f1_cd" : 'production-data-infra.pilot_warehouse.sub_f1_celldensity_w_analytical'
}






def query_formatter(database, proj_id, query_parameters):

    ## 0 pos is what to select
    ## 1 pos is additional args to make
    ## 2 pos is WHERE
    ## 3 pos is additional args in this postion
    ## 4 pos is LIMIT

    query = "SELECT " + query_parameters[0] + " " + query_parameters[1] + " " + "FROM"+ " " + database + " " + "WHERE"+ " " + query_parameters[2] + " " + query_parameters[3]+ " " + "LIMIT" + " " + query_parameters[4]

    return query, proj_id




def query_parser(input_query, proj_id):

    df = pd.read_gbq(input_query, proj_id)
    return df





def reform_query_tp(df,column):
    reform_column = []
    for i in df[column]:
        if i==None:
            reform_column.append('')
            pass
        else:
            reform_column.append(float(i.strip("T")))

    df[column] = reform_column

    return df


#### function that takes a queried table and splits it into reactors
def query_split_rcts(rct_list,df,sort_col):
    
    df = pd.DataFrame(df)
    dfs = []
    for a in rct_list:
        index=0
        t_df = pd.DataFrame()
        for i in df[sort_col]:
            if a in i:
                temp = pd.DataFrame(df.iloc[index])
                # print(temp)
                temp = temp.transpose()
                # print(temp)
                t_df = pd.concat([t_df,temp],ignore_index=True)
            index = index+1
        # print(t_df)
        dfs.append(t_df)


    return dfs



# ####### EXAMPLE QUERY
#### example of DCW query
# ex_q_params = ["fermenter_lot_id, timepoint_h, avg_cell_density_g_per_l","","fermenter_lot_id LIKE '%DL11%'","ORDER BY timepoint_h, fermenter_lot_id","10000"]

# ex_database = "`production-data-infra.pilot_warehouse.sub_f1_celldensity_w_analytical`"

# ex_query, proj = query_formatter(ex_database,'production-data-infra',ex_q_params)
# print(ex_query)

# ex_p_query = query_parser(ex_query,proj)

#### example of ignition query
# ex_q_params = ["batch_number,t_rel_hrs, o2, co2, gas_1_sp,gas_2_sp","","batch_number LIKE '%DL11%'","ORDER BY batch_number, t_rel_hrs","10000"]

# ex_db = "`"+qs["mc_f1_ds"]+"`"

# ex_query, proj = query_formatter(ex_db,qs["db"],ex_q_params)

# ex_q_pars = query_parser(ex_query,proj)

###############



# #### EXAMPLE QUERY SPLIT BY REACTORS
# rct_list = ["F1B1","F1B2"]
# rcts = query_split_rcts(rct_list,ex_p_query,"fermenter_lot_id")
# print(rcts[0])

#####




# ####EXAMPLE QUERY REFORMED FOR NO T_timepoints

# # print(ex_p_query)

# ex_reformed_query = reform_query_tp(ex_p_query,'timepoint_h')
# # print(ex_reformed_query)
###########

##### GRAPH THE EXAMPLEQUERY
# criteria = ["F1A1","F1A2","F1B1","F1B2","F1C1"]
# example_ = graphing.sort_xy(ex_q_pars,"batch_number",criteria,"co2","t_rel_hrs")
# print(example_)
# graphing_settings = {
#     "xlimit" : [0,24],
#     "ylimit" : ['','']
# }
# graphing.graph_over(example_,criteria,graphing_settings)
########
