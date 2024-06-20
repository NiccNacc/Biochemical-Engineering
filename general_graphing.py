import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np



"""
The desired format for a graphing object should be a dictionary of all the required attributes.
x_dats and y_dats should be a an array of arrays.
e.g. x_dats or y_dats should look like below:

This example is of TWO sets of data showing 3 attributes in 3 different arrays.
Specifically, it is intended for overlaying things like pH, DO, O2 etc for an individual reactor itteratively. 

example set:
x dat = {
    [
     [[array1],[array2],[array3]],
     [[array1],[array2],[array3]] 
    ]
}


TO DO:
- a parsing function to verify the graphing objects are formatted correctly
- add more graph customization such as
    - line thickness
    - fonts, font sizes
    - size of markers

"""

graphing_settings = {

    "legend_location" : "best",
    "line_thickness" : 5
} 

example_graphing_obj = {
    "x_dats" : 
    [
        [[0, 1, 2, 3, 4, 5],[0, 1, 2, 3, 4, 5, 6]],
        [[1,2,3]]
    ],
    "y_dats" : 
    [
        [[0, None , 2, 4, 8, 16],[5, 8, 9, 10, 4, 3, 2]],
        [[3,2,1]]
        ],
    "title" : "example graph",
    "data_sets": ["You", "Me"],
    "x_label" : "Time [hrs]",
    "y_label" : ["Pibbles (#)",
                 "Dribbles (avg yibbles/L)"],
    "x_tick_int" : 1,
    "y_tick_int" : 2,
    "x_lim" : "",
    "y_lim" : "",
    "legend" : True,
    "c_pal" : [['r','b'],['darkred','darkblue'],['darkgreen']],
    "s_pal" : ['.','^','.'],
    "p_type" : "line",

}

def g_obj_check():
    '''
    function for verifying the formatting of a graphing_object
    '''
    pass



# def g_labels(graphing_obj,axes_table):
        

#     plt.title(graphing_obj["title"])
#     plt.xlabel(graphing_obj["x_label"])
#     plt.ylabel(graphing_obj["y_label"])

#     if graphing_obj["x_lim"]=="":
#         pass
#     else:
#         plt.xlim((graphing_obj["x_lim"][0],graphing_obj["x_lim"][1]))

#     if graphing_obj["y_lim"]=="":
#         pass
#     else:
#         plt.ylim((graphing_obj["y_lim"][0],graphing_obj["y_lim"][1]))

def g_labels(graphing_obj,axes_table):

    plt.title(graphing_obj["title"])
    axes_table[0].set_xlabel(graphing_obj["x_label"])
    if graphing_obj["x_lim"]=="":
        pass
    else:
        plt.xlim((graphing_obj["x_lim"][0],graphing_obj["x_lim"][1]))

    iter=0

    for i in axes_table:
        i.set_ylabel(graphing_obj["y_label"][iter])
        # if iter==0:
        #     axes_table[iter].legend(loc=graphing_settings["legend_location"])
        # iter=iter+1

    iter = 0

    legend_elements = []
    for i in graphing_obj["y_label"]:
        legend_elements.append(Line2D([0],[0],
                                      color= 'black', # graphing_obj["c_pal"][0][iter],
                                      marker=graphing_obj["s_pal"][iter],
                                      label=i))
        iter = iter+1

    iter = 0
    for i in graphing_obj["data_sets"]:
        legend_elements.append(Line2D([0],[0],
                               color = graphing_obj["c_pal"][0][iter],
                               lw=4,
                               label=i))
        iter = iter+1    
        
    axes_table[0].legend(handles=legend_elements,loc=graphing_obj["leg_loc"])




# def s_a_overlay(graphing_obj: dict, iter=0, iter_inner=0):
    
#     if graphing_obj["p_type"]=="scatter":
#         plt.scatter(graphing_obj["x_dats"][iter][iter_inner],graphing_obj["y_dats"][iter][iter_inner],c=graphing_obj["color_pallete"][iter][iter_inner])

#     elif graphing_obj["p_type"]=="line":
#         plt.plot(graphing_obj["x_dats"][iter][iter_inner],graphing_obj["y_dats"][iter][iter_inner],c=graphing_obj["color_pallete"][iter][iter_inner])

def a_overlay(g_obj,ax_x: plt.Axes,o_i=0,i_i=0):

    x = np.array(g_obj["x_dats"][o_i][i_i]).astype(float)
    y = np.array(g_obj["y_dats"][o_i][i_i]).astype(float)

    y_mask = np.isfinite(y)
    print("iterating inside the sub")
    print(o_i, i_i)
    
    # for i in g_obj["data_sets"]:



    if g_obj["p_type"]=="scatter":
        ax_x.scatter(x[y_mask],y[y_mask],
                     c=g_obj["c_pal"][o_i][i_i],
                     marker=g_obj["s_pal"][o_i],
                     label=g_obj["data_sets"][i_i])

    elif g_obj["p_type"]=="line":
        ax_x.plot(x[y_mask],y[y_mask],
                  c=g_obj["c_pal"][o_i][i_i],
                  marker=g_obj["s_pal"][o_i],
                  label=g_obj["data_sets"][i_i])


# def single_axis_graphing(graphing_obj,iter):

#     iter_inner = 0
#     for i in graphing_obj["x_dats"][iter]:
#         s_a_overlay(graphing_obj,iter,iter_inner)
#         iter_inner = iter_inner+1

#     g_labels(graphing_obj)
    
# def d_a_overlay(graphing_obj,ax_x: plt.Axes,o_iter, i_iter,):

#     if graphing_obj["p_type"]=="scatter":
#         ax_x.scatter(graphing_obj["x_dats"][o_iter][i_iter],graphing_obj["y_dats"][o_iter][i_iter],c=graphing_obj["color_pallete"][i_iter])

#     elif graphing_obj["p_type"]=="line":
#         ax_x.plot(graphing_obj["x_dats"][o_iter][i_iter],graphing_obj["y_dats"][o_iter][i_iter],c=graphing_obj["color_pallete"][i_iter])



def multi_axis_graphing(graphing_obj):

    fig, ax_i = plt.subplots()

    o_i=0
    
    axes_table=[]

    for i in graphing_obj["x_dats"]:

        i_i=0
        if o_i==0:
            for j in i:
                ax_x = ax_i
                a_overlay(graphing_obj,ax_x,o_i,i_i)
                i_i = i_i +1 
        
        else:
            ax_x = ax_i.twinx()
            for j in i:
                print("iterating inside the main")
                print(o_i, i_i)
                a_overlay(graphing_obj,ax_x,o_i,i_i)
                i_i = i_i +1 
        axes_table.append(ax_x)
        o_i = o_i+1
    
    
    g_labels(graphing_obj,axes_table)





# ex_path = r'C:\Users\nicol\Documents\F1_dat_processing\F1B2DG20.csv'

# ex_dat = pd.read_csv(ex_path)
# example_graphing_obj["y_dats"] = [[[ex_dat["pH"]]],[[ex_dat["DO(%)"]]],[[ex_dat["Air_Flow(slpm)"]]]]
# example_graphing_obj["x_dats"] = [[[ex_dat["Age(Hours)"]]],[[ex_dat["Age(Hours)"]]],[[ex_dat["Age(Hours)"]]]]
# example_graphing_obj["data_sets"] = ["F1B2 pH","F1B2 DO","F1B2 airflow"]
# example_graphing_obj["y_label"] = ["pH", "DO","airflow"]


def dat_org_graph(dat_array,pullx_array,pully_array,graphing_obj = {"x_dats" : [], "y_dats" : []}):

    if len(pullx_array)==1:
        for i in range(len(pully_array)-1):
            pullx_array.append(pullx_array[0])
    print("pull x array is " + str(pullx_array))

    o_i = 0
    for dats in dat_array:
        xcontainer = pd.DataFrame([])
        ycontainer = pd.DataFrame([])
        i_i = 0
        for i in pullx_array:
            print(i)
            xcontainer = [pd.concat([dats[i]])]
            ycontainer = [pd.concat([dats[pully_array[i_i]]])]
            i_i = i_i +1
        
        graphing_obj["x_dats"].append(xcontainer)
        graphing_obj["y_dats"].append(ycontainer)
        o_i = o_i +1
    print("len is")
    print(len(graphing_obj["x_dats"]))
    return graphing_obj

        
def xy_query_org(graphing_obj,query_df,x_axis,req_data,req_ids=[]):


    if req_ids == []:
        o_ter = 0

        for i in query_df:

            i_ter = 0


            for j in req_data:

                print(i_ter)
                print(o_ter)

                graphing_obj["x_dats"][i_ter][o_ter] = i[x_axis]
                graphing_obj["y_dats"][i_ter][o_ter] = i[j]
                

                i_ter = i_ter + 1

            o_ter = o_ter + 1
        
    return graphing_obj


# multi_axis_graphing(example_graphing_obj)
# example_graphing_obj["y_dats"]=[]
# example_graphing_obj["x_dats"]=[]
# new_ex = dat_org_graph([ex_dat],["Age(Hours)"],["pH","DO(%)"],graphing_obj=example_graphing_obj)

# multi_axis_graphing(new_ex)
# plt.show()
