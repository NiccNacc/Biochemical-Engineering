import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.pyplot import cm
from matplotlib import inline
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np


graphing_options = {
    "Column names" : {
        "dt" : "DateTime(UTC)",
        "pH" : "pH",
        "eft" : "EFT",
        "agi" : "Agitation(RPM)",
        "bm" : "Biomass",
        "S_util" : "S_util",
        "S_tot" : "S_tot"
    },
    "Data style settings" : {
        "dt" : { 
            "Label" : "DateTime(UTC)",
        },
        "pH" : {
            "Label" : "pH",
            "Line style" : "dashdot",
            "Marker" : "",
            "x limits" : [],
            "y limits" : []
         },
        "eft" : {
            "Label" : "EFT (hr)"
        },
        "DO" : {
            "Label" : "DO (%)",
            "Line style" : ":",
            "Marker" : "",
            "x limits" : [],
            "y limits" : []
        },
        "agi" : {
            "Label" : "Agitation (RPM)",
            "Line style" : "-",
            "Marker" : "",
            "x limits" : [],
            "y limits" : []
        },
        "bm" : {
            "Label" : "Biomass (g/L)",
            "Line style" : "-",
            "Marker" : "",
            "x limits" : [0,25],
            "y limits" : [0,20]
        },
        "S_util" : {
            "Label" : "Substrate Utilization rate (gS/dt)",
            "Line style" : "--",
            "Marker" : "",
            "x limits" : [],
            "y limits" : [0,10]
        },
        "S_tot" : {
            "Label" : "Total substrate (gS)",
            "Line style" : "--",
            "Marker" : "",
            "x limits" : [],
            "y limits" : []
        }
    },
    "Plot Setup" : {
        "Time axis" : "eft",
        "Plot adjustments" : {
            "Left" : 0.0,
            "Right" : 0.75 
        },
        "Multi-Y axis distance" : 0.15,
        "Color map" : cm.tab20(np.linspace(0,1,20)),
        "Legend Settings" : {
            "Main Legend" : {
                "Location" : "best"
            },
            "Secondary Legends" : {
                "Location" : "best"
            }
        }
    }
}






def xy_org(dfs, info_req):

    xydats = []

    time_axis = graphing_options["Plot Setup"]["Time axis"]
    time_col = graphing_options["Column_names"][time_axis]

    for i in dfs:
        
        x = i[time_col]
        xy_frame = [x]
        y_dats = []

        for j in info_req:
            info_id = graphing_options["Column_names"][j]
            y_dats.append(i[info_id])

        for k in y_dats:
            xy_frame.append(k)
        xydats.append(xy_frame)

    return xydats


### Takes xydata list of lists, overlay plots them x, y1, y2, (...) yn 
def plot_over(xydats,info_reqs,data_info):
    
    y_axis = {}
    n=0
    fig, ax = plt.subplots()

    for a in info_reqs:

        y_axis[a] = "twinx{0}".format(n)
        n = n+1



    # fig.subplots_adjust(
    #     left=graphing_options["Plot Setup"]["Plot adjustments"]["Left"],
    #     right=graphing_options["Plot Setup"]["Plot adjustments"]["Right"]
    # )
    labels = []
    line_styles = []
    for i in info_reqs:
        labels.append(graphing_options["Data style settings"][i]["Label"])
        line_styles.append(graphing_options["Data style settings"][i]["Line style"])
        if not i == info_reqs[0]:
            (y_axis[i]) = ax.twinx()

    c_map = graphing_options["Plot Setup"]["Color map"]
    c_iter = iter(c_map)

    for j in xydats:
        n=1
        c_c = next(c_iter)
        ax.plot(j[0],j[n],c=c_c,ls=line_styles[n-1],label=labels[0])
        ax.set_ylabel(labels[0])
        ax.set_xlabel(graphing_options["Data style settings"][graphing_options["Plot Setup"]["Time axis"]]["Label"])
        x_lim = graphing_options["Data style settings"][info_reqs[0]]["x limits"]
        y_lim = graphing_options["Data style settings"][info_reqs[0]]["y limits"]
        if not x_lim == []:
            ax.set_xlim(x_lim)
        if not y_lim == []:
            ax.set_ylim(y_lim)
        n=n+1
        h1, l1 = ax.get_legend_handles_labels()
        h_list = h1
        l_list = l1
        for i in info_reqs:
            x_lim = graphing_options["Data style settings"][i]["x limits"]
            y_lim = graphing_options["Data style settings"][i]["y limits"] 
            if not i==info_reqs[0]:
                # c_c = next(c_iter)
                y_axis[i].plot(j[0],j[n],c=c_c,ls=line_styles[n-1],label=labels[n-1])
                y_axis[i].spines.right.set_position(('axes',1+(n-2)*graphing_options["Plot Setup"]["Multi-Y axis distance"]))
                y_axis[i].set_ylabel(labels[n-1])
                h, l = y_axis[i].get_legend_handles_labels()
                h_list = h_list + h
                l_list = l_list + l
                n = n+1
                # y_axis[i].legend(loc=graphing_options["Plot Setup"]["Legend Settings"]["Secondary Legends"]["Location"],bbox_to_anchor=(0.5,0.,0.5,0.5))
                if not x_lim == []:
                    y_axis[i].set_xlim(x_lim)
                if not y_lim == []:
                    y_axis[i].set_ylim(y_lim)
        
        ax.legend(h_list,l_list,loc=graphing_options["Plot Setup"]["Legend Settings"]["Main Legend"]["Location"],bbox_to_anchor=(0.,0.5,0.5,0.5))


    plt.tight_layout()
    plt.show()


def plot_sub():
    
    pass

### INPUT: list of reactor dataframe objects, style of graph, and requested info
def graph_handle(rct_dfs,graph,info_reqs):
    
    dfs = []
    for i in rct_dfs:
        dfs.append(i.dataframe)

    pass

example = [[0,20,30],[1,3,4],[0,2,20],[5,15,22]],[[5,15,22],[1,7,10],[3,6,9],[1,3,4]]

example_data_info = {
    "Set1"
}

plot_over(example,["pH","DO","agi"],example_data_info)