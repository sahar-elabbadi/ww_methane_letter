#import necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
import pathlib
<<<<<<< HEAD
import geopandas as gpd
import seaborn as sns
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
from a_my_utilities import load_chp_facilities, load_ad_facilities
=======

from a_my_utilities import load_chp_facilities, load_ad_facilities, mgd_to_m3_per_day
>>>>>>> main

#load wastewater treatment plant (WWTP) data 
data_path = pathlib.Path("01_raw_data", "supplementary_database_C.xlsx")
wwtp_data = pd.read_excel(data_path)

#load combined heat and power (CHP) and anaerobic digester (AD) facilities data
chp_data = load_chp_facilities()
ad_data = load_ad_facilities()

<<<<<<< HEAD
#convert flow data from MGD to Mm3/day
chp_data['flow [Mm3/day]'] = chp_data['flow_mgd'] * 0.00378541
ad_data['flow [Mm3/day]'] = ad_data['flow_mgd'] * 0.00378541
=======
# Data concversion from MGD to m3/day
chp_data['flow_m3_per_day'] = chp_data['flow_mgd'].apply(mgd_to_m3_per_day)
ad_data['flow_m3_per_day'] = ad_data['flow_mgd'].apply(mgd_to_m3_per_day)

>>>>>>> main

#initialize figure and axis
fig, ax = plt.subplots(figsize=(30, 30))
sns.set_theme(style='white')

#read in and plot CONUS state boundaries
states = gpd.read_file(pathlib.Path("01_raw_data", "cb_2018_us_state_500k.shp"))
states[~states['STUSPS'].isin(['MP','GU','HI','VI','AS','AK','PR'])].plot(ax=ax, color='white',edgecolor='black', linewidth=2)

#set mapping parameters
combined_flow = pd.concat([chp_data['flow [Mm3/day]'], ad_data['flow [Mm3/day]']])
sizes = (combined_flow.min() * 2000, combined_flow.max() * 2000)
chp_color = sns.color_palette("tab10")[2]
ad_color = sns.color_palette("tab10")[1]

#create bubble map of CHP WWTPs
sns.scatterplot(
    data=chp_data,
    x='longitude', y='latitude',
    size='flow [Mm3/day]', sizes=sizes,
    alpha=0.5, color=chp_color,
    legend=False, edgecolor='k', linewidth=1.5
)

#create bubble map of AD WWTPs
sns.scatterplot(
    data=ad_data,
    x='longitude', y='latitude',
    size='flow [Mm3/day]', sizes=sizes,
    alpha=0.5, color=ad_color,
    legend=False, edgecolor='k', linewidth=1.5
)

#add bubble legend
ax.scatter(x=-122, y=28.4, marker='o', s=4.4*2000, color = 'white', linewidths=3, alpha=1, edgecolor='k')
ax.scatter(x=-122, y=28.4, marker='o', s=2.7*2000, color = 'white', linewidths=3, alpha=1, edgecolor='k')
ax.scatter(x=-122, y=28.4, marker='o', s=1.1*2000, color = 'white', linewidths=3, alpha=1, edgecolor='k')
ax.scatter(x=-122, y=28.4, marker='o', s=0.27*2000, color = 'white', linewidths=3, alpha=1, edgecolor='k')
plt.figtext(0.179-.0125, 0.4, r'$\mathbf{Flow\ Rate\ [Mm^3/day]}$', fontdict={'fontsize': 28,'color':'k','fontweight':'bold'})
plt.figtext(0.245-.0125, 0.4 - 0.015, '1st layer: 4.4', fontdict={'fontsize': 24,'color':'k','style':'italic'})
plt.figtext(0.245-.0125, 0.4 - 0.015*2, '2nd layer: 2.7', fontdict={'fontsize': 24,'color':'k','style':'italic'})
plt.figtext(0.245-.0125, 0.4 - 0.015*3, '3rd layer: 1.1', fontdict={'fontsize': 24,'color':'k','style':'italic'})
plt.figtext(0.245-.0125, 0.4 - 0.015*4, '4th layer: 0.27', fontdict={'fontsize': 24,'color':'k','style':'italic'})

#add color legend for CHP and AD
legend_x = -123.75
legend_y = 25.5
spacing = 1.5 
ax.scatter(legend_x, legend_y, s=500, color=chp_color, edgecolor='k', linewidth=1.5)
plt.figtext(0.18, 0.318, 'CHP Facility', fontdict={'fontsize': 26, 'color': 'k'})
ax.scatter(legend_x, legend_y - spacing, s=500, color=ad_color, edgecolor='k', linewidth=1.5)
plt.figtext(0.18, 0.318 - 0.023, 'Anaerobic Digester', fontdict={'fontsize': 26, 'color': 'k'})

#set aspect ratio and turn off axes
ax.set_aspect(1.27)
ax.set_axis_off()

#export figure as a jpeg
plt.savefig(pathlib.Path("03_figures", "figure_1.png"), dpi=300, bbox_inches='tight')

# For Brian: 
# chp_data.sort_values(by='flow_mgd', inplace=True, ascending=False)
# chp_data.to_csv(pathlib.PurePath("02_clean_data", "chp_facilities.csv"), index=False)
