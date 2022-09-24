
import pandas as pd

#for styling the dataframe.
style_df = lambda df:df.head(30).style.set_table_styles(
[{'selector': 'th',
  'props': [('background', '#FFFEE3'), 
            ('color', 'black'),
            ('font-family', 'verdana')]},
 
 {'selector': 'td',
  'props': [('font-family', 'verdana')]},

 {'selector': 'tr:nth-of-type(odd)',
  'props': [('background', '#ADD8E6')]}, 
 
 {'selector': 'tr:nth-of-type(even)',
  'props': [('background', 'white')]},
 
 {'selector': 'tr:hover',
  'props': [('background-color', '#FFFEE3')]}
]
)

#For dropping nan from a set.
drop_nan = lambda Set: {x for x in Set if x == x}  

#For loading the pandas dataframe stored in paths specified in path_list.
csv_to_df = lambda path_list: [pd.read_csv(path) for path in path_list]