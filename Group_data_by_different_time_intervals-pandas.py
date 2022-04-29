

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''
Group data by time intervals with Pandas


Refer-
https://towardsdatascience.com/how-to-group-data-by-different-time-intervals-using-python-pandas-eb7134f9b9b0
https://towardsdatascience.com/pandas-put-away-novice-data-analyst-status-part-1-7e1f0107dae0
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
'''


data = pd.read_csv("dataset_date_resample.csv", low_memory = False)

data.shape
# (35631, 41)

data.dtypes
'''
item_name                         object
item_code                          int64
bh_name                           object
bh_code                          float64
created_at                        object
brand                             object
created_fr_lat                   float64
created_fr_lon                   float64
obs_uid                           object
size                             float64
quantity                           int64
total_size                       float64
converted_total_size             float64
normalized_total_size            float64
price                            float64
normalized_price                 float64
fx_usd                           float64
fx_local                         float64
normalized_price_usd             float64
normalized_price_local           float64
currency                          object
units                             object
reference_unit_of_measurement     object
reference_quantity               float64
place_uuid                        object
place_name                        object
place_lat                        float64
place_long                       float64
store_type                        object
country                           object
l0                                object
l1                                object
l2                                object
l0_geo                            object
l1_geo                            object
l2_geo                            object
l3_geo                            object
pop_density                      float64
dist_l2                          float64
city_radius                        int64
metadata_json                     object
dtype: object
'''

# Select some features for further analysis-
reqd_cols = ['item_name', 'item_code', 'created_at', 'size', 'quantity', 'price', 'store_type']

# Reduce dataset-
data = data.loc[:, reqd_cols]

# Sanity check-
data.shape
# (35631, 7)

# Sort by 'created_at' datetime feature-
data['created_at'] = pd.to_datetime(data['created_at'])
data.sort_values(by = 'created_at', inplace = True)
data.reset_index(drop = True, inplace = True)

data.head()
'''
                                      item_name  item_code              created_at    size  quantity    price         store_type
0  Wrist-watch, men's, CITIZEN Eco-Drive BM6060  111231102 2015-12-14 18:10:03.587     1.0         1  3470.00                NaN
1                    Men's lace-up shoes, WKB-L  110321101 2015-12-14 18:12:04.662     1.0         1  1900.00                NaN
2          Dinner plate, flat, porcelain, WKB-L  110541104 2015-12-14 18:24:47.561     1.0         1    79.90                NaN
3                             Cooking salt, WKB  110119102 2015-12-14 19:59:06.881   500.0         1     5.89                NaN
4               Water, still, large bottle, WKB  110122101 2015-12-14 19:59:57.011  2000.0         1    10.09  small_medium_shop
'''

# Check for missing values-
data.isna().values.any()
# True

# data.isna().sum()

for col in data.columns.tolist():
    if data[col].isna().values.any():
        print(f"{col} has {data[col].isna().sum()} NAs")
'''
store_type has 9926 NAs
'''




'''
Combining data based on different Time Intervals-
Pandas provides the 'resample' API which can be used to resample the data
into different intervals.
'''

# Total Amount added each hour:
# To find the amount added by a contributor in an hour-
data.resample(rule = 'H', on = 'created_at').price.sum()
# Or-
# data.resample(rule = 'H', on = 'created_at')['price'].sum()
'''
created_at
2015-12-14 18:00:00     5449.90
2015-12-14 19:00:00       15.98
2015-12-14 20:00:00       66.98
2015-12-14 21:00:00        0.00
2015-12-14 22:00:00        0.00
                         ...
2016-08-16 17:00:00    15219.91
2016-08-16 18:00:00     1484.00
2016-08-16 19:00:00        0.00
2016-08-16 20:00:00    14432.00
2016-08-16 21:00:00     5039.71
Freq: H, Name: price, Length: 5908, dtype: float64
'''

# To sum two features-
# data.resample(rule = 'H', on = 'created_at')['price', 'size'].sum()

'''
Refer to Pandas Time Frequencies for a complete list of frequencies. You
can even go up to nanoseconds-
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
'''

'''
By default, the time interval starts from the starting of the hour
i.e. the 0th minute like 18:00, 19:00, and so on. We can change that
to start from different minutes of the hour using 'offset' attribute-
'''
data.resample(rule = 'H', on = 'created_at', offset = '15min10s')['price'].sum()
'''
created_at
2015-12-14 17:15:10     5370.00
2015-12-14 18:15:10       79.90
2015-12-14 19:15:10       64.56
2015-12-14 20:15:10       18.40
2015-12-14 21:15:10        0.00
                         ...
2016-08-16 17:15:10    11059.79
2016-08-16 18:15:10        0.00
2016-08-16 19:15:10        0.00
2016-08-16 20:15:10    19400.01
2016-08-16 21:15:10       71.70
Freq: H, Name: price, Length: 5909, dtype: float64
'''


# Total Amount added each week.
# To resample data based on each week and choose top 5 values-
data.resample(rule = 'W', on = 'created_at')['price'].sum()[:5]
'''
created_at
2015-12-20     43056.38
2015-12-27     67338.51
2016-01-03     44434.59
2016-01-10     18222.36
2016-01-17    190838.54
Freq: W-SUN, Name: price, dtype: float64
'''

'''
By default, the week starts from Sunday, we can change that to start
from different days. Example, to aggregate based on the week starting
on Monday-
'''
data.resample(rule = 'W-MON', on = 'created_at')['price'].sum()[:5]
'''
created_at
2015-12-14     5532.86
2015-12-21    38507.62
2015-12-28    66863.29
2016-01-04    53924.10
2016-01-11    12608.69
Freq: W-MON, Name: price, dtype: float64
'''

# Total Amount added each month-
data.resample(rule = 'M', on = 'created_at')['price'].sum()[:5]
'''
created_at
2015-12-31    1.538769e+05
2016-01-31    4.297143e+05
2016-02-29    9.352684e+05
2016-03-31    7.425185e+06
2016-04-30    1.384351e+07
Freq: M, Name: price, dtype: float64
'''

'''
Note: the output labels for each month are based on the last day of the month,
we can use the ‘MS’ frequency to start it from 1st day of the month i.e.,
instead of 2015–12–31 it would be 2015–12–01-
'''
data.resample(rule = 'MS', on = 'created_at')['price'].sum()[:5]
'''
created_at
2015-12-01    1.538769e+05
2016-01-01    4.297143e+05
2016-02-01    9.352684e+05
2016-03-01    7.425185e+06
2016-04-01    1.384351e+07
Freq: MS, Name: price, dtype: float64
'''


'''
Multiple Aggregation on sampled data:

Often, we need to apply different aggregations on different columns/
features. For example, for this dataset, we might want to compute—

1. Unique items that were added in each hour.
2. Total quantity that was added in each hour.
3. Total amount that was added in each hour.

We can do so with a one-line by using 'agg()' on the resampled data.
'''
data.resample(rule = 'H', on = 'created_at')\
.agg({'price': 'sum', 'quantity': 'sum', 'item_code': 'nunique'})[:5]
'''
                       price  quantity  item_code
created_at
2015-12-14 18:00:00  5449.90         3          3
2015-12-14 19:00:00    15.98         2          2
2015-12-14 20:00:00    66.98         7          4
2015-12-14 21:00:00     0.00         0          0
2015-12-14 22:00:00     0.00         0          0
'''
# Top 5 rows only.




'''
Grouping data based on different Time intervals:

In the examples above, we re-sampled the data and applied aggregations
on it. What if we would like to group data by other fields in addition
to time-interval? Pandas provide an API known as 'grouper()' which can
help us do that.

In this section, we will see how we can group data on different fields
and analyze them for different intervals.

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Grouper.html
'''

# Amount added for each store type in each month.
# Say, we need to analyze data based on store type for each month.
# Group data based on month and store type-
data.groupby([pd.Grouper(key = 'created_at', freq = 'M'), 'store_type'])['price'].sum().head(10)
'''
created_at  store_type
2015-12-31  other                          34300.00
            public_semi_public_service       833.90
            small_medium_shop               2484.23
            specialized_shop              107086.00
2016-01-31  market                           473.75
            other                         314741.00
            private_service_provider         325.00
            public_semi_public_service       276.79
            small_medium_shop              31042.79
            specialized_shop               29648.44
Name: price, dtype: float64
'''

'''
Explanation-

1. First, we passed the Grouper object as part of the groupby statement
which groups the data based on month i.e. ‘M’ frequency. This is similar
to resample(), so whatever we discussed above applies here as well.

2. We added 'store_type' to the groupby so that for each month we can see
different store types.

3. For each group, we selected the price, calculated the sum, and selected
the top 10 rows.
'''

# Total Amount added based on item_name in each month.
# Group data based on every month and item name-
data.groupby([pd.Grouper(key = 'created_at', freq = 'M'), 'item_name'])['price'].sum().head(10)
'''
created_at  item_name
2015-12-31  Bar soap, solid, SB                          33.17
            Beer, domestic brand, single bottle, WKB     29.79
            Black tea, BL                                12.00
            Black tea, in bags, WKB                      60.99
            Bread, white, sliced, WKB                    85.45
            Bread, whole wheat, WKB                     148.04
            Butter, unsalted, WKB                        15.90
            COCA COLA / PEPSI COLA, can                  46.20
            Cat food, tin, WKB                           28.10
            Chicken eggs, caged hen, large size          36.00
Name: price, dtype: float64
'''


'''
Multiple Aggregation for store_type in each month:

We can apply aggregation on multiple fields similarly the way we did using
'resample()'. The only thing which is different here is that the data would
be grouped by 'store_type' attribute as well and also, we can do 'NamedAggregation'
(assign a name to each aggregation) on groupby object which doesn’t work for resample.
'''
# Group data and named aggregation on 'item_code', 'quantity' & 'price'-
data.groupby([pd.Grouper(key = 'created_at', freq = 'M'), 'store_type'])\
.agg(unique_items = ('item_code', 'nunique'), total_quantity = ('quantity', 'sum'),\
total_amount = ('price', 'sum'))
'''
                                       unique_items  total_quantity  total_amount
created_at store_type
2015-12-31 other                                  3               6      34300.00
           public_semi_public_service             1               1        833.90
           small_medium_shop                     27              88       2484.23
           specialized_shop                       2              20     107086.00
2016-01-31 market                                 2              12        473.75
...                                             ...             ...           ...
2016-08-31 private_service_provider              14             220     363661.60
           public_semi_public_service             4              21       1353.23
           small_medium_shop                     86            2286     701298.32
           specialized_shop                      35             625     492062.47
           street_outlet                         29              34       1404.46

[68 rows x 3 columns]
'''

