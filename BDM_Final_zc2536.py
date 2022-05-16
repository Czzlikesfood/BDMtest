import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import sys

from pyproj import Transformer
import math
import json
import shapely
from shapely.geometry import Point

if __name__ == '__main__':
    sc = pyspark.SparkContext.getOrCreate()
    spark = SparkSession(sc)

    nyc_centroid = './nyc_cbg_centroids.csv'
    nyc_stores = './nyc_supermarkets.csv'
    weekly_pattern = '/tmp/bdm/weekly-patterns-nyc-2019-2020/*'
    # weekly_pattern = '/tmp/bdm/weekly-patterns-nyc-2019-2020-sample.csv'

    supermarkets_rdd = spark.read.load(nyc_stores, format='csv', header=True, inferSchema=True).select('safegraph_placekey')
    weekly_rdd = spark.read.load(weekly_pattern, format='csv', header=True, inferSchema=True, escape='"').select('placekey', 'poi_cbg', 'visitor_home_cbgs', 'date_range_start', 'date_range_end')\
    .withColumn('date_range_start', F.col('date_range_start').cast('timestamp'))\
    .withColumn('date_range_end', F.col('date_range_end').cast('timestamp'))

    filtered_stores = weekly_rdd.join(supermarkets_rdd, weekly_rdd['placekey'] == supermarkets_rdd['safegraph_placekey'],how='inner' )

    geo_transformer = Transformer.from_crs(4326, 2263)
    centroids = pd.read_csv(nyc_centroid)
    nyc_cbg = centroids.loc[centroids['cbg_fips'].astype(str).str.startswith(('36061', '36005', '36047', '36081', '36085'))]
    nyc_cbg['cbg_fips'] = nyc_cbg['cbg_fips'].astype(int)
    nyc_centroid_set = set(nyc_cbg['cbg_fips'])
    centroids_dict = {}
    for k, v in zip(nyc_cbg['cbg_fips'], zip(nyc_cbg['latitude'], nyc_cbg['longitude'])):
        centroids_dict[k] = geo_transformer.transform(v[0], v[1])
    
    # Filter date
    def filter_date(start_date, end_date):
        def check_date(d):
            mar_1_2019 = pd.datetime(2019, 3, 1, 0, 0, 0)
            mar_31_2019 = pd.datetime(2019, 3, 31, 23, 59, 59)
            oct_1_2019 = pd.datetime(2019, 10, 1, 0, 0, 0)
            oct_31_2019 = pd.datetime(2019, 10, 31, 23, 59, 59)
            mar_1_2020 = pd.datetime(2020, 3, 1, 0, 0, 0)
            mar_31_2020 = pd.datetime(2020, 3, 31, 23, 59, 59)
            oct_1_2020 = pd.datetime(2020, 10, 1, 0, 0, 0)
            oct_31_2020 = pd.datetime(2020, 10, 31, 23, 59, 59)
            if mar_1_2019 <= d <= mar_31_2019:
                return 1
            if oct_1_2019 <= d <= oct_31_2019:
                return 2
            if mar_1_2020 <= d <= mar_31_2020:
                return 3
            if oct_1_2020 <= d <= oct_31_2020:
                return 4
            return 0
        start, end = check_date(start_date), check_date(end_date)
        if start or end:
            return max(start, end)
        else:
            return 0

    fun1 = F.udf(filter_date)
    filtered_date = filtered_stores.withColumn("within_date", fun1(F.col('date_range_start'), F.col('date_range_end'))).filter(F.col('within_date') > 0)

    def compute_distance(poi_cbg, home_cbgs):
        poi = centroids_dict.get(int(poi_cbg), None)
        if poi:
            vis_count, total_dist = 0, 0
            for key,value in json.loads(home_cbgs).items():
                count = int(value)
                home_cbg = int(key)
                home_pt = centroids_dict.get(home_cbg, None)
                if home_pt:
                    vis_count += count
                    tmp_dist = Point(poi[0], poi[1]).distance(Point(home_pt[0], home_pt[1]))/5280
                    total_dist += tmp_dist 
            if vis_count >0:
                return str(round(total_dist / vis_count, 2))

    funcDistance = F.udf(compute_distance)
    weeklyDistance = filtered_date.withColumn("dist", funcDistance(F.col('poi_cbg'),F.col('visitor_home_cbgs'))).select('placekey', 'poi_cbg', 'within_date', F.col('dist'))
    
    output = weeklyDistance.groupBy('poi_cbg').pivot('within_date').agg(F.first('dist')).na.fill('').sort('poi_cbg', ascending=True).select(F.col('poi_cbg').alias('cbg_fips'), F.col('1').alias('2019-03'), F.col('2').alias('2019-10'), F.col('3').alias('2020-03'), F.col('4').alias('2020-10'))
    # output = final_pivot.na.fill('').sort('poi_cbg', ascending=True).select(F.col('poi_cbg').alias('cbg_fips'), F.col('1').alias('2019-03'), F.col('2').alias('2019-10'), F.col('3').alias('2020-03'), F.col('4').alias('2020-10'))
    output.rdd.map(tuple).saveAsTextFile(sys.argv[1])