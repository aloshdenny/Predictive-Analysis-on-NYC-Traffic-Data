import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx

gdf = gpd.read_file("backend/taxi_zones.zip")

def traffic_analysis_2(start_date, end_date=None, borough=None, brand=None, k=3):
    def plot(df, start, end, borough, k):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='mixed')
        if (start != end):
            if borough:
                filtered_df = df[(df['pickup_datetime'] >=start) & 
                                (df['pickup_datetime'] <=end) &
                                (df['PUBorough'] == borough)]
            else:
                filtered_df = df[(df['pickup_datetime'] >=start) & 
                                (df['pickup_datetime'] <=end)]
        elif (start == end):
            if borough:
                filtered_df = df[ (df['pickup_datetime'].dt.date == pd.to_datetime(start).date()) & (df['PUBorough'] == borough) ]
            else:
                filtered_df = df[df['pickup_datetime'].dt.date == pd.to_datetime(end).date()]
        daily_aggregate_driver_pay = filtered_df['driver_pay'].sum()
        daily_aggregate_miles = filtered_df['trip_miles'].sum()

        if borough is None:
            fig, ax = plt.subplots(figsize=(20, 20))
            gdf.plot(ax=ax, alpha=0.4, edgecolor='k')
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
            ax.axis('off')
            plt.savefig("backend/img/map.jpg")  # Save the map plot as an image
            plt.close(fig)
        else:
            borough_zones = gdf[gdf['borough'] == borough]
            fig, ax = plt.subplots(figsize=(10, 10))
            borough_zones.plot(ax=ax, alpha=0.5, edgecolor='k')
            ctx.add_basemap(ax, crs=borough_zones.crs.to_string(), source=ctx.providers.CartoDB.Positron)
            ax.set_title(f"{borough} Taxi Zones")
            ax.axis('off')
            plt.savefig("backend/img/map.jpg")  # Save the map plot as an image
            plt.close(fig)
        
        if end != start:
            daily_traffic = filtered_df.groupby(filtered_df['pickup_datetime'].dt.date).size().reset_index(name='trip_counts')
            plt.figure(figsize=(10, 6))
            sns.barplot(data=daily_traffic, x='pickup_datetime', y='trip_counts', color='skyblue')
            plt.title(f'Daily Traffic Volume from {start.date()} to {end.date()} {"in " + borough if borough else "New York State"}')
            plt.xlabel('Date')
            plt.ylabel('Number of Trips')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.savefig("backend/img/traffic.jpg")  # Save the traffic plot as an image
            plt.close()
        elif end == start:
            filtered_df['hour'] = filtered_df['pickup_datetime'].dt.hour
            hourly_traffic = filtered_df.groupby('hour').size().reset_index(name='trip_counts')
            plt.figure(figsize=(10, 6))
            sns.barplot(data=hourly_traffic, x='hour', y='trip_counts', color='skyblue')
            plt.title(f'Hourly Traffic Volume on {start.date()} {"in " + borough if borough else "New York State"}')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Trips')
            plt.xticks(range(0, 24))
            plt.grid(axis='y')
            plt.savefig("backend/img/traffic.jpg")  # Save the traffic plot as an image
            plt.close()
        
        most_common_pu_zone = filtered_df['PUZone'].value_counts().idxmax()
        most_common_do_zone = filtered_df['DOZone'].value_counts().idxmax()

        most_common_routes = df.groupby(['PUZone', 'DOZone'])['trip_miles'].size().reset_index(name='count').sort_values(by='count', ascending=False).head(k)
        least_common_routes = df.groupby(['PUZone', 'DOZone'])['trip_miles'].size().reset_index(name='count').sort_values(by='count').head(k)
        
        print(f"Most common pickup zone: {most_common_pu_zone}")
        print(f"Most common drop-off zone: {most_common_do_zone}")
        print(f"Top-{k} Most Common Routes:\n{most_common_routes}")
        print(f"Top-{k} Least Common Routes:\n{least_common_routes}")
        if end!=start:
            print(f"Aggregate Revenue on {start.date()} to {end.date()}: ${daily_aggregate_driver_pay:.2f}")
            print(f"Aggregate Miles Covered on {start.date()} to {end.date()}: {daily_aggregate_miles:.2f} miles")
        elif end==start:
            print(f"Aggregate Revenue on {start.date()}: ${daily_aggregate_driver_pay:.2f}")
        print(f"Aggregate Miles Covered on {start.date()}: {daily_aggregate_miles:.2f} miles")

    if end_date is None:
        start = pd.to_datetime(start_date)
        end=start
    else:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

    year = start.year

    if end.year != year:
        df1 = pd.read_parquet(f"./backend/Dataset/{year}.parquet", engine='pyarrow')
        df2 = pd.read_parquet(f"./backend/Dataset/{end.year}.parquet", engine='pyarrow')
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        if brand == 'Uber':
            file_path = f"./backend/Dataset/Uber_{year}.parquet"
        elif brand == 'Lyft':
            file_path = f"./backend/Dataset/Lyft_{year}.parquet"
        else:
            file_path = f"./backend/Dataset/{year}.parquet"
        
        df = pd.read_parquet(file_path, engine='pyarrow')
        
    plot(df, start, end, borough, k)

# format: traffic_analysis(start_date, end_date=None, borough=None, brand=None, k=3)
traffic_analysis_2('2022-04-06', '2022-07-08', 'Queens', 'Queens', 5)