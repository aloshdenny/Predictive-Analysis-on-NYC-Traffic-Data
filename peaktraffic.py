import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import random

gdf = gpd.read_file("backend/taxi_zones.zip")

def peak_traffic(start_date, end_date=None, borough=None, brand=None, n=3):
    def plot(df, start, end, borough, n):
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

        if end != start:
            hourly_traffic = filtered_df.groupby(filtered_df['pickup_datetime'].dt.hour).size().reset_index(name='trip_counts')
            peak_hours = hourly_traffic.sort_values(by='trip_counts', ascending=False).head(n)
            peak_hours['Peak Hours'] = peak_hours['pickup_datetime'].apply(lambda x: f"{x:02d}:{00:02d}")
            
            all_hours = pd.DataFrame({'pickup_datetime': range(24), 'trip_counts': [0] * 24})
            all_hours = all_hours.set_index('pickup_datetime').reindex(range(24)).reset_index()
            all_hours.update(peak_hours.set_index('pickup_datetime')['trip_counts'])
            
            max_trips = all_hours['trip_counts'].max() + 300
            
            plt.figure(figsize=(10, 6))
            colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(24)]
            plt.scatter(all_hours['pickup_datetime'], all_hours['trip_counts'], s=all_hours['trip_counts'], c=colors)
            plt.title(f'Peak Traffic Hours from {start.date()} to {end.date()} {"in " + borough if borough else "New York State"}')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Trips')
            plt.xticks(range(24), [f"{hour:02d}:{00:02d}" for hour in range(24)], rotation=45)
            plt.ylim(0, max_trips)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig("backend/img/peak_traffic.jpg")
            plt.close()
            
            print(f"Top-{n} Peak Traffic Hours from {start.date()} to {end.date()} {'in ' + borough if borough else 'New York State'}:")
            print(peak_hours[['Peak Hours', 'trip_counts']])
        elif end == start:
            hourly_traffic = filtered_df.groupby(filtered_df['pickup_datetime'].dt.hour).size().reset_index(name='trip_counts')
            peak_hours = hourly_traffic.sort_values(by='trip_counts', ascending=False).head(n)
            peak_hours['Peak Hours'] = peak_hours['pickup_datetime'].apply(lambda x: f"{x:02d}:{00:02d}")
            
            all_hours = pd.DataFrame({'pickup_datetime': range(24), 'trip_counts': [0] * 24})
            all_hours = all_hours.set_index('pickup_datetime').reindex(range(24)).reset_index()
            all_hours.update(peak_hours.set_index('pickup_datetime')['trip_counts'])
            
            max_trips = all_hours['trip_counts'].max() + 300
            
            plt.figure(figsize=(10, 6))
            colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(24)]
            plt.scatter(all_hours['pickup_datetime'], all_hours['trip_counts'], s=all_hours['trip_counts']**1.2, c=colors)
            plt.title(f'Peak Traffic Hours on {start.date()} {"in " + borough if borough else "New York State"}')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Trips')
            plt.xticks(range(24), [f"{hour:02d}:{00:02d}" for hour in range(24)], rotation=45)
            plt.ylim(0, max_trips)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig("backend/img/peak_traffic.jpg")
            plt.close()
            
            print(f"Top-{n} Peak Traffic Hours on {start.date()} {'in ' + borough if borough else 'New York State'}:")
            print(peak_hours[['Peak Hours', 'trip_counts']])

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
        
    plot(df, start, end, borough, n)

# format: peak_traffic(start_date, end_date=None, borough=None, brand=None, n=3)
peak_traffic('2022-04-06', '2022-07-08', 'Queens', 'Uber', 5)