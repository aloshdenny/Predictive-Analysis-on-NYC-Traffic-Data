from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from fastapi.staticfiles import StaticFiles
import random

# gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')
gdf = gpd.read_file("taxi_zones.zip")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class FilteredDataRow(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_miles: float
    driver_pay: float
    pickup_datetime: object
    dropoff_datetime: object
    PUZone: str
    DOZone: str
    PUBorough: str
    DOBorough: str

class PeakTrafficData(BaseModel):
    Peak_Hours: str
    Trip_Count: int

class TrafficAnalysisRequest(BaseModel):
    start_date: str
    end_date: str = None
    borough: str = None
    brand: str = None
    k: int = 3
    n: int = 3

class TrafficAnalysisResponse(BaseModel):
    most_common_pu_zone: str
    most_common_do_zone: str
    most_common_routes: List[str]
    least_common_routes: List[str]
    aggregate_driver_pay: float
    aggregate_miles: float
    title: str
    data: List[PeakTrafficData]
    filtered_data: List[FilteredDataRow]

def traffic_analysis(start_date: str, end_date: str = None, borough: str = None, brand: str = None, k: int = 3, n: int = 3) -> TrafficAnalysisResponse:

    def plot(df, start, end, borough, k, n):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='mixed')
        if (start != end):
            if borough:
                filtered_df = df[(df['pickup_datetime'] >= start) & 
                                (df['pickup_datetime'] <= end) &
                                (df['PUBorough'] == borough)]
            else:
                filtered_df = df[(df['pickup_datetime'] >= start) & 
                                (df['pickup_datetime'] <= end)]
        elif (start == end):
            if borough:
                filtered_df = df[ (df['pickup_datetime'].dt.date == pd.to_datetime(start).date()) & (df['PUBorough'] == borough) ]
            else:
                filtered_df = df[df['pickup_datetime'].dt.date == pd.to_datetime(end).date()]
        aggregate_driver_pay = filtered_df['driver_pay'].sum()
        aggregate_miles = filtered_df['trip_miles'].sum()

        if borough is None:
            fig, ax = plt.subplots(figsize=(20, 20))
            gdf.plot(ax=ax, alpha=0.4, edgecolor='k')
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
            ax.axis('off')
            plt.savefig("img/map.jpg")
            plt.close(fig)
        else:
            borough_zones = gdf[gdf['borough'] == borough]
            fig, ax = plt.subplots(figsize=(10, 10))
            borough_zones.plot(ax=ax, alpha=0.5, edgecolor='k')
            ctx.add_basemap(ax, crs=borough_zones.crs.to_string(), source=ctx.providers.CartoDB.Positron)
            ax.set_title(f"{borough} Taxi Zones")
            ax.axis('off')
            plt.savefig("img/map.jpg")
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
            plt.savefig("img/traffic.jpg")
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
            plt.savefig("img/traffic.jpg")
            plt.close()
        
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
            plt.savefig("img/peak_traffic.jpg")
            plt.close()
            
            data = []
            for index, row in peak_hours.iterrows():
                data.append(PeakTrafficData(
                    Peak_Hours=row["Peak Hours"],
                    Trip_Count=row["trip_counts"]
                ))

            title=f"Top-{n} Peak Traffic Hours from {start.date()} to {end.date()} {'in ' + borough if borough else 'New York State'}"

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
            plt.savefig("img/peak_traffic.jpg")
            plt.close()
            
            data = []
            for index, row in peak_hours.iterrows():
                data.append(PeakTrafficData(
                    Peak_Hours=row["Peak Hours"],
                    Trip_Count=row["trip_counts"]
                ))

            title=f"Top-{n} Peak Traffic Hours on {start.date()} {'in ' + borough if borough else 'New York State'}"
        
        most_common_pu_zone = filtered_df['PUZone'].value_counts().idxmax()
        most_common_do_zone = filtered_df['DOZone'].value_counts().idxmax()

        most_common_routes = df.groupby(['PUZone', 'DOZone'])['trip_miles'].size().reset_index(name='count').sort_values(by='count', ascending=False).head(k)
        least_common_routes = df.groupby(['PUZone', 'DOZone'])['trip_miles'].size().reset_index(name='count').sort_values(by='count').head(k)

        most_common_routes = [f"{pu} -> {do}" for pu, do in most_common_routes[['PUZone', 'DOZone']].itertuples(index=False, name=None)]
        least_common_routes = [f"{pu} -> {do}" for pu, do in least_common_routes[['PUZone', 'DOZone']].itertuples(index=False, name=None)]

        filtered_data = filtered_df.head(10).to_dict('records')

        return {
            "most_common_pu_zone": most_common_pu_zone,
            "most_common_do_zone": most_common_do_zone,
            "most_common_routes": most_common_routes,
            "least_common_routes": least_common_routes,
            "aggregate_driver_pay": aggregate_driver_pay,
            "aggregate_miles": aggregate_miles,
            "title": title,
            "data": data,
            "filtered_data": filtered_data
        }

    if end_date == start_date:
        start = pd.to_datetime(start_date)
        end=start
    else:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

    year = start.year

    if end.year != year:
        df1 = pd.read_parquet(f"Dataset/{year}.parquet", engine='pyarrow')
        df2 = pd.read_parquet(f"Dataset/{end.year}.parquet", engine='pyarrow')
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        if brand == 'Uber':
            file_path = f"Dataset/Uber_{year}.parquet"
        elif brand == 'Lyft':
            file_path = f"Dataset/Lyft_{year}.parquet"
        else:
            file_path = f"Dataset/{year}.parquet"
        
        df = pd.read_parquet(file_path, engine='pyarrow')
        
    return plot(df, start, end, borough, k, n)

@app.post("/traffic_analysis", response_model=TrafficAnalysisResponse)
async def analyze_traffic(request_data: TrafficAnalysisRequest) -> TrafficAnalysisResponse:

    return traffic_analysis(
        request_data.start_date,
        request_data.end_date,
        request_data.borough,
        request_data.brand,
        request_data.k,
        request_data.n
    )

app.mount("/images", StaticFiles(directory='img'), name="images")