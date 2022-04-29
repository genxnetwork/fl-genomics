import os

import pandas as pd
import geopandas as gpd

import plotly.express as px

DATA_DIR = '/home/dkolobok/Downloads/'
BIRTHPLACE_FN = os.path.join(DATA_DIR, 'birthplace.tsv')
PCS_FN = os.path.join(DATA_DIR, 'pruned_pca_white.eigenvec')
# MAP_FN = os.path.join(DATA_DIR, 'birthplace.html')
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# df = pd.read_csv('my_points.csv')
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
# result = gpd.sjoin(gdf, world, how='left')


class BirthPlaces(object):
    def __init__(self):
        self.oscoord = pd.read_csv(BIRTHPLACE_FN, sep='\t', dtype=int).query("north > -1 and east > -1")
        self.gdf = gpd.GeoDataFrame(self.oscoord, crs='EPSG:27700', geometry=gpd.points_from_xy(self.oscoord['east'],
                                                                                                self.oscoord['north']))
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        self.world.crs = "epsg:4326"
        pass

    def get_country(self, gdf):
        return gpd.sjoin(gdf, self.world, how='left')

    def add_pcs(self, gdf):
        pcs = pd.read_csv(PCS_FN, sep='\t')
        return pd.merge(gdf, pcs, how='inner', on='IID')

    def gpd_to_wgs(self,  gdf):
        gdf = gdf.to_crs("epsg:4326")
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        return gdf

    def plot(self, gdf, color_by, out_fn):
        fig = px.scatter_mapbox(data_frame=gdf,
                                lat="lat",
                                lon="lon",
                                # hover_name="StationNam",
                                # hover_data=["Altitude"],
                                color=color_by,
                                zoom=5,
                                # height=900,
                                # size=1,
                                # size_max=12,
                                opacity=0.2,
                                # width=1300
                                )
        fig.update_layout(mapbox_style='carto-positron')
        fig.update_traces(marker={'size': 5})
        # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        # fig.update_layout(title_text="Air quality level in Europe 2018")
        # fig.show()
        fig.write_html(out_fn)
        pass

    def routine(self):
        gdf = self.add_pcs(self.gdf)
        gdf = self.gpd_to_wgs(gdf)
        # gdf = self.get_country(gdf)
        for i in range(1, 6):
            self.plot(gdf=gdf, color_by=f'PC{i}', out_fn=os.path.join(DATA_DIR, f'birthplace_pruned_pc{i}.html'))
        pass


if __name__ == '__main__':
    BirthPlaces().routine()