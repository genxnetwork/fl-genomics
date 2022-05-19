import os

import pandas as pd
import geopandas as gpd

import plotly.express as px

DATA_DIR = '/home/dkolobok/Downloads/'
BIRTHPLACE_FN = os.path.join(DATA_DIR, 'birthplace.tsv')
TAG = 'uk_wb_pcs'
# TAG = 'ukb_snps_pcaprojections'
PSAM_FN = os.path.join(DATA_DIR, 'ukb_wb.psam')
# MAP_FN = os.path.join(DATA_DIR, 'birthplace.html')
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# df = pd.read_csv('my_points.csv')
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
# result = gpd.sjoin(gdf, world, how='left')


class BirthPlaces(object):
    def __init__(self):
        self.oscoord = pd.read_csv(BIRTHPLACE_FN, sep='\t', dtype=int)#.query("north > -1 and east > -1")
        self.oscoord['not_gb_born'] = (self.oscoord['north'] == -1) | (self.oscoord['east'] == -1)
        # to-do: remove filtering by north and east (mb add a separate column to show if they are -1)
        self.gdf = gpd.GeoDataFrame(self.oscoord, crs='EPSG:27700', geometry=gpd.points_from_xy(self.oscoord['east'],
                                                                                                self.oscoord['north']))
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        self.world.crs = "epsg:4326"

        self.uk = gpd.read_file('/home/dkolobok/Downloads/NUTS_Level_1_(January_2018)_Boundaries.shp')
        pass

    def get_country(self, gdf, out_fn):
        gdf = gpd.sjoin(gdf, self.uk[['nuts118cd', 'nuts118nm', 'geometry']], how='left')
        gdf[['IID', 'nuts118cd', 'nuts118nm']].to_csv(out_fn, index=False)
        return gdf

    def read_fast_pca_results(self, pcs_fn):
        pcs = pd.read_csv(pcs_fn, delim_whitespace=True, header=None)
        pcs.columns = [f'PC{i + 1}' for i in range(pcs.shape[1])]
        psam = pd.read_csv(PSAM_FN, sep='\t', usecols=['IID'])
        assert len(psam) == len(pcs)
        return pd.concat([psam, pcs], axis=1)

    def add_pcs(self, gdf):
        pcs = pd.read_csv(os.path.join(DATA_DIR, f'{TAG}.eigenvec'), sep='\t')
        # pcs = self.read_fast_pca_results(os.path.join(DATA_DIR, f'{TAG}.txt'))
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
        gdf = self.get_country(gdf, out_fn=os.path.join(DATA_DIR, 'UK_division.csv'))
        for i in range(1, 3):
            self.plot(gdf=gdf, color_by=f'PC{i}', out_fn=os.path.join(DATA_DIR, f'{TAG}_pc{i}.html'))
        pass


if __name__ == '__main__':
    BirthPlaces().routine()