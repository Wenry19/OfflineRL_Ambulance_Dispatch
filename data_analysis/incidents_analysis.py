
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import folium
import io
from PIL import Image

if __name__ == '__main__':
    
    data_2014 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2014.csv'), dtype=str)
    data_2015 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2015.csv'), dtype=str)
    data_2016 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2016.csv'), dtype=str)
    data_2017 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2017.csv'), dtype=str)
    data_2018 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2018.csv'), dtype=str)
    data_2019 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2019.csv'), dtype=str)
    data_2020 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2020.csv'), dtype=str)
    data_2021 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2021.csv'), dtype=str)
    data_V6 = pd.read_csv(Path('CHUV/ChuvExportIncidentsV6.csv'), dtype=str)

    dataframes = [('2014', data_2014), ('2015', data_2015), ('2016', data_2016),
                  ('2017', data_2017), ('2018', data_2018), ('2019', data_2019),
                  ('2020', data_2020), ('2021', data_2021), ('V6', data_V6)]
    
    mv_evolution = []
    years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', 'V6']

    for id, df in dataframes:

        print('****Year: ' + id + '****')

        # SHAPE AND BASIC INFORMATION
        print('Columns:', list(df.columns))
        print('Number of columns:', df.shape[1])
        print('Number of rows:', df.shape[0])
        print()

        # MISSING VALUES
        print('Missing values:\n' + str(pd.isnull(df).sum()))
        print('Missing values %:\n' + str(100 * pd.isnull(df).sum()/df.shape[0]))
        mv_evolution.append(100*pd.isnull(df).any(axis=1).sum()/df.shape[0])
        print()

        # STATISTICS
        print('Priority %:')
        print('P0:', 100 * df[df['Priority'] == '0'].shape[0]/df.shape[0])
        print('P1:', 100 * df[df['Priority'] == '1'].shape[0]/df.shape[0])
        print('Pother:', 100 * df[(df['Priority'] != '0') & (df['Priority'] != '1')].shape[0]/df.shape[0])
        print()

        print('#########################################################')


    # EVOLUTION OF THE MISSING VALUES OVER THE YEARS
    plt.bar(years, mv_evolution)
    plt.xlabel('Dataset')
    plt.ylabel('% of rows with missing values')
    plt.savefig(Path('data_analysis/figures/incidents_missing_values.png'), dpi=1000)
    plt.close()

    # PLOT MAPS WITH INCIDENT POINTS
    # https://python-visualization.github.io/folium/

    # with the preprocessed incidents data!

    prep_2016 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2016.pkl'), 'rb'))
    prep_2017 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2017.pkl'), 'rb'))
    prep_2018 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2018.pkl'), 'rb'))
    prep_2019 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2019.pkl'), 'rb'))
    prep_2020 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2020.pkl'), 'rb'))
    prep_2021 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2021.pkl'), 'rb'))
    prep_V6 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_V6.pkl'), 'rb'))

    prep_data = [('2016', prep_2016), ('2017', prep_2017), ('2018', prep_2018), ('2019', prep_2019),
            ('2020', prep_2020), ('2021', prep_2021), ('V6', prep_V6)]
    
    for id, d in prep_data:

        m = folium.Map(location=[46.6739, 6.6830], zoom_start=9, height=680, width=680, zoom_control=False)

        for incident in d:
            if incident[2] == '0':
                color = 'red'
            elif incident[2] == '1':
                color = 'orange'
            else:
                color = 'green'
            folium.CircleMarker(location=incident[3], radius=1, color=color, fill=True).add_to(m)

        #m.save(Path('data_analysis/incident_maps/index.html'))
        img = m._to_png(20)
        img = Image.open(io.BytesIO(img))
        img.save(Path('data_analysis/incident_maps/' + id + '.png'))


    # Percentage of priorities in preprocessed incidents
    all_prep_inci = pickle.load(open(Path('generated_data/preprocessed_incidents/all_prep_inci.pkl'), 'rb'))

    print('Priority % (all preprocessed incidents):')

    print('P0:', 100*all_prep_inci[all_prep_inci[:, 2] == '0'].shape[0]/all_prep_inci.shape[0])
    print('P1:', 100*all_prep_inci[all_prep_inci[:, 2] == '1'].shape[0]/all_prep_inci.shape[0])
    print('Pother:', 100*all_prep_inci[(all_prep_inci[:, 2] != '0') & (all_prep_inci[:, 2] != '1')].shape[0]/all_prep_inci.shape[0])
