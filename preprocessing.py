import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


def binary_map(feature):
    return feature.map({True: 1, False: 0})


def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """

    if (option == "Online"):
        binary_list = ['Tinted windows', 'general_inspection', 'Panorama roof', 'Fog lights', 'Sport seats', 'Roof rack', 'Electrically adjustable seats', 'Lumbar support', 'Keyless central door lock', 'Seat heating', 'Parking assist system camera',
                       'Trailer hitch', 'Adaptive Cruise Cntrl', 'On-board computer', 'Power steering', 'Electric tailgate', 'Immobilizer']
        df[binary_list] = df[binary_list].apply(binary_map)
        df.drop(columns=['make', 'model'], axis=1, inplace=True)

        for column in df.select_dtypes(exclude=[np.number]).columns:

            df[f'{column}'] = labelencoder.fit_transform(df[f'{column}'])
        
        return df.iloc[0]
