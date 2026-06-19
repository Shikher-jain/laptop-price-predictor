import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def enable_numpy_randomstate_pickle_compat():
    """Allow loading pickle payloads that call __randomstate_ctor with 2 args."""
    try:
        import numpy.random._pickle as np_pickle

        original_ctor = np_pickle.__randomstate_ctor

        def _compat_ctor(state=None, *args):
            return original_ctor(state)

        np_pickle.__randomstate_ctor = _compat_ctor
    except Exception:
        # If internals are unavailable, continue with default behavior.
        pass


def enable_xgboost_pickle_compat():
    """Backfill missing attrs on unpickled XGBoost estimators for sklearn get_params."""
    try:
        from xgboost.sklearn import XGBModel

        original_get_params = XGBModel.get_params
        if getattr(original_get_params, "__name__", "") == "_compat_get_params":
            return

        def _compat_get_params(self, deep=True):
            # Older serialized estimators can miss newer/expected init attrs.
            for param_name in self._get_param_names():
                if not hasattr(self, param_name):
                    setattr(self, param_name, None)
            return original_get_params(self, deep=deep)

        XGBModel.get_params = _compat_get_params
    except Exception:
        # If xgboost is unavailable, continue with default behavior.
        pass


def load_pickle_or_stop(file_name):
    """Load pickle artifacts and show a friendly error if dependency versions mismatch."""
    try:
        with open(Path(file_name), 'rb') as f:
            return pickle.load(f)
    except AttributeError as exc:
        message = str(exc)
        if '__pyx_unpickle_CyHalfSquaredError' in message:
            st.error(
                "Model artifact and installed scikit-learn version are incompatible. "
                "This project expects scikit-learn==1.4.2. "
                "Reinstall dependencies with: pip install -r requirements.txt"
            )
            st.stop()
        raise

# import the model
enable_numpy_randomstate_pickle_compat()
enable_xgboost_pickle_compat()
pipe = load_pickle_or_stop('pipe.pkl')
df = load_pickle_or_stop('df.pkl')

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = pd.DataFrame(
        [{
            'Company': company,
            'TypeName': type,
            'Ram': int(ram),
            'Weight': float(weight),
            'Touchscreen': int(touchscreen),
            'Ips': int(ips),
            'ppi': float(ppi),
            'Cpu brand': cpu,
            'HDD': int(hdd),
            'SSD': int(ssd),
            'Gpu brand': gpu,
            'os': os,
        }]
    )
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

