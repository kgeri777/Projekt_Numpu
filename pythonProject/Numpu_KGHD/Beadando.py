import pandas as pd
import numpy as np
import scipy as sp
from functools import reduce
import matplotlib.pyplot as plt


df_dba_original = pd.read_csv('DBA.csv', index_col=0)
df_tlt_original = pd.read_csv('TLT.csv', index_col=0)
df_vde_original = pd.read_csv('VDE.csv', index_col=0)
df_xlv_original = pd.read_csv('XLV.csv', index_col=0)
df_xme_original = pd.read_csv('XME.csv', index_col=0)
df_risk_free_rate = pd.read_csv('DTB3.csv')
df_risk_free_rate.index = pd.to_datetime(df_risk_free_rate['DATE'])
df_risk_free_rate = df_risk_free_rate[['DTB3']]
df_risk_free_rate.columns = ['risk_free']
msk = df_risk_free_rate['risk_free'] != '.'
df_risk_free_rate = df_risk_free_rate[msk]
df_risk_free_rate = df_risk_free_rate.astype(float)


df_dba = df_dba_original[['Adj Close']]
df_tlt = df_tlt_original[['Adj Close']]
df_vde = df_vde_original[['Adj Close']]
df_xlv = df_xlv_original[['Adj Close']]
df_xme = df_xme_original[['Adj Close']]

df_dba = df_dba.rename(columns={'Adj Close': 'Adj Close_dba'})
df_tlt = df_tlt.rename(columns={'Adj Close': 'Adj Close_tlt'})
df_vde = df_vde.rename(columns={'Adj Close': 'Adj Close_vde'})
df_xlv = df_xlv.rename(columns={'Adj Close': 'Adj Close_xlv'})
df_xme = df_xme.rename(columns={'Adj Close': 'Adj Close_xme'})
pass