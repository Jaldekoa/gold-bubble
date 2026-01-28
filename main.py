import streamlit as st
from lppls import lppls_cmaes
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go

# Configuración compacta de la página
st.set_page_config(page_title="Gold LPPLS Analyzer", initial_sidebar_state="collapsed")

# Estilo CSS para reducir espacios en blanco superiores y optimizar el "Above the Fold"
st.markdown("""
    <style>
        /* Reducir márgenes superiores */
        .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
        h1 {margin-top: -1rem; margin-bottom: 0.5rem; font-size: 3.2rem !important; text-align: center;}
        
        /* Forzar que las métricas no se corten y tengan buen tamaño */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            word-break: break-word;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
        }
        /* Ajustar el espacio entre pestañas */
        .stTabs [data-baseweb="tab-list"] {gap: 25px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎈 The Gold Bubble (GC=F)")

ticker = "GC=F"
start_date_val = dt(2022, 8, 29)

@st.cache_data
def get_data(ticker, start):
    end = dt.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, multi_level_index=False, auto_adjust=False)
    df.reset_index(inplace=True)
    return df

@st.cache_data
def compute_lppls_parameters(time, price_log):
    observations = np.array([time, price_log])
    lppls_model = lppls_cmaes.LPPLSCMAES(observations=observations)
    params = lppls_model.fit(max_iteration=5000, pop_size=8)
    return params

data = get_data(ticker, start_date_val)

if not data.empty:
    time_ord = [date.toordinal() for date in data['Date']]
    price_log = np.log(data['Adj Close'].values)
    
    with st.spinner('Calculating...'):
        res = compute_lppls_parameters(time_ord, price_log)
        tc, m, w, a, b, c, c1, c2, O, D = res

    tc_date = dt.fromordinal(int(tc))
    
    # --- AJUSTE: Rango de visualización vs Rango de la curva ---
    # 1. El eje X sigue yendo hasta tc + 45 para ver el espacio vacío tras el crash
    end_visual_ordinal = int(tc) + 45
    time_extended_visual = np.array(range(time_ord[0], end_visual_ordinal))
    
    # 2. La curva LPPLS se calcula SOLO hasta tc (truncada)
    time_lppls_fit = np.array(range(time_ord[0], int(tc) + 1))
    
    model_fast = lppls_cmaes.LPPLSCMAES(observations=np.array([time_ord, price_log]))
    fit_log_truncated = model_fast.lppls(time_lppls_fit, tc, m, w, a, b, c1, c2)
    fit_nominal_truncated = np.exp(fit_log_truncated)
    
    # Precio en tc (último valor de la curva truncada)
    price_at_tc = fit_nominal_truncated[-1]

    # Métricas en una sola fila compacta
    col1, col2, col3 = st.columns([2, 2, 5]) # El tercer col es para empujar las métricas a la izquierda
    col1.metric("Bursting Date", tc_date.strftime('%Y-%m-%d'))
    col2.metric("Bursting Price", f"${price_at_tc:,.2f}")

    tab_log, tab_nominal = st.tabs(["📉 Logarithmic", "💵 Nominal (USD)"])

    def create_fig(y_real, y_pred, label):
        fig = go.Figure()
        
        # Fechas para el eje X visual completo (tc + 45)
        dates_visual = [dt.fromordinal(int(o)) for o in time_extended_visual]
        # Fechas solo para la curva LPPLS (hasta tc)
        dates_fit = [dt.fromordinal(int(o)) for o in time_lppls_fit]
        
        # Histórico Real
        fig.add_trace(go.Scatter(x=data['Date'], y=y_real, name="Real Price", line=dict(color="gold", width=1.5)))
        
        # Ajuste LPPLS Truncado (solo hasta tc)
        fig.add_trace(go.Scatter(x=dates_fit, y=y_pred, name="LPPLS Fit", line=dict(color="cyan", dash="dot", width=2)))
        
        # Línea tc vertical
        fig.add_vline(x=tc_date, line_width=2, line_dash="dash", line_color="red")
        
        fig.update_layout(
            template="plotly_dark", 
            height=450, # Altura reducida para evitar scroll
            hovermode="x unified",
            xaxis=dict(range=[dates_visual[0], dates_visual[-1]]), # Forzar rango del eje X
            xaxis_title=None, 
            yaxis_title=label,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h"),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return fig

    with tab_log:
        st.plotly_chart(create_fig(price_log, fit_log_truncated, "Log(Price)"), use_container_width=True, config={'displayModeBar': False})

    with tab_nominal:
        st.plotly_chart(create_fig(data['Adj Close'].values, fit_nominal_truncated, "Price (USD)"), use_container_width=True, config={'displayModeBar': False})

else:
    st.error("No data could be loaded.")