import streamlit as st
from modules import distributions, concepts, testing, regression, time_series_other

st.set_page_config(page_title="Statistika vizuálně", layout="wide")

# --- Založení navigace v Sidebar ---
svg_icon = '''
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px; color: #FF4B4B;">
  <path d="M3 3v18h18"/>
  <path d="M18 17V9"/>
  <path d="M13 17V5"/>
  <path d="M8 17v-3"/>
</svg>
'''
st.markdown(f"<h1 style='display: flex; align-items: center;'>{svg_icon} Statistika vizuálně</h1>", unsafe_allow_html=True)
st.sidebar.header("Navigace a nastavení")

topics = [
    "Normální rozdělení",
    "Rovnoměrné rozdělení",
    "Exponenciální rozdělení",
    "Poissonovo rozdělení",
    "Studentovo t-rozdělení",
    "Chi-kvadrát rozdělení",
    "F-rozdělení",
    "Beta rozdělení",
    "Log-normální rozdělení",
    "Hustota (PDF) vs Distribuční funkce (CDF)",
    "Centrální limitní věta (CLV)",
    "Chyba I. a II. druhu",
    "ANOVA",
    "Lineární regrese",
    "Analýza reziduí",
    "Dekompozice časové řady",
    "Stacionarita",
    "Q-statistika (Míra diverzity)",
    "Intervaly spolehlivosti",
    "Zákon velkých čísel (LLN)",
    "Dvouvýběrový t-test",
    "Korelační analýza",
    "Logistická regrese"
]

selected_topic = st.sidebar.selectbox("Vyberte téma k vizualizaci:", topics)
st.sidebar.markdown("---")
st.sidebar.subheader("Parametry modelu")
st.header(selected_topic)

# --- Routing / Rozcestník do modulů ---
dist_topics = ["Normální rozdělení", "Rovnoměrné rozdělení", "Exponenciální rozdělení", 
               "Poissonovo rozdělení", "Studentovo t-rozdělení", "Chi-kvadrát rozdělení", 
               "F-rozdělení", "Beta rozdělení", "Log-normální rozdělení"]
concept_topics = ["Hustota (PDF) vs Distribuční funkce (CDF)", "Centrální limitní věta (CLV)", "Zákon velkých čísel (LLN)"]
testing_topics = ["Chyba I. a II. druhu", "ANOVA", "Dvouvýběrový t-test", "Intervaly spolehlivosti"]
regression_topics = ["Lineární regrese", "Analýza reziduí", "Korelační analýza", "Logistická regrese"]
ts_other_topics = ["Dekompozice časové řady", "Stacionarita", "Q-statistika (Míra diverzity)"]

if selected_topic in dist_topics:
    distributions.render(selected_topic)
elif selected_topic in concept_topics:
    concepts.render(selected_topic)
elif selected_topic in testing_topics:
    testing.render(selected_topic)
elif selected_topic in regression_topics:
    regression.render(selected_topic)
elif selected_topic in ts_other_topics:
    time_series_other.render(selected_topic)
