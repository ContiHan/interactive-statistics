import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression

plt.style.use('ggplot')

def _plot_dist(x, y, title, color, label=None, xlim=None, ylim=None, figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, lw=2, color=color, label=label)
    plt.fill_between(x, y, alpha=0.2, color=color)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Hustota pravděpodobnosti')
    if label is not None:
        plt.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

def render(topic):
    if topic == 'Lineární regrese':
        st.info("""
        **Sklon** určuje strmost přímky, **Průsečík** bod na ose Y a **Šum** rozptyl bodů kolem přímky.
        Koeficient determinace **R²** ukazuje, jak dobře model (červená čára) data (modré body) vysvětluje (1.0 je dokonalost).
        """)
        slope = st.sidebar.slider('Sklon (Slope):', -2.0, 2.0, 1.5, 0.1)
        intercept = st.sidebar.slider('Průsečík (Intercept):', -5.0, 5.0, 2.0, 0.1)
        noise = st.sidebar.slider('Šum v datech:', 0.1, 10.0, 2.0, 0.1)

        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = slope * x + intercept + np.random.normal(0, noise, 50)
        res = stats.linregress(x, y)

        fig = plt.figure(figsize=(10, 5))
        plt.scatter(x, y, alpha=0.6, color='slateblue')
        plt.plot(x, res.slope * x + res.intercept, color='crimson', lw=2, label=f'Model, R² = {res.rvalue**2:.3f}')
        plt.title('Lineární regrese')
        plt.xlim(-1, 11)
        plt.ylim(-35, 35)
        plt.legend(loc='upper left')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Analýza reziduí':
        st.info("""
        **Předpoklady regresních modelů (Rezidua = Chyby predikce):**
        *   **Střed (Nula)**: Model by neměl systematicky podstřelovat nebo přestřelovat.
        *   **Homoskedasticita (Základní rozptyl)**: Stejně tlustý pruh chyb napříč grafem je ideál. 'Trychtýř' (Heteroskedasticita) je špatně.
        *   **Normalita**: Většina chyb by měla být malých (kolem nuly) a velké chyby jen vzácně.
        """)
        mean_error = st.sidebar.slider('Střed chyby (Systematická ch.):', -5.0, 5.0, 0.0, 0.5)
        scale_error = st.sidebar.slider('Zákl. rozptyl:', 0.5, 10.0, 2.0, 0.5)
        heteroskedasticity = st.sidebar.slider('Trychtýř (Heteroskedasticita):', 0.0, 10.0, 0.0, 0.5)
        normality = st.sidebar.checkbox('Normální rozdělení chyb', value=True)

        np.random.seed(42)
        x = np.linspace(0, 100, 200)
        if normality:
            errors = np.random.normal(loc=mean_error, scale=scale_error, size=200)
        else:
            errors = np.random.exponential(scale=scale_error, size=200) - scale_error
        if heteroskedasticity > 0:
            noise_growth = (x / 100) * heteroskedasticity
            errors = errors * (1 + noise_growth)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.scatter(x, errors, alpha=0.6, color='darkorchid')
        ax1.axhline(0, color='black', linestyle='--', lw=1)
        ax1.set_title('Graf reziduí (Závislost na čase/hodnotě)')
        ax1.set_xlim(-5, 105)
        ax1.set_ylim(-60, 60)

        sns.histplot(errors, kde=True, ax=ax2, color='darkorchid', alpha=0.4)
        ax2.set_title('Rozdělení reziduí (Histogram)')
        ax2.set_xlim(-60, 60)
        ax2.set_ylim(0, 60) 
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Korelační analýza':
        st.info("""
        **Korelace (Pearsonův koeficient r)**
        Ukazuje sílu a směr lineární závislosti dvou veličin.

        *   **Příklady ze života:**
            * **r = 1 (Přímá úměra):** Čas strávený během a spálené kalorie. Úhledná čára nahoru.
            * **r = -1 (Nepřímá úměra):** Čas strávený na mobilu v noci a kvalita spánku. Úhledná čára dolů.
            * **r = 0 (Žádná závislost):** Počet snědených rohlíků a výsledek IQ testu. Náhodný shluk (brokovnice).
        *   **Tip:** Zkuste nastavit $r = 0.6$. Všimněte si, že i středně silná korelace vypadá na pohled stále docela zmateně!
        """)
        r = st.sidebar.slider('Korelační koeficient (r):', -1.0, 1.0, 0.6, 0.05)
        n_points = st.sidebar.slider('Počet bodů:', 10, 1000, 200, 10)

        np.random.seed(42)
        cov_matrix = [[1, r], [r, 1]]
        data = np.random.multivariate_normal([0, 0], cov_matrix, n_points)
        x_data, y_data = data[:, 0], data[:, 1]

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(x_data, y_data, color='teal', alpha=0.6, edgecolor='w')

        # Zafixování os a přidání mřížky pro lepší vnímání rotace mračna
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.axhline(0, color='black', lw=1, ls='--')
        plt.axvline(0, color='black', lw=1, ls='--')
        plt.title(f'Korelační analýza (Pearson r = {r:.2f})')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Logistická regrese':
        st.info("""
        **Logistická regrese**
        Modeluje pravděpodobnost, že objekt patří do určité kategorie (Třída 0 nebo 1). Na rozdíl od lineární regrese se predikce bezpečně vlní od 0 % do 100 % (S-křivka).

        *   **Příklad ze života:** Snažíme se předpovědět, zda student složí zkoušku (Ano/Ne), na základě počtu hodin učení (osa X).
            * **Modré body (Dole):** Studenti, co zkoušku neudělali.
            * **Červené body (Nahoře):** Studenti, co zkoušku udělali.
            * **Černá S-křivka:** Ukazuje plynulou *pravděpodobnost* složení zkoušky v závislosti na čase učení.
        *   **Rozhodovací hranice (50 %):** Místo, od kterého už model říká "Tenhle student to pravděpodobně zvládne".
        """)
        sep = st.sidebar.slider('Vzdálenost tříd (Oddělitelnost):', 0.0, 10.0, 3.0, 0.5)

        np.random.seed(42)
        n_points = 100
        # X hodnoty pro Třídu 0 (kolem 3) a Třídu 1 (kolem 3 + sep)
        X_0 = np.random.normal(3, 1.5, n_points)
        X_1 = np.random.normal(3 + sep, 1.5, n_points)

        X_all = np.concatenate([X_0, X_1])
        y_all = np.concatenate([np.zeros(n_points), np.ones(n_points)])

        # Jednoduchá logistická křivka (fit the model via sklearn)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_all.reshape(-1, 1), y_all)

        X_test = np.linspace(-5, 20, 300).reshape(-1, 1)
        y_prob = model.predict_proba(X_test)[:, 1]

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(X_0, np.zeros(n_points) + 0.02, color='royalblue', alpha=0.6, label='Třída 0 (Ne)')
        plt.scatter(X_1, np.ones(n_points) - 0.02, color='crimson', alpha=0.6, label='Třída 1 (Ano)')

        plt.plot(X_test, y_prob, color='black', lw=3, label='Logistická křivka (Pravděpodobnost Třídy 1)')
        plt.axhline(0.5, color='gray', linestyle='--', label='Rozhodovací hranice (50 %)')

        boundary = -model.intercept_[0] / model.coef_[0][0] if model.coef_[0][0] != 0 else 0
        plt.axvline(boundary, color='gray', linestyle=':')

        plt.title('Logistická regrese (Klasifikace)')
        plt.xlabel('Závislá proměnná (X)')
        plt.ylabel('Pravděpodobnost (0 až 1)')
        plt.xlim(-2, 15)
        plt.ylim(-0.1, 1.1)
        plt.legend(loc='center left')
        st.pyplot(fig)
        plt.close(fig)

