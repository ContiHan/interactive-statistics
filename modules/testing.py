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
    if topic == 'Chyba I. a II. druhu':
        st.info("""
        **Co dělají parametry:**
        *   **Alfa (α)**: Hladina významnosti. Riziko chyby I. druhu (odsoudíme nevinného - modrá plocha).
        *   **Efekt (μ1)**: O kolik se liší realita (H1) od naší představy (H0).
        *   **Šum (σ)**: Rozptyl dat. Větší šum zhoršuje schopnost rozpoznat rozdíl.
        *   **Beta (II. druh)**: Riziko, že nepoznáme pravdu (propustíme viníka - červená plocha).
        """)
        mu_alt = st.sidebar.slider('Efekt (μ1):', 0.5, 5.0, 2.5, 0.1)
        alpha = st.sidebar.slider('Alfa (α) - Hladina význ.:', 0.01, 0.20, 0.05, 0.01)
        sigma = st.sidebar.slider('Šum v datech (σ):', 0.5, 2.0, 1.0, 0.1)

        mu0 = 0
        critical_val = stats.norm.ppf(1 - alpha, mu0, sigma)
        x = np.linspace(-6, 12, 1000)
        y0 = stats.norm.pdf(x, mu0, sigma)
        y1 = stats.norm.pdf(x, mu_alt, sigma)
        beta = stats.norm.cdf(critical_val, mu_alt, sigma)
        mask_alpha = x >= critical_val
        mask_beta = x <= critical_val

        fig = plt.figure(figsize=(12, 6))
        plt.plot(x, y0, label='H0 (Nulová)', color='royalblue', lw=2)
        plt.plot(x, y1, label='H1 (Alternativní)', color='crimson', lw=2)
        plt.fill_between(x[mask_alpha], y0[mask_alpha], color='royalblue', alpha=0.4, label=f'Alfa (I. druh) = {alpha:.3f}')
        plt.fill_between(x[mask_beta], y1[mask_beta], color='crimson', alpha=0.4, label=f'Beta (II. druh) = {beta:.3f}')
        plt.axvline(critical_val, color='black', linestyle='--', label='Kritická mez')
        plt.title(f'Testování hypotéz: Síla testu (1-β) = {1-beta:.3f}')
        plt.xlim(-5, 10)
        plt.ylim(0, 0.9)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'ANOVA':
        st.info("""
        **Princip ANOVY (F-statistika):**
        *   **Vnitroskupinový rozptyl (Barevné zvony)**: Šum uvnitř skupin. Chceme ho co nejmenší.
        *   **Meziskupinový rozptyl (Černý šrafovaný zvon)**: Jak moc se od sebe liší průměry. Chceme ho co největší.
        *   Pokud jsou skupiny daleko od sebe (velké μ rozdíly) a mají malý šum (malé σ), F hodnota je velká a rozdíl je významný.
        """)
        mu1 = st.sidebar.slider('μ Skupina 1:', 0.0, 10.0, 4.0, 0.5)
        mu2 = st.sidebar.slider('μ Skupina 2:', 0.0, 10.0, 5.0, 0.5)
        mu3 = st.sidebar.slider('μ Skupina 3:', 0.0, 10.0, 6.0, 0.5)
        sigma = st.sidebar.slider('Společné σ pro všechny:', 1.0, 20.0, 1.0, 1.0)

        x = np.linspace(-20, 40, 1000)
        means = np.array([mu1, mu2, mu3])

        np.random.seed(42)
        groups = [np.random.normal(mu, sigma, 200) for mu in means]
        f_stat, p_val = stats.f_oneway(*groups)

        fig = plt.figure(figsize=(12, 6))
        colors = ['royalblue', 'seagreen', 'indianred']
        for i, (mu, color) in enumerate(zip(means, colors)):
            y = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, y, label=f'Skupina {i+1} (μ={mu})', color=color, lw=2)
            plt.fill_between(x, y, alpha=0.1, color=color)
            plt.axvline(mu, color=color, linestyle='--', alpha=0.5)

        mean_of_means = np.mean(means)
        std_of_means = np.std(means, ddof=1) if len(means) > 1 else 0.1
        x_means = np.linspace(-20, 40, 500)
        y_means = stats.norm.pdf(x_means, mean_of_means, std_of_means)

        plt.plot(x_means, y_means, color='black', lw=3, label='Rozdělení průměrů (Meziskupinové)')
        plt.fill_between(x_means, y_means, alpha=0.2, color='black', hatch='//')

        sig_text = 'p < 0.05 → Signifikantní rozdíl' if p_val < 0.05 else 'p ≥ 0.05 → Nesignifikantní'
        title_color = 'darkred' if p_val < 0.05 else 'steelblue'
        plt.title(f'ANOVA: F = {f_stat:.2f}, p = {p_val:.4f} | {sig_text}', color=title_color)
        plt.xlim(-10, 20)
        plt.ylim(0, 0.5)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Dvouvýběrový t-test':
        st.info("""
        **Dvouvýběrový t-test**
        Slouží k ověření, zda je rozdíl mezi dvěma skupinami skutečný, nebo jestli vznikl jen obrovskou náhodou.

        *   **Příklad ze života:** Skupina A dostala nový lék na krevní tlak, Skupina B dostala placebo (lentilku).
        *   **Interpretace p-hodnoty:** 
            * **Zelená (p < 0.05)**: Signifikantní rozdíl. Zvony se překrývají málo. Lék prokazatelně funguje.
            * **Červená (p ≥ 0.05)**: Rozdíl mohl vzniknout náhodou. Zvony se překrývají příliš. Lék nefunguje o nic lépe než placebo.
        *   **Vliv parametrů:** Když na panelu vlevo zvýšíte šum (rozptyl), zvony se rozpliznou. I když budou mít stále stejné průměry, p-hodnota přestane být signifikantní.
        """)
        mu1 = st.sidebar.slider('Průměr A:', 0.0, 10.0, 4.0, 0.5)
        mu2 = st.sidebar.slider('Průměr B:', 0.0, 10.0, 5.0, 0.5)
        sigma = st.sidebar.slider('Společný šum (Rozptyl):', 0.5, 5.0, 1.0, 0.1)
        n_obs = st.sidebar.slider('Počet měření ve skupině:', 5, 200, 30, 5)

        np.random.seed(42)
        group_a = np.random.normal(mu1, sigma, n_obs)
        group_b = np.random.normal(mu2, sigma, n_obs)

        t_stat, p_val = stats.ttest_ind(group_a, group_b)

        fig = plt.figure(figsize=(10, 6))
        x = np.linspace(-5, 15, 1000)
        plt.plot(x, stats.norm.pdf(x, mu1, sigma), color='royalblue', lw=2, label=f'Skupina A (Teorie μ={mu1})')
        plt.fill_between(x, stats.norm.pdf(x, mu1, sigma), alpha=0.2, color='royalblue')

        plt.plot(x, stats.norm.pdf(x, mu2, sigma), color='crimson', lw=2, label=f'Skupina B (Teorie μ={mu2})')
        plt.fill_between(x, stats.norm.pdf(x, mu2, sigma), alpha=0.2, color='crimson')

        # Kreslení bodů naspod
        plt.scatter(group_a, np.zeros_like(group_a) + 0.02, color='darkblue', alpha=0.5, s=20)
        plt.scatter(group_b, np.zeros_like(group_b) + 0.04, color='darkred', alpha=0.5, s=20)

        sig_text = "SIGNIFIKANTNÍ ROZDÍL" if p_val < 0.05 else "NESIGNIFIKANTNÍ (Náhoda)"
        title_color = 'seagreen' if p_val < 0.05 else 'firebrick'

        plt.title(f'T-test: t = {t_stat:.2f} | p-hodnota = {p_val:.4f}\n{sig_text}', color=title_color)
        plt.xlim(-5, 15)
        plt.ylim(0, 0.8)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Intervaly spolehlivosti':
        st.info("""
        **Interval spolehlivosti (Confidence Interval)**
        Ukazuje, jak přesný je náš odhad a s jakou jistotou v něm leží skutečná hodnota.

        *   **Příklad ze života:** Chceme zjistit průměrnou výšku mužů v ČR (skutečný průměr je černá čára, kterou ale v reálu neznáme). Změříme 100 náhodných mužů a spočítáme průměr. Protože jsme nezměřili všechny, musíme přidat "rezervu" (šířka úsečky). 
        *   **95% hladina:** Znamená, že pokud bychom tento průzkum zopakovali 100x (což tady děláme), 95krát se skutečná výška populace trefí do naší naměřené rezervy (modrá čára) a 5krát (červená čára) budeme mít prostě smůlu na extrémní vzorek (třeba zrovna změříme partu basketbalistů).
        """)
        n_samples = st.sidebar.slider('Velikost každého vzorku:', 10, 200, 50, 10)
        conf_level = st.sidebar.selectbox('Hladina spolehlivosti:', [0.90, 0.95, 0.99], index=1)

        np.random.seed(42)
        true_mean = 0
        true_std = 1
        n_experiments = 100

        samples = np.random.normal(true_mean, true_std, (n_experiments, n_samples))
        means = np.mean(samples, axis=1)
        stds = np.std(samples, axis=1, ddof=1)

        t_crit = stats.t.ppf((1 + conf_level) / 2, df=n_samples-1)
        margins = t_crit * (stds / np.sqrt(n_samples))

        lower_bounds = means - margins
        upper_bounds = means + margins

        fig = plt.figure(figsize=(10, 6))
        plt.axvline(true_mean, color='black', lw=2, linestyle='--', label='Skutečný průměr')

        for i in range(n_experiments):
            if lower_bounds[i] <= true_mean <= upper_bounds[i]:
                color = 'royalblue'
            else:
                color = 'crimson'
            plt.plot([lower_bounds[i], upper_bounds[i]], [i, i], color=color, lw=2, alpha=0.7)
            plt.plot(means[i], i, 'o', color=color, markersize=3)

        plt.title(f'{int(conf_level*100)}% Intervaly spolehlivosti pro {n_experiments} nezávislých vzorků')
        plt.yticks([])
        plt.xlim(-1.5, 1.5)
        plt.ylim(-2, 102)
        plt.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    elif topic == 'Neparametrické testy (Mann-Whitney)':
        st.info("""
        **Mann-Whitney U test (Neparametrický)**
        Co dělat, když data nemají krásný tvar zvonu (normální rozdělení) nebo obsahují extrémní výkyvy (outliery)? Klasický t-test by selhal.
        
        *   **Příklad ze života:** Porovnáváme platy ve Firmě A a Firmě B. V obou berou lidé zhruba stejně (kolem 40 tisíc). Ale do Firmy B najednou nastoupí miliardář (extrém). Klasický průměr Firmy B nesmyslně vyletí. T-test by chybně ohlásil, že se platy ve firmách drasticky liší.
        *   **Řešení:** Mann-Whitney U test zahodí reálné částky a nahradí je **Pořadím** (Ranky). Miliardář dostane prostě pořadí "Nejbohatší" (nejvyšší číslo ranku), takže velikost jeho majetku test nezkreslí.
        *   **Grafy:** Všimněte si, že i když je vlevo plat miliardáře mimo graf, T-test to rozhodí. Graf vpravo ukazuje, jak rankování outliera zneškodní.
        """)
        
        add_outlier = st.sidebar.checkbox('Přidat miliardáře (Outliera) do Skupiny B', value=True)
        
        np.random.seed(42)
        group_a = np.random.lognormal(mean=10.5, sigma=0.5, size=50) # Běžné platy
        group_b = np.random.lognormal(mean=10.5, sigma=0.5, size=50)
        
        if add_outlier:
            group_b[-1] = 50000000 # Miliardář
            
        t_stat, t_pval = stats.ttest_ind(group_a, group_b)
        u_stat, u_pval = stats.mannwhitneyu(group_a, group_b)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Obyčejná data (Log scale for visualization since outlier is huge)
        ax1.boxplot([group_a, group_b], labels=['Skupina A', 'Skupina B'])
        ax1.set_title(f'Původní data (Log. měřítko)\nKlasický T-test p-hod. = {t_pval:.4f}')
        ax1.set_ylabel('Plat (Kč)')
        ax1.set_yscale('log')
        
        # Rankovaná data
        all_data = np.concatenate([group_a, group_b])
        ranks = stats.rankdata(all_data)
        ranks_a = ranks[:50]
        ranks_b = ranks[50:]
        
        ax2.boxplot([ranks_a, ranks_b], labels=['Skupina A', 'Skupina B'])
        ax2.set_title(f'Data převedená na Pořadí (Ranky)\nMann-Whitney p-hod. = {u_pval:.4f}')
        ax2.set_ylabel('Pořadí (1 = nejchudší, 100 = nejbohatší)')
        
        st.pyplot(fig)
        plt.close(fig)

