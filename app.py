import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Statistika vizuálně", layout="wide")
plt.style.use('ggplot')

# --- Pomocné funkce ---
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

# --- Založení navigace v Sidebar ---
st.title("Statistika vizuálně")
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
    "Q-statistika (Míra diverzity)"
]

selected_topic = st.sidebar.selectbox("Vyberte téma k vizualizaci:", topics)
st.sidebar.markdown("---")
st.sidebar.subheader("Parametry modelu")

st.header(selected_topic)

# --- Jednotlivá témata ---

if selected_topic == "Normální rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Střed (μ)**: Posouvá celý zvon doleva nebo doprava po ose x.
    *   **Odchylka (σ)**: Určuje, jak je zvon široký. Malá sigma znamená úzký a vysoký zvon, velká sigma plochý.
    
    **Kdy se používá:** Pro modelování přirozených jevů, kde se většina hodnot nachází kolem průměru.
    
    **Reálný příklad:** Výška dospělých lidí v populaci nebo chyby při měření.
    """)
    mu = st.sidebar.slider('Střed (μ):', -5.0, 5.0, 0.0, 0.1)
    sigma = st.sidebar.slider('Odchylka (σ):', 0.1, 5.0, 1.0, 0.1)
    
    x = np.linspace(-25, 25, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    # Fixní osy zohledňující extrémy parametrů
    _plot_dist(x, y, 'Normální rozdělení', 'royalblue', label=f'μ={mu}, σ={sigma}', xlim=(-25, 25), ylim=(0, 4.0))

elif selected_topic == "Rovnoměrné rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Minimum (a)** a **Maximum (b)**: Definují hranice, mezi kterými mají všechny hodnoty naprosto stejnou šanci na výskyt.
    
    **Kdy se používá:** Když víme, že hodnota leží v určitém intervalu, ale nemáme důvod upřednostnit žádnou část tohoto intervalu.
    
    **Reálný příklad:** Hod kostkou (diskrétní verze) nebo čekání na tramvaj, která jezdí v přesných intervalech.
    """)
    a = st.sidebar.slider('Minimum (a):', 0.0, 5.0, 0.0, 0.1)
    b = st.sidebar.slider('Maximum (b):', 5.1, 10.0, 5.0, 0.1)
    
    if b <= a: b = a + 0.1
    x = np.linspace(-2, 12, 1000)
    y = stats.uniform.pdf(x, loc=a, scale=b - a)
    _plot_dist(x, y, 'Rovnoměrné rozdělení', 'seagreen', label=f'a={a}, b={b}', xlim=(-2, 12), ylim=(0, 10.5))

elif selected_topic == "Exponenciální rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Intenzita (λ)**: Vyšší lambda znamená, že události následují rychleji za sebou a graf klesá strměji.
    
    **Kdy se používá:** Pro modelování času mezi náhodnými událostmi.
    
    **Reálný příklad:** Čas mezi příchody zákazníků do prodejny nebo životnost elektronické součástky.
    """)
    lam = st.sidebar.slider('Intenzita (λ):', 0.1, 5.0, 1.0, 0.1)
    A = st.sidebar.slider('Start (A):', 0.0, 5.0, 0.0, 0.1)
    
    x = np.linspace(0, 30, 1000)
    y = stats.expon.pdf(x, loc=A, scale=1 / lam)
    _plot_dist(x, y, 'Exponenciální rozdělení', 'indianred', label=f'λ={lam}, A={A}', xlim=(0, 20), ylim=(0, 5.5))

elif selected_topic == "Poissonovo rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Intenzita (λ)**: Průměrný počet výskytů události za jednotku času nebo prostoru. Vyšší lambda posouvá těžiště grafu doprava.
    
    **Kdy se používá:** Pro počítání náhodných událostí v pevném intervalu.
    
    **Reálný příklad:** Počet e-mailů, které obdržíte za hodinu, nebo počet kazů na metru čtverečním látky.
    """)
    lam = st.sidebar.slider('Lambda (λ):', 0.1, 15.0, 4.0, 0.1)
    
    k = np.arange(0, 40)
    pmf = stats.poisson.pmf(k, lam)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(k, pmf, color='darkorange', alpha=0.7)
    plt.title('Poissonovo rozdělení')
    plt.xlim(-1, 31)
    plt.ylim(0, 0.5)
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Studentovo t-rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Stupně volnosti (df)**: Určují tloušťku "chvostů" rozdělení. S rostoucím počtem stupňů volnosti se t-rozdělení blíží normálnímu rozdělení.
    
    **Kdy se používá:** Při odhadu průměru populace, když máme malý vzorek a neznáme skutečný rozptyl.
    
    **Reálný příklad:** Testování, zda má nový lék vliv na krevní tlak u malé skupiny pacientů.
    """)
    df = st.sidebar.slider('St. volnosti (df):', 1, 50, 1)
    
    x = np.linspace(-10, 10, 1000)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, stats.t.pdf(x, df), color='royalblue', lw=2, label=f't-dist (df={df})')
    plt.plot(x, stats.norm.pdf(x), color='crimson', linestyle='--', alpha=0.6, label='Normální (referenční)')
    plt.title(f'Studentovo t-rozdělení (df={df})')
    plt.xlim(-5, 5)
    plt.ylim(0, 0.45)
    plt.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Chi-kvadrát rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Stupně volnosti (df)**: Určují tvar rozdělení. Pro malá df je rozdělení silně zešikmené doprava. S rostoucím df se začíná podobat normálnímu rozdělení a posouvá se doprava.
    
    **Kdy se používá:** Při testování nezávislosti v kontingenčních tabulkách, dobré shody nebo odhadování rozptylu.
    
    **Reálný příklad:** Testování, zda barva očí souvisí s preferencí politické strany.
    """)
    df = st.sidebar.slider('St. volnosti (df):', 1, 20, 1)
    
    x = np.linspace(0.1, 40, 1000)
    y = stats.chi2.pdf(x, df)
    _plot_dist(x, y, 'Chi-kvadrát rozdělení', 'darkmagenta', label=f'df={df}', xlim=(0, 40), ylim=(0, 0.5))

elif selected_topic == "F-rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Stupně volnosti (df1 a df2)**: Určují tvar rozdělení. df1 patří čitateli a df2 jmenovateli. Typicky je rozdělení zešikmené doprava.
    
    **Kdy se používá:** Primárně pro **testování shody rozptylů** dvou souborů a jako základ pro ANOVA testy.
    
    **Reálný příklad:** Porovnání přesnosti dvou různých strojů ve výrobě.
    """)
    df1 = st.sidebar.slider('df1:', 1, 100, 5)
    df2 = st.sidebar.slider('df2:', 1, 100, 10)
    
    x = np.linspace(0.001, 5, 1000)
    y = stats.f.pdf(x, df1, df2)
    _plot_dist(x, y, 'F-rozdělení', 'sienna', label=f'df1={df1}, df2={df2}', xlim=(0, 5), ylim=(0, 2.5))

elif selected_topic == "Beta rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Alfa (α)** a **Beta (β)**: Tvarové parametry. Pokud jsou obě větší než 1, je rozdělení uprostřed vyklenuté. Pokud je α > β, je hustota nakloněna k 1, při β > α k 0.
    
    **Kdy se používá:** Pro modelování chování procentuálních podílů nebo pravděpodobností.
    
    **Reálný příklad:** Odhad procentuální míry prokliku (CTR) u reklamy.
    """)
    a = st.sidebar.slider('Alpha (α):', 0.1, 10.0, 2.0, 0.1)
    b = st.sidebar.slider('Beta (β):', 0.1, 10.0, 5.0, 0.1)
    
    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, a, b)
    _plot_dist(x, y, 'Beta rozdělení', 'darkviolet', label=f'α={a}, β={b}', xlim=(0, 1), ylim=(0, 5.0))

elif selected_topic == "Log-normální rozdělení":
    st.info("""
    **Co dělají parametry:**
    *   **Mu (μ)** a **Sigma (σ)**: Parametry odpovídající průměru a odchylce logaritmu dané proměnné. Způsobují zešikmení grafu.
    
    **Kdy se používá:** Pro hodnoty, které nemohou být záporné a mají dlouhý pravý chvost.
    
    **Reálný příklad:** Rozdělení příjmů v populaci nebo délka hovorů v call centru.
    """)
    mu = st.sidebar.slider('Mu (μ):', -1.0, 2.0, 0.0, 0.1)
    sigma = st.sidebar.slider('Sigma (σ):', 0.1, 1.5, 0.5, 0.1)
    
    x = np.linspace(0.01, 25, 1000)
    y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    _plot_dist(x, y, 'Log-normální rozdělení', 'teal', label=f'μ={mu}, σ={sigma}', xlim=(0, 20), ylim=(0, 1.5))

elif selected_topic == "Hustota (PDF) vs Distribuční funkce (CDF)":
    st.info("""
    **Co uvidíte v grafech:**
    *   **Hustota (PDF - vlevo)**: Ukazuje šanci na výskyt konkrétní hodnoty. Vybarvená plocha pod křivkou od -∞ do bodu x představuje celkovou pravděpodobnost.
    *   **Distribuční f. (CDF - vpravo)**: Ukazuje kumulativní pravděpodobnost. Hodnota na ose Y v bodě x přesně odpovídá velikosti vybarvené plochy v levém grafu.
    
    **Kdy se to hodí:** Když potřebujete vědět, jaká je šance, že výsledek bude menší nebo roven určité hodnotě.
    """)
    dist_type = st.sidebar.radio("Rozdělení:", ['Normální', 'Exponenciální'])
    
    if dist_type == 'Normální':
        mu = st.sidebar.slider('Střed (μ):', -5.0, 5.0, 0.0)
        sigma = st.sidebar.slider('Odchylka (σ):', 0.5, 3.0, 1.0)
        lam = 1
        x_val = st.sidebar.slider('Bod x (kde chci zjistit plochu):', -15.0, 15.0, float(mu), 0.1)
    else:
        mu, sigma = 0, 1
        lam = st.sidebar.slider('Intenzita (λ):', 0.1, 5.0, 1.0)
        x_val = st.sidebar.slider('Bod x (kde chci zjistit plochu):', 0.0, 50.0, 1.0, 0.1)

    if dist_type == 'Normální':
        dist = stats.norm(loc=mu, scale=sigma)
        x = np.linspace(-20, 20, 500)
        start_x = -20
        x_limit, y_limit = (-15, 15), (0, 1.0)
    else:
        dist = stats.expon(scale=1/lam)
        x = np.linspace(0, 60, 500)
        start_x = 0
        x_limit, y_limit = (0, 40), (0, 5.5)
        
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    current_cdf = dist.cdf(x_val)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(x, pdf, lw=2, color='royalblue', label='Hustota (PDF)')
    # Fix x_val visualization limits so fill area doesn't break if x_val is out of theoretical bounds
    x_val_capped = min(max(x_val, start_x), x[-1])
    x_fill = np.linspace(start_x, x_val_capped, 100)
    ax1.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, color='royalblue')
    ax1.axvline(x_val, color='crimson', linestyle='--')
    ax1.set_title(f'Hustota (Plocha = {current_cdf:.3f})')
    ax1.set_xlabel('x')
    ax1.set_xlim(*x_limit)
    ax1.set_ylim(*y_limit)
    ax1.legend()
    
    ax2.plot(x, cdf, lw=2, color='seagreen', label='Distribuční f. (CDF)')
    ax2.axvline(x_val, color='crimson', linestyle='--')
    ax2.axhline(current_cdf, color='crimson', alpha=0.3, linestyle=':')
    ax2.scatter([x_val], [current_cdf], color='crimson')
    ax2.set_title(f'Distribuční funkce (F(x) = {current_cdf:.3f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_xlim(*x_limit)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Centrální limitní věta (CLV)":
    st.info("""
    **Jak se za jakéhokoli rozdělení stane normální rozdělení (zvon)**:
    *   **Velikost (n)**: Počet prvků pro výpočet jednoho průměru (typicky potřebujeme n ≥ 30).
    *   **Počet simulací**: Kolikrát provedeme výběr a spočítáme průměr. Čím více, tím jasnější zvon.
    """)
    dist_type = st.sidebar.radio("Základní rozdělení:", ['Rovnoměrné (Kostky)', 'Exponenciální (Čekání)'])
    n_samples = st.sidebar.slider('Velikost výběru (n):', 30, 100, 30, 10)
    n_simulations = st.sidebar.slider('Počet simulací:', 10, 2000, 100, 10)
    
    np.random.seed(42)
    if 'Rovnoměrné' in dist_type:
        samples = np.random.uniform(0, 1, (n_simulations, n_samples))
        mu_theory, sigma_theory = 0.5, np.sqrt(1 / (12 * n_samples))
        x_limit, y_limit = (0.2, 0.8), (0, 12)
    else:
        samples = np.random.exponential(scale=1.0, size=(n_simulations, n_samples))
        mu_theory, sigma_theory = 1.0, 1.0 / np.sqrt(n_samples)
        x_limit, y_limit = (0.4, 1.8), (0, 4.0)
        
    means = samples.mean(axis=1)
    dynamic_bins = int(np.clip(np.sqrt(n_simulations) * 1.8 + n_samples / 3, 15, 100))
    
    fig = plt.figure(figsize=(10, 6))
    plt.hist(means, bins=dynamic_bins, density=True, alpha=0.6, color='skyblue', label='Simulované průměry')
    x = np.linspace(x_limit[0]-1, x_limit[1]+1, 200)
    plt.plot(x, stats.norm.pdf(x, mu_theory, sigma_theory), 'r', lw=2, label='Teoretické N (CLT)')
    plt.title(f'CLT: Distribuce {n_simulations} průměrů (n={n_samples})')
    plt.xlim(*x_limit)
    plt.ylim(*y_limit)
    plt.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Chyba I. a II. druhu":
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

elif selected_topic == "ANOVA":
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

elif selected_topic == "Lineární regrese":
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

elif selected_topic == "Analýza reziduí":
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

elif selected_topic == "Dekompozice časové řady":
    st.info("""
    **Složky časové řady:**
    *   **Trend (Sklon)**: Dlouhodobý směr dat (růst nebo pokles).
    *   **Sezónnost (Amplituda)**: Krátkodobé, pravidelně se opakující výkyvy (např. vánoční nákupy - trvají 1 rok).
    *   **Cyklus (Amplituda)**: Dlouhodobé vlny (např. hospodářský cyklus).
    *   **Rezidua (Šum)**: Náhodné a nepředvídatelné kolísání.
    """)
    trend_slope = st.sidebar.slider('Trend:', -3.0, 3.0, 0.5, 0.05)
    seasonal_amp = st.sidebar.slider('Sezónnost (roční):', 0, 50, 10)
    cycle_amp = st.sidebar.slider('Cyklus (víceletý):', 0, 50, 15)
    noise_level = st.sidebar.slider('Šum:', 0.0, 20.0, 2.0, 0.5)
    
    t = np.arange(120)
    dates = pd.date_range(start='2015-01-01', periods=120, freq='MS')
    trend = trend_slope * t
    seasonality = seasonal_amp * np.sin(2 * np.pi * t / 12)
    cycle = cycle_amp * np.sin(2 * np.pi * t / (12 * 4))
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, len(t))
    total = trend + seasonality + cycle + noise
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    components = [
        (total, 'Výsledná řada (Součet všech níže)', 'black', (-400, 500)), 
        (trend, 'Trend', 'royalblue', (-400, 400)),
        (seasonality, 'Sezónnost', 'teal', (-60, 60)), 
        (cycle, 'Cyklus', 'goldenrod', (-60, 60)),
        (noise, 'Rezidua (Šum)', 'firebrick', (-60, 60))
    ]
    for i, (ax, (data, title, color, y_lim)) in enumerate(zip(axes, components)):
        ax.plot(dates, data, color=color, lw=1.5 if i==0 else 1.2)
        ax.set_title(title, fontsize=10, pad=5)
        ax.grid(True, alpha=0.15)
        ax.set_ylim(*y_lim)
    plt.tight_layout(h_pad=0.4)
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Stacionarita":
    st.info("""
    **Stacionarita (pravidla hry se nemění)** znamená, že časová řada má konstantní průměr, rozptyl a chová se stabilně.
    *   **Stacionární (φ < 1)**: Jako Bungee jumper. Vždycky se vrátí k průměru.
    *   **Jednotkový kořen (φ = 1)**: Jako Opilec na louce (Random walk). Kam šlápne, tam stojí, průměr nedává smysl.
    *   **Explozivní (φ > 1)**: Raketa, čísla okamžitě letí do extrémů.
    """)
    phi = st.sidebar.slider('Koeficient paměti (φ):', 0.0, 1.03, 0.5, 0.01)
    noise_std = st.sidebar.slider('Šum:', 0.1, 5.0, 1.0, 0.1)
    
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(0, noise_std)
    
    try:
        res = adfuller(y)
        p_val = res[1]
    except Exception:
        p_val = 1.0 # Fallback for explosive series where ADF fails
    
    fig = plt.figure(figsize=(12, 5))
    plt.plot(y, color='darkblue', lw=1.5)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    
    # Pro stacionaritu a random walk udržíme fixní okno, pro explozi ať to raději uletí
    plt.ylim(-50, 50)
    plt.xlim(0, 200)
    
    status = "STACIONÁRNÍ" if phi < 1 else "NESTACIONÁRNÍ (Random Walk / Explozivní)"
    plt.title(f"Autoregresní proces AR(1) s φ={phi} | p-hodnota (ADF): {p_val:.4f}\n{status}")
    st.pyplot(fig)
    plt.close(fig)

elif selected_topic == "Q-statistika (Míra diverzity)":
    st.info("""
    Q-statistika měří, jak moc jsou chyby dvou modelů závislé (pro Ensemble strojové učení).
    *   **Q ≈ 0**: Modely se pletou nezávisle na sobě (ideál).
    *   **Q > 0**: Kladná korelace. Když se splete A, splete se často i B (špatně).
    *   **Q < 0**: Záporná korelace. Modely se doplňují.
    """)
    n_common_correct = st.sidebar.slider('Oba správně (a):', 0, 100, 60)
    n_only_b_wrong = st.sidebar.slider('A správně, B chyba (b):', 0, 50, 10)
    n_only_a_wrong = st.sidebar.slider('A chyba, B správně (c):', 0, 50, 10)
    n_both_wrong = st.sidebar.slider('Oba chyba (d):', 0, 100, 20)
    
    a, b, c, d = n_common_correct, n_only_b_wrong, n_only_a_wrong, n_both_wrong
    denominator = a * d + b * c
    q_val = (a * d - b * c) / denominator if denominator != 0 else 0
    data = [[a, b], [c, d]]
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=['B Správně', 'B Chyba'], yticklabels=['A Správně', 'A Chyba'], vmin=0, vmax=100)
    plt.title(f'Matice chyb\nQ-statistika = {q_val:.3f}')
    st.pyplot(fig)
    plt.close(fig)
