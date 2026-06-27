import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

def render(topic):
    if topic == 'Lineární programování':
        st.info("""
        **Lineární programování (Operační výzkum)**
        Slouží k hledání optimálního řešení při omezených zdrojích.
        
        *   **Příklad ze života:** Továrna vyrábí Stoly (osa X) a Židle (osa Y). Stůl přináší větší zisk, ale spotřebuje více dřeva a hodin práce. Kolik máme vyrobit od každého, abychom maximalizovali zisk a nepřekročili sklady?
        *   **Jak číst graf:** Zelená plocha (**Mnohoúhelník přípustných řešení**) představuje všechny fyzicky proveditelné výrobní plány. Zlatá tečkovaná čára (Izozisk) stoupá vzhůru, dokud se neodtrhne od zelené plochy. Bod odtržení je **Optimum**.
        """)
        
        c1 = st.sidebar.slider('Zisk ze Stolu (Kč):', 100, 1000, 500, 50)
        c2 = st.sidebar.slider('Zisk ze Židle (Kč):', 100, 1000, 400, 50)
        
        mat_cap = st.sidebar.slider('Sklad dřeva (ks):', 10, 200, 120, 10)
        mat_x = st.sidebar.slider('Dřevo na 1 Stůl (ks):', 1, 10, 2, 1)
        mat_y = st.sidebar.slider('Dřevo na 1 Židli (ks):', 1, 10, 1, 1)
        
        lab_cap = st.sidebar.slider('Čas zaměstnanců (hod):', 10, 200, 100, 10)
        lab_x = st.sidebar.slider('Hodin na 1 Stůl:', 1, 10, 1, 1)
        lab_y = st.sidebar.slider('Hodin na 1 Židli:', 1, 10, 2, 1)
        
        x = np.linspace(0, 250, 400) # Výpočtový x-range raději širší
        
        y1 = (mat_cap - mat_x * x) / mat_y
        y2 = (lab_cap - lab_x * x) / lab_y
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x, y1, label='Omezení materiálu', color='crimson', lw=2)
        ax.plot(x, y2, label='Omezení práce', color='royalblue', lw=2)
        
        # Intersekce podminek
        vertices = [(0, 0)]
        if lab_cap/lab_y >= 0: vertices.append((0, lab_cap/lab_y))
        if mat_cap/mat_y >= 0 and mat_cap/mat_y < lab_cap/lab_y: vertices[-1] = (0, mat_cap/mat_y)
        
        det = mat_x*lab_y - mat_y*lab_x
        if det != 0:
            ix = (mat_cap*lab_y - mat_y*lab_cap) / det
            iy = (mat_x*lab_cap - mat_cap*lab_x) / det
            if ix >= 0 and iy >= 0:
                vertices.append((ix, iy))
        
        rest_x = min(mat_cap/mat_x, lab_cap/lab_x)
        if rest_x >= 0: vertices.append((rest_x, 0))
        
        valid_vertices = []
        for vx, vy in vertices:
            if round(mat_x*vx + mat_y*vy, 5) <= mat_cap and round(lab_x*vx + lab_y*vy, 5) <= lab_cap:
                valid_vertices.append((vx, vy))
        
        max_z = -1
        opt_x, opt_y = 0, 0
        for vx, vy in valid_vertices:
            z = c1*vx + c2*vy
            if z > max_z:
                max_z = z
                opt_x, opt_y = vx, vy
                
        y_fill = np.minimum(y1, y2)
        y_fill = np.maximum(y_fill, 0)
        ax.fill_between(x, 0, y_fill, where=(x >= 0) & (y_fill >= 0), color='seagreen', alpha=0.3, label='Množina přípustných řešení')
        
        if c2 != 0:
            y_profit = (max_z - c1 * x) / c2
            ax.plot(x, y_profit, label=f'Izozisková funkce Z = {max_z:.0f}', color='goldenrod', linestyle='--', lw=3)
        
        ax.plot(opt_x, opt_y, 'o', color='gold', markersize=12, markeredgecolor='black', label='OPTIMUM')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Počet vyrobených Stolů')
        ax.set_ylabel('Počet vyrobených Židlí')
        ax.set_title(f'Optimální produkce: {opt_x:.1f} Stolů, {opt_y:.1f} Židlí\nMaximální Zisk = {max_z:.0f} Kč', fontweight='bold', fontsize=14)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
