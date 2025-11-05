# Finance Dashboard: Rendimiento, Riesgo y Optimización de Portafolio

### Descripción

**Finance Dashboard** es una aplicación interactiva desarrollada en **Python + Streamlit**, diseñada para:
- Descargar datos financieros desde **Yahoo Finance**.
- Calcular **retornos**, **volatilidad** y **Sharpe Ratio** por activo.
- Visualizar **correlaciones**, **drawdowns** y **series acumuladas**.
- Realizar **optimización de portafolio (máx. Sharpe)** con restricciones y regularización L2 usando **PyPortfolioOpt**.

---

### Stack Tecnológico

| Componente | Descripción |
|-------------|-------------|
| **Streamlit** | Interfaz web interactiva |
| **yfinance** | Descarga de precios desde Yahoo Finance |
| **Pandas / Numpy** | Cálculos y transformación de datos |
| **Matplotlib** | Visualización de curvas y heatmaps |
| **PyPortfolioOpt** | Optimización moderna de portafolios |

---

### Instalación local

```bash
git clone https://github.com/tuusuario/finance-dashboard.git
cd finance-dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
