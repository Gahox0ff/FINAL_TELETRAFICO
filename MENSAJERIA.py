import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#  CARGAR ARCHIVO EXCEL
archivo = "MOVIL.xlsx"
df = pd.read_excel(archivo)

#  EMPRESAS PERMITIDAS
empresas_permitidas = [
    "AVANTEL S.A.S",
    "COLOMBIA MOVIL  S.A ESP",
    "COLOMBIA TELECOMUNICACIONES S.A. E.S.P.",
    "VIRGIN MOBILE COLOMBIA S.A.S.",
    "TELEFONICA MOVILES COLOMBIA S.A.",
    "UNE EPM TELECOMUNICACIONES S.A."
]

df["EMPRESA"] = df["EMPRESA"].astype(str).str.strip()
df = df[df["EMPRESA"].isin(empresas_permitidas)]

print("\nEmpresas incluidas en el análisis:")
for e in empresas_permitidas:
    print("-", e)

#  SELECCIÓN DE EMPRESA
print("\nEmpresas disponibles:")
for i, e in enumerate(empresas_permitidas):
    print(f"{i+1}. {e}")

op = int(input("\nSeleccione el número de la empresa: "))
empresa_sel = empresas_permitidas[op - 1]

df_f = df[df["EMPRESA"] == empresa_sel]

print(f"\n▶ Empresa seleccionada: {empresa_sel}")
print("Filas encontradas:", len(df_f))

# LIMPIEZA
df_f["CANTIDAD"] = (
    df_f["CANTIDAD"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df_f["CANTIDAD"] = df_f["CANTIDAD"].astype(float)
df_f["ANO"] = df_f["ANO"].astype(int)

# AGRUPAR POR AÑO
df_year = df_f.groupby("ANO")["CANTIDAD"].sum().reset_index()

print("\nDatos agrupados por año:")
print(df_year)

# REGRESIÓN
X = df_year["ANO"].values.reshape(-1, 1)
Y = df_year["CANTIDAD"].values

modelo = LinearRegression()
modelo.fit(X, Y)

Y_pred = modelo.predict(X)

pend = modelo.coef_[0]
intercepto = modelo.intercept_
r2 = r2_score(Y, Y_pred)

ecuacion = f"Y = {pend:.4f}·X + {intercepto:.4f}"

print("\n=========== RESULTADOS ===========")
print("Ecuación de la regresión:")
print(ecuacion)
print(f"R² = {r2:.4f}")
print("=================================\n")

# ===============================
#     PROYECCIÓN A 2 AÑOS
# ===============================
ultimo_anio = df_year["ANO"].max()
anios_proy = np.array([ultimo_anio + 1, ultimo_anio + 2]).reshape(-1, 1)
Y_proy = modelo.predict(anios_proy)

print("\nProyección de tráfico (2 años futuros):")
for a, yp in zip(anios_proy, Y_proy):
    print(f"Año {int(a)} → {float(yp):.2f}")

# ===============================
#  GRAFICAR
# ===============================

plt.figure(figsize=(9,6))

# Datos
plt.scatter(X, Y, color="red", label="Datos reales")

# Línea de regresión
plt.plot(X, Y_pred, color="green", label=f"Regresión\n{ecuacion}\nR²={r2:.4f}")

plt.scatter(anios_proy, Y_proy, color="blue", s=100, marker="X", label="Proyección 2 años")

# Línea de proyección
plt.plot(anios_proy, Y_proy, color="blue", linestyle="--")

plt.title(f"\nEmpresa: {empresa_sel}")
plt.xlabel("AÑO")
plt.ylabel("SUMA DE CANTIDAD")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
