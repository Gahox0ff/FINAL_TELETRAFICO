import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ============================================
# El CODIGO PUEDE TENER ERRORES CON LOS CARACTERES ESPECIALES DEL CSV A ULTILIZAR, SE NECESARIO REVISAR CAMBIAR LOS NOMBRES DE LAS EMPRESAS QUE LO TENGAN
#  CARGAR CSV
# ============================================
archivo = "MOVIL.csv"
df = pd.read_csv(archivo, sep=";", engine="python")
# ============================================
#  EMPRESAS PERMITIDAS
# ============================================
empresas_permitidas = [
    "AVANTEL S.A.S",
    "COLOMBIA MOVIL S.A. ESP",
    "COLOMBIA TELECOMUNICACIONES S.A. E.S.P.",
    "VIRGIN MOBILE COLOMBIA S.A.S.",
    "TELEFÓNICA MOVILES COLOMBIA S.A.",
    "UNE EPM TELECOMUNICACIONES S.A."
]

df["EMPRESA"] = df["EMPRESA"].str.strip()
df = df[df["EMPRESA"].isin(empresas_permitidas)]

print("\nEmpresas incluidas en el análisis:")
for e in empresas_permitidas:
    print("-", e)

# ============================================
#  SELECCIÓN DE EMPRESA
# ============================================
print("\nEmpresas disponibles:")
for i, e in enumerate(empresas_permitidas):
    print(f"{i+1}. {e}")

op = int(input("\nSeleccione el número de la empresa: "))
empresa_sel = empresas_permitidas[op - 1]

df_f = df[df["EMPRESA"] == empresa_sel]

print(f"\n▶ Empresa seleccionada: {empresa_sel}")
print("Filas encontradas:", len(df_f))

# ============================================
#  LIMPIEZA DE CANTIDAD (quitar comas finales)
# ============================================
df_f["CANTIDAD,"] = df_f["CANTIDAD,"].astype(str).str.replace(",", "", regex=False)

# Convertir a número
df_f["CANTIDAD,"] = df_f["CANTIDAD,"].astype(float)
df_f["ANO"] = df_f["ANO"].astype(int)

# ============================================
#  AGRUPAR POR AÑO Y SUMAR CANTIDAD
# ============================================
df_year = df_f.groupby("ANO")["CANTIDAD,"].sum().reset_index()

print("\nDatos agrupados por año:")
print(df_year)

# Preparar datos para regresión
X = df_year["ANO"].values.reshape(-1, 1)
Y = df_year["CANTIDAD,"].values

# ============================================
#  REGRESIÓN LINEAL
# ============================================
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

# ============================================
#  GRAFICAR
# ============================================
plt.figure(figsize=(8,5))
plt.scatter(X, Y, label="Suma anual real")
plt.plot(X, Y_pred, label=f"{ecuacion}\nR²={r2:.4f}")

plt.title(f"Regresión lineal - Empresa: {empresa_sel}")
plt.xlabel("AÑO")
plt.ylabel("SUMA DE CANTIDAD")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
