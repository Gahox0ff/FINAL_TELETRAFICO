import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================================
# CARGAR ARCHIVO
# ==========================================================
archivo = "INTERNET.xlsx"
df = pd.read_excel(archivo)

# ==========================================================
# EMPRESAS PERMITIDAS
# ==========================================================
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

# ==========================================================
# NORMALIZACIÓN DE TRAFICO (BYTES)
# ==========================================================
df["TRAFICO"] = (
    df["TRAFICO"]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["TRAFICO"] = pd.to_numeric(df["TRAFICO"], errors="coerce")

# Normalizar años
df["ANNO"] = pd.to_numeric(df["ANNO"], errors="coerce")

# ==========================================================
# NORMALIZAR MODALIDAD (PREPAGO / POSPAGO)
# ==========================================================
df["MODALIDAD_PAGO"] = (
    df["MODALIDAD_PAGO"]
    .astype(str)
    .str.upper()
    .str.replace(" ", "")
    .str.replace("-", "")
    .str.replace("_", "")
)

df.loc[df["MODALIDAD_PAGO"].str.contains("PRE"), "MODALIDAD_PAGO"] = "PREPAGO"
df.loc[df["MODALIDAD_PAGO"].str.contains("POS"), "MODALIDAD_PAGO"] = "POSPAGO"

print("\nModalidades encontradas:", df["MODALIDAD_PAGO"].unique())

# ==========================================================
# SELECCIONAR MODALIDAD
# ==========================================================
print("\nSeleccione modalidad de tráfico:")
print("1 = Prepago")
print("2 = Pospago")
print("3 = Ambas")
op = input("Opción: ")

if op == "1":
    df = df[df["MODALIDAD_PAGO"] == "PREPAGO"]
    modalidad_seleccionada = "PREPAGO"
elif op == "2":
    df = df[df["MODALIDAD_PAGO"] == "POSPAGO"]
    modalidad_seleccionada = "POSPAGO"
else:
    modalidad_seleccionada = "AMBAS"

if df.empty:
    print("\n❌ No hay datos para esta modalidad.")
    exit()

# ==========================================================
# AGRUPACIÓN ANUAL POR EMPRESA + MODALIDAD
# ==========================================================
df_yearly = (
    df.groupby(["EMPRESA", "ANNO"], as_index=False)["TRAFICO"]
    .sum()
    .rename(columns={"TRAFICO": "TRAFICO_BYTES_ANUAL"})
)

# ==========================================================
# CONVERSIÓN BYTES → Mbps
# Fórmula solicitada: (TRAFICO * 8) / (2.592 * 10^6)
# ==========================================================
df_yearly["TRAFICO_Mbps"] = (df_yearly["TRAFICO_BYTES_ANUAL"] * 8) / (2.592e6)

# ==========================================================
# SELECCIÓN DE EMPRESA
# ==========================================================
empresas = sorted(df_yearly["EMPRESA"].unique())
print("\nSeleccione la empresa:")

for i, e in enumerate(empresas, 1):
    print(f"{i}. {e}")

op = int(input("\nEmpresa: "))
empresa_sel = empresas[op - 1]

df_emp = df_yearly[df_yearly["EMPRESA"] == empresa_sel]

if df_emp.empty:
    print("❌ No hay datos para esta empresa.")
    exit()

# ==========================================================
# GRAFICAR + REGRESIÓN
# ==========================================================
X = df_emp["ANNO"].values.reshape(-1, 1)
Y = df_emp["TRAFICO_Mbps"].values

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color="red", label="Datos reales", s=80)

# Regresión
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

m = model.coef_[0]
b = model.intercept_
r2 = r2_score(Y, Y_pred)

ecuacion = f"Y = {m:.4f}·X + {b:.2f}"

# Línea regresión
plt.plot(X, Y_pred, color="green", linewidth=2,
         label=f"Regresión\n{ecuacion}\nR² = {r2:.4f}")

# Proyección 2 años
ultimo_anio = int(df_emp["ANNO"].max())
anios_proy = np.array([ultimo_anio + 1, ultimo_anio + 2]).reshape(-1, 1)
Y_proy = model.predict(anios_proy)

print("\n===== RESULTADOS =====")
print("Ecuación de la recta:", ecuacion)
print(f"R² = {r2:.6f}")
print("\nProyecciones:")
for an, val in zip(anios_proy.flatten(), Y_proy):
    print(f"Año {an}: {val:.2f} Mbps")

plt.scatter(anios_proy, Y_proy, color="blue", s=120, marker="X",
            label="Proyección 2 años")

plt.plot(anios_proy, Y_proy, color="blue", linestyle="--")

# ==========================================================
# FORMATO FINAL
# ==========================================================
plt.title(f"Tráfico anual (Mbps) - {empresa_sel} - Modalidad: {modalidad_seleccionada}")
plt.xlabel("Año")
plt.ylabel("Tráfico (Mbps)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
