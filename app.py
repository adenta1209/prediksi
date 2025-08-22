import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =========================
# Cek file CSV yang ada
# =========================
if os.path.exists("gabungan.csv"):
    DATA_FILE = "gabungan.csv"
elif os.path.exists("Data Gabungan.csv"):
    DATA_FILE = "Data Gabungan.csv"
else:
    st.error("⚠️ File gabungan.csv atau Data Gabungan.csv tidak ditemukan di folder!")
    st.stop()

# =========================
# Fungsi load data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
    df = df.dropna(subset=["TANGGAL"])
    df["LABA"] = df["PENJUALAN"] - (df["PRODUKSI"] + df["BEBAN"])
    return df

# =========================
# Fungsi train model
# =========================
@st.cache_resource
def train_model(df):
    X = df[["PENJUALAN", "PRODUKSI", "BEBAN"]]
    y = df["LABA"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    joblib.dump(model, "model.pkl")
    return model, rmse, mae, mape

# =========================
# Login sederhana
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "123":
            st.session_state.logged_in = True
            st.success("Login berhasil! 🎉")

            # --- Tambahan: redirect otomatis ke Dashboard ---
            st.session_state.menu = "Dashboard"
            st.rerun()
            # ------------------------------------------------

        else:
            st.error("Username/Password salah!")
    st.stop()

# =========================
# Navigasi dengan tombol
# =========================
st.sidebar.title("Menu Utama")
menu = st.session_state.get("menu", "Dashboard")

if st.sidebar.button("🏠 Dashboard"):
    st.session_state.menu = "Dashboard"
if st.sidebar.button("📈 Prediksi"):
    st.session_state.menu = "Prediksi"
if st.sidebar.button("📜 Riwayat Prediksi"):
    st.session_state.menu = "Riwayat Prediksi"

menu = st.session_state.get("menu", "Dashboard")

# =========================
# Load data & model
# =========================
df = load_data()
model, rmse, mae, mape = train_model(df)

# =========================
# Halaman Dashboard
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard")
    st.subheader("Keunggulan Metode Regresi Linier")
    st.write("""
    - Mudah dipahami dan diimplementasikan  
    - Cocok untuk data dengan hubungan linier  
    - Memberikan interpretasi koefisien yang jelas  
    """)

    st.subheader("Evaluasi Model (Hasil dari Jupyter Notebook)")
    st.markdown("""
    - **Root Mean Squared Error (RMSE):** 2.39374 × 10⁻⁸  
    - **Mean Absolute Error (MAE):** 2.37065 × 10⁻⁸  
    - **Mean Absolute Percentage Error (MAPE):** 6.49328 × 10⁻¹² %  
    """)

# =========================
# Halaman Prediksi
# =========================
elif menu == "Prediksi":
    st.title("📈 Prediksi Laba Bersih")

    # Input manual jumlah hari prediksi
    pilihan = st.number_input("Masukkan jumlah hari prediksi:", min_value=1, max_value=365, value=7, step=1)

    # Tentukan tanggal terakhir dari data
    last_date = df["TANGGAL"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pilihan, freq="D")

    # Ambil baris terakhir sebagai dasar prediksi
    last_row = df[["PENJUALAN", "PRODUKSI", "BEBAN"]].iloc[-1].values.reshape(1, -1)

    # Iteratif prediksi ke depan dengan fluktuasi
    future_preds = []
    current_row = last_row.copy()
    for _ in range(len(future_dates)):
        pred = model.predict(current_row)[0]

        # Tambahkan variasi acak (±5%)
        noise = np.random.uniform(0.95, 1.05)  
        pred = pred * noise

        future_preds.append(pred)

        # Update data untuk prediksi berikutnya (acak naik/turun kecil)
        current_row = current_row.copy()
        current_row[0][0] = current_row[0][0] * np.random.uniform(0.98, 1.02)  # PENJUALAN fluktuatif ±2%
        current_row[0][1] = current_row[0][1] * np.random.uniform(0.98, 1.02)  # PRODUKSI fluktuatif ±2%
        current_row[0][2] = current_row[0][2] * np.random.uniform(0.99, 1.01)  # BEBAN fluktuatif ±1%

   # Buat dataframe hasil prediksi
    hasil_prediksi = pd.DataFrame({"Tanggal": future_dates, "Prediksi Laba": future_preds})

    # Simpan angka asli untuk grafik
    laba_asli = hasil_prediksi["Prediksi Laba"].copy()

    # Format Rupiah untuk tampilan tabel
    hasil_prediksi["Prediksi Laba"] = hasil_prediksi["Prediksi Laba"].apply(
        lambda x: "Rp {:,.0f}".format(x).replace(",", ".")
    )

    # Tampilkan tabel (Rp)
    st.dataframe(hasil_prediksi)

    # --- Buat grafik batang dengan matplotlib ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(future_dates, laba_asli)

    # Format sumbu Y jadi Rupiah
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'Rp {x:,.0f}'.replace(",", ".")))

    ax.set_title("Prediksi Laba Bersih")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Laba Bersih")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # --- Tombol download hasil prediksi ---
    import io
    # Buat versi CSV
    csv = hasil_prediksi.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediksi (CSV)",
        data=csv,
        file_name="prediksi_laba.csv",
        mime="text/csv"
    )

    # Buat versi Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        hasil_prediksi.to_excel(writer, index=False, sheet_name="Prediksi Laba")
    excel_data = output.getvalue()

    st.download_button(
        label="📥 Download Prediksi (Excel)",
        data=excel_data,
        file_name="prediksi_laba.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

        # Simpan riwayat prediksi
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []
    if st.button("Simpan ke Riwayat"):
        st.session_state.riwayat.append(hasil_prediksi)
        st.success("✅ Hasil prediksi disimpan ke Riwayat")


# =========================
# Halaman Riwayat Prediksi
# =========================
elif menu == "Riwayat Prediksi":
    st.title("📜 Riwayat Prediksi")
    if "riwayat" in st.session_state and len(st.session_state.riwayat) > 0:
        for i, riw in enumerate(st.session_state.riwayat):
            st.write(f"### Riwayat {i+1}")
            st.dataframe(riw)
            st.line_chart(riw.set_index("Tanggal"))
    else:
        st.info("Belum ada riwayat prediksi.")

# =========================
# Fitur Logout (Tambahan)
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Tampilkan tombol logout jika user sudah login
if st.session_state.logged_in:
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.success("✅ Anda berhasil logout.")
        st.rerun()