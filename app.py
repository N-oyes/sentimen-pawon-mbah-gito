import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preprocessing import preprocess_text_for_prediction

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Klasifikasi Sentimen - Pawon Mbah Gito",
    page_icon="🍛",
    layout="wide",
)

DATA_DIR = "data"

# =============================================================
# CUSTOM THEME / CSS
# =============================================================
st.markdown(
    """
    <style>
        :root {
            --primary: #c0392b;
            --primary-light: #e74c3c;
            --secondary: #e67e22;
            --accent: #f39c12;
            --bg-soft: #fdf6f2;
            --border: #f0e3dc;
            --text: #2c3e50;
            --text-muted: #7f8c8d;
            --positive: #27ae60;
            --negative: #c0392b;
        }

        .stApp {
            background-color: var(--bg-soft);
        }

        /* Hide default streamlit chrome noise */
        #MainMenu, footer, header {visibility: hidden;}
        .stApp > header {background: transparent;}

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #fdf6f2 100%);
            border-right: 1px solid var(--border);
        }

        /* Titles */
        h1, h2, h3, h4 {
            color: var(--text) !important;
            font-weight: 700 !important;
            letter-spacing: -0.3px;
        }
        h1 { font-size: 2.1rem !important; }
        h2 { font-size: 1.6rem !important; }

        /* Step card */
        .step-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            min-height: 170px;
            box-shadow: 0 2px 8px rgba(192, 57, 43, 0.04);
            transition: transform .2s ease, box-shadow .2s ease;
        }
        .step-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(192, 57, 43, 0.10);
            border-color: var(--primary-light);
        }
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px; height: 34px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #fff;
            font-weight: 800;
            font-size: 15px;
            margin-bottom: 10px;
        }
        .step-title { font-weight: 700; color: var(--text); font-size: 15px; }
        .step-desc  { font-size: 13px; color: var(--text-muted); margin-top: 6px; line-height: 1.5; }

        /* Metric */
        .metric-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(192, 57, 43, 0.04);
        }
        .metric-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 28px; font-weight: 800; color: var(--primary); margin-top: 6px; }

        /* Section divider */
        .section-divider {
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), transparent);
            border-radius: 3px;
            margin: 24px 0;
        }

        /* Pipeline tab content */
        .pipe-card {
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 10px rgba(192, 57, 43, 0.05);
        }

        /* Result badge */
        .result-badge {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 24px;
            border-radius: 16px;
            font-size: 22px;
            font-weight: 800;
        }
        .result-positive {
            background: linear-gradient(135deg, #e8f8ee, #d4f0dd);
            color: #1e8449;
            border: 1px solid #abebc6;
        }
        .result-negative {
            background: linear-gradient(135deg, #fdecea, #f7d4ce);
            color: #922b21;
            border: 1px solid #f0b5ad;
        }
        .result-emoji { font-size: 44px; }

        /* Sidebar nav */
        .nav-radio label { font-weight: 500; }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            padding: 10px 22px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-light), var(--accent));
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            background: #fff;
            border: 1px solid var(--border);
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
            color: #fff !important;
        }

        /* Code block */
        .stCodeBlock, pre {
            border-radius: 10px !important;
        }

        /* Caption muted */
        .stCaption, .st-emphasis-caption {
            color: var(--text-muted) !important;
        }

        /* Empty state */
        .empty-state {
            background: #fff;
            border: 1px dashed #d8c5bc;
            border-radius: 12px;
            padding: 28px;
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
        }

        /* Sentiment preview chips */
        .chip {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
        }
        .chip-pos { background: #e8f8ee; color: #1e8449; }
        .chip-neg { background: #fdecea; color: #922b21; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# HELPERS
# =============================================================
def load_csv_safe(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("svm_linear80_smote.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf


def metric_card(label, value, delta=None):
    delta_html = f"<div style='font-size:11px;color:var(--text-muted);margin-top:4px'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(msg):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:14px 0 8px;">
            <div style="font-size:46px;">🍛</div>
            <div style="font-size:20px; font-weight:800; color:var(--text); margin-top:6px;">
                Pawon Mbah Gito
            </div>
            <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                Sistem Klasifikasi Sentimen
            </div>
        </div>
        <hr style="border:none; border-top:1px solid var(--border); margin:16px 0;" />
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "⚙️ Sistem", "📊 Visualisasi", "🔍 Analisis Sentimen"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <hr style="border:none; border-top:1px solid var(--border); margin:16px 0;" />
        <div style="font-size:11px; color:var(--text-muted); line-height:1.7;">
            <div><b>Model</b>: Linear SVM + SMOTE</div>
            <div><b>Pembobotan</b>: TF-IDF</div>
            <div><b>Split</b>: 80:20</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================
# PAGE 1 — BERANDA
# =============================================================
if page == "🏠 Beranda":
    st.markdown("# 🍛 Klasifikasi Sentimen Ulasan Pawon Mbah Gito")
    st.caption("Analisis sentimen ulasan pelanggan menggunakan pendekatan Machine Learning")

    st.markdown(
        """
        Sistem ini menganalisis **sentimen ulasan pelanggan** terhadap restoran *Pawon Mbah Gito*.
        Dari 2.755 ulasan hasil scraping, label sentimen diberikan berdasarkan rating
        (4–5 → **Positif**, 1–3 → **Negatif**). Lima algoritma diuji pada 3 rasio split data
        dengan dan tanpa **SMOTE**. Model terpilih: **SVM Linear + SMOTE (split 80:20)**
        dengan pembobotan **TF-IDF**.
        """
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Pipeline steps
    st.markdown("### 🔄 Alur Sistem")
    steps = [
        ("1", "Scraping Data", "Mengumpulkan 2.755 ulasan Google Reviews beserta rating."),
        ("2", "Preprocessing", "Case folding, cleansing, tokenizing, normalisasi slang, stopword, stemming."),
        ("3", "Pelabelan", "Rating 4–5 → Positif, rating 1–3 → Negatif."),
        ("4", "Pembobotan TF-IDF", "Vektorisasi teks: 3000 fitur, ngram (1,2), sublinear_tf."),
        ("5", "Splitting Data", "Dibagi 3 rasio: 70:30, 80:20, dan 90:10."),
        ("6", "SMOTE", "Menyeimbangkan kelas minoritas pada data latih."),
        ("7", "Klasifikasi", "Naive Bayes, SVM Linear, KNN, Decision Tree, Random Forest."),
        ("8", "Evaluasi", "Accuracy, precision, recall, F1-score → SVM+SMOTE dipilih."),
    ]
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Dataset summary
    st.markdown("### 📦 Ringkasan Dataset")
    summary_df = load_csv_safe("dataset_summary.csv")
    c1, c2, c3, c4 = st.columns(4)
    if summary_df is not None:
        row = summary_df.iloc[0]
        with c1: metric_card("Total Ulasan", f"{int(row.get('total', 0)):,}")
        with c2: metric_card("Positif", f"{int(row.get('positif', 0)):,}")
        with c3: metric_card("Negatif", f"{int(row.get('negatif', 0)):,}")
        with c4: metric_card("Akurasi Model", f"{row.get('akurasi', 0)}%")
    else:
        with c1: metric_card("Total Ulasan", "2.732")
        with c2: metric_card("Positif", "2.446")
        with c3: metric_card("Negatif", "286")
        with c4: metric_card("Akurasi Model", "90.9%")

    st.markdown(
        """
        ---
        ### 🧭 Jelajahi Sistem
        - **⚙️ Sistem** — lihat tahapan pemrosesan data secara detail
        - **📊 Visualisasi** — perbandingan performa antar model
        - **🔍 Analisis Sentimen** — uji klasifikasi pada ulasan apapun
        """
    )


# =============================================================
# PAGE 2 — SISTEM
# =============================================================
elif page == "⚙️ Sistem":
    st.markdown("## ⚙️ Sistem: Alur Pemrosesan Data")
    st.caption("Tahapan dari data mentah hingga evaluasi model")

    tabs = st.tabs([
        "1️⃣ Scraping",
        "2️⃣ Preprocessing",
        "3️⃣ Pelabelan",
        "4️⃣ TF-IDF",
        "5️⃣ Splitting",
        "6️⃣ SMOTE",
        "7️⃣ Model SVM",
        "8️⃣ Evaluasi",
    ])

    # 1. Scraping
    with tabs[0]:
        with st.container():
            st.markdown("#### 📥 Scraping Data")
            st.write("Dataset `datasetpawon.csv` berisi **2.755 ulasan** pelanggan beserta rating (1–5).")
            df_scrap = load_csv_safe("hasil_scraping.csv")
            if df_scrap is not None:
                st.dataframe(df_scrap.head(20), use_container_width=True, height=380)
            else:
                empty_state("Data scraping belum tersedia.")

    # 2. Preprocessing
    with tabs[1]:
        st.markdown("#### 🧹 Preprocessing Teks")
        st.markdown(
            """
            <div class="pipe-card">
            <b>6 tahap berurutan</b> diterapkan pada setiap ulasan:
            <ol style="margin-top:10px; line-height:1.9;">
                <li><b>Case Folding</b> — semua huruf jadi huruf kecil</li>
                <li><b>Cleansing</b> — hapus URL, angka, dan karakter non-huruf</li>
                <li><b>Tokenizing</b> — pecah kalimat menjadi token (<code>RegexpTokenizer</code>)</li>
                <li><b>Normalisasi Slang</b> — ubah kata gaul ke bentuk formal (kamus <code>slang.csv</code>)</li>
                <li><b>Stopword Removal</b> — buang kata umum, <i>kecuali</i> kata sentimen (tidak, enak, mahal, kecewa, puas)</li>
                <li><b>Stemming</b> — kata ke bentuk dasar (Sastrawi)</li>
            </ol>
            <div style="font-size:13px; color:var(--text-muted); margin-top:8px;">
            Dari 2.755 ulasan mentah → 2.732 ulasan tersisa setelah baris kosong dibuang.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🧪 Coba Langsung")
        sample_text = st.text_input("Masukkan contoh teks:", "Makanannya gk EnAk, tp pelayanan lama bgt")
        if sample_text:
            result = preprocess_text_for_prediction(sample_text)
            st.markdown("**Hasil preprocessing:**")
            st.code(result, language="text")

        df_prep = load_csv_safe("hasil_preprocessing.csv")
        if df_prep is not None:
            st.markdown("**Perbandingan sebelum & sesudah:**")
            st.dataframe(df_prep.head(20), use_container_width=True, height=380)

    # 3. Pelabelan
    with tabs[2]:
        st.markdown("#### 🏷️ Pelabelan Sentimen")
        st.write("Label ditentukan otomatis dari rating:")
        st.code("rating >= 4  ->  Positif (1)\nrating <= 3  ->  Negatif (0)", language="text")

        df_label = load_csv_safe("distribusi_label.csv")
        if df_label is not None:
            fig = px.pie(
                df_label, names="label", values="jumlah",
                title="Distribusi Label Sentimen (2.732 data)",
                color_discrete_sequence=["#27ae60", "#c0392b"],
                hole=0.4,
            )
            fig.update_layout(showlegend=True, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <div class="pipe-card" style="background:#fff8f6;">
                ⚠️ Distribusi kelas <b>tidak seimbang</b>: Positif 89,5% vs Negatif 10,5% —
                alasan utama SMOTE diterapkan pada data latih.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            empty_state("Data distribusi label belum tersedia.")

    # 4. TF-IDF
    with tabs[3]:
        st.markdown("#### 📊 Pembobotan Fitur TF-IDF")
        st.write(
            "TF-IDF mengubah teks hasil preprocessing menjadi representasi numerik "
            "berdasarkan kepentingan kata dalam dokumen relatif terhadap seluruh korpus."
        )
        st.markdown(
            """
            <div class="pipe-card">
            <b>Konfigurasi <code>TfidfVectorizer</code>:</b>
            <ul style="margin-top:8px; line-height:1.8;">
                <li><code>max_features = 3000</code></li>
                <li><code>min_df = 2</code>, <code>max_df = 0.9</code></li>
                <li><code>ngram_range = (1, 2)</code> — unigram & bigram</li>
                <li><code>sublinear_tf = True</code></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            _, tfidf = load_model_artifacts()
            n_features = len(tfidf.vocabulary_)
            m1, m2 = st.columns(2)
            with m1: metric_card("Jumlah Fitur", f"{n_features:,}")
            with m2: metric_card("Vektor", "Sparse Matrix")
        except Exception as e:
            st.warning(f"Tidak bisa memuat `tfidf.pkl`: {e}")

        st.markdown("#### 🔝 10 Kata dengan Bobot TF-IDF Tertinggi")
        top_words = pd.DataFrame({
            "Kata": ["makan", "enak", "nyaman", "bagus", "makan enak", "sekali", "suasana", "tempat", "tidak", "menu"],
            "Bobot TF-IDF": [0.0546, 0.0529, 0.0369, 0.0282, 0.0265, 0.0265, 0.0264, 0.0255, 0.0245, 0.0239],
        })
        fig_tfidf = px.bar(
            top_words, x="Bobot TF-IDF", y="Kata", orientation="h",
            color="Bobot TF-IDF",
            color_continuous_scale=["#f8c9bd", "#c0392b"],
        )
        fig_tfidf.update_yaxes(autorange="reversed")
        fig_tfidf.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_tfidf, use_container_width=True)

    # 5. Splitting
    with tabs[4]:
        st.markdown("#### ✂️ Splitting Data")
        st.write(
            "Data dibagi menjadi latih & uji dengan **3 rasio berbeda** "
            "(`train_test_split`, `stratify=y`, `random_state=42`)."
        )
        df_split = load_csv_safe("hasil_split.csv")
        if df_split is not None:
            melted = df_split.melt(id_vars="rasio", value_vars=["data_latih", "data_uji"],
                                   var_name="set", value_name="jumlah")
            fig_split = px.bar(
                melted, x="rasio", y="jumlah", color="set", barmode="group",
                color_discrete_map={"data_latih": "#c0392b", "data_uji": "#e67e22"},
                text="jumlah",
            )
            fig_split.update_layout(height=420)
            st.plotly_chart(fig_split, use_container_width=True)
            st.dataframe(df_split, use_container_width=True, hide_index=True)
            st.success("Model final: rasio **80:20** → 2.185 data latih, 547 data uji.")
        else:
            empty_state("Data hasil splitting belum tersedia.")

    # 6. SMOTE
    with tabs[5]:
        st.markdown("#### ⚖️ SMOTE")
        st.write(
            "SMOTE **hanya diterapkan pada data latih** untuk menyeimbangkan kelas Positif & Negatif "
            "dengan membuat data sintetis pada kelas minoritas (Negatif)."
        )
        df_smote = load_csv_safe("perbandingan_smote.csv")
        if df_smote is not None:
            fig = px.bar(
                df_smote, x="label", y="jumlah", color="kondisi", barmode="group",
                color_discrete_map={"Sebelum SMOTE": "#95a5a6", "Sesudah SMOTE": "#c0392b"},
                text="jumlah",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Data latih 80:20: 1.955 Positif : 230 Negatif → setelah SMOTE menjadi 1.955 : 1.955 (seimbang).")
        else:
            empty_state("Data perbandingan SMOTE belum tersedia.")

    # 7. Model SVM
    with tabs[6]:
        st.markdown("#### 🤖 Model Klasifikasi SVM")
        st.write(
            "Model final: **Linear SVM** (`sklearn.svm.LinearSVC`) — cocok untuk data teks "
            "berdimensi tinggi hasil TF-IDF. Notebook juga membandingkan Naive Bayes, KNN, "
            "Decision Tree, dan Random Forest (lihat menu **Visualisasi**)."
        )
        try:
            model, _ = load_model_artifacts()
            info = pd.DataFrame({
                "Parameter": ["Algoritma", "C", "Loss", "Random State"],
                "Nilai": [
                    type(model).__name__,
                    str(getattr(model, "C", "—")),
                    str(getattr(model, "loss", "—")),
                    str(getattr(model, "random_state", "—")),
                ],
            })
            st.table(info)
        except Exception as e:
            st.warning(f"Tidak bisa memuat detail model: {e}")

    # 8. Evaluasi
    with tabs[7]:
        st.markdown("#### 📈 Evaluasi Model")
        df_eval = load_csv_safe("evaluasi_model.csv")
        if df_eval is not None:
            st.dataframe(df_eval, use_container_width=True, hide_index=True)
        else:
            empty_state("Data evaluasi model belum tersedia.")


# =============================================================
# PAGE 3 — VISUALISASI
# =============================================================
elif page == "📊 Visualisasi":
    st.markdown("## 📊 Visualisasi Performa Model")
    st.caption("Perbandingan antar model & hasil evaluasi model terpilih")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Perbandingan model
    st.markdown("### 🏆 Perbandingan Performa Model")
    st.caption("5 algoritma, dengan & tanpa SMOTE — split 80:20 (data uji = 547 ulasan)")

    df_compare = load_csv_safe("perbandingan_model.csv")
    if df_compare is not None:
        fig = px.bar(
            df_compare, x="model", y=["accuracy", "precision", "recall", "f1_score"],
            barmode="group",
            color_discrete_sequence=["#c0392b", "#e67e22", "#f39c12", "#27ae60"],
        )
        fig.update_layout(xaxis_tickangle=-20, legend_title_text="Metrik", height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)
        st.info(
            "**SVM Linear (SMOTE)** dipilih: meski akurasi sedikit di bawah versi tanpa SMOTE "
            "(90,9% vs 93,4%), F1-score dan recall pada kelas minoritas (Negatif) jauh lebih baik — "
            "krusial karena distribusi data asli timpang (89,5% Positif)."
        )
    else:
        empty_state("Data perbandingan model belum tersedia.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Tabel evaluasi
    st.markdown("### 📋 Tabel Hasil Evaluasi")
    df_eval2 = load_csv_safe("evaluasi_model.csv")
    if df_eval2 is not None:
        st.dataframe(df_eval2, use_container_width=True, hide_index=True)
    else:
        empty_state("Data evaluasi model belum tersedia.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confusion matrix
    st.markdown("### 🔲 Confusion Matrix")
    df_cm = load_csv_safe("confusion_matrix.csv")
    if df_cm is not None and {"actual", "predicted", "jumlah"}.issubset(df_cm.columns):
        pivot = df_cm.pivot(index="actual", columns="predicted", values="jumlah")
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0, "#fdf0eb"], [1, "#c0392b"]],
                text=pivot.values, texttemplate="%{text}",
                textfont={"size": 20, "color": "#fff"},
            )
        )
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted",
                             yaxis_title="Actual", height=440)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        empty_state("Data confusion matrix belum tersedia.")


# =============================================================
# PAGE 4 — ANALISIS SENTIMEN
# =============================================================
elif page == "🔍 Analisis Sentimen":
    st.markdown("## 🔍 Analisis Sentimen Ulasan")
    st.caption("Klasifikasikan sentimen dari ulasan apapun secara langsung")

    try:
        model, tfidf = load_model_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### ✍️ Input Ulasan")
        text = st.text_area(
            "Masukkan teks ulasan:",
            height=220,
            placeholder="Contoh: Makanannya enak banget, pelayanan ramah, harga terjangkau...",
        )
        run = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📤 Hasil Analisis")

        if run and text.strip():
            clean_text = preprocess_text_for_prediction(text)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]

            st.markdown("#### Hasil Preprocessing")
            st.code(clean_text, language="text")

            st.markdown("---")
            st.markdown("#### Hasil Klasifikasi")

            if prediction == 1:
                st.markdown(
                    """
                    <div class="result-badge result-positive">
                        <div class="result-emoji">😊</div>
                        <div>POSITIF</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-badge result-negative">
                        <div class="result-emoji">😞</div>
                        <div>NEGATIF</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if hasattr(model, "decision_function"):
                score = model.decision_function(vector)[0]
                confidence = abs(score)
                st.progress(min(max(confidence / 5, 0.05), 1.0))
                st.caption(f"Decision score: {score:.4f} — semakin jauh dari 0, semakin yakin model.")
        else:
            empty_state("Masukkan ulasan dan klik **Analisis Sentimen** untuk melihat hasilnya.")