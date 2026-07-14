import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import datetime
from urllib.parse import urlparse

# --- Sabitler ---
ALLOWED_DOMAIN = "koctas.com.tr"
LOGO_URL = "https://koctas-img.mncdn.com/static/koctas-logo.svg"
FAVICON_URL = "https://koctas-img.mncdn.com/static/favicon-32x32.png"

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="Koçtaş Content Update & Quality Check",
    page_icon=FAVICON_URL,  # Koçtaş Favicon
    layout="wide"
)
st.markdown("""
    <style>
        h1, h2, h3 { color: #222220 !important; }
        [data-testid="stSidebar"] { border-right: 2px solid #EC6E00; }
    </style>
""", unsafe_allow_html=True)

# --- Yardımcı Fonksiyonlar ---

def clean_text(html_content):
    """HTML içeriğini temizler, gereksiz footer/header alanlarını atar."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Gereksiz etiketleri kaldır
    for script in soup(["script", "style", "iframe", "nav", "footer", "header", "noscript", "svg"]):
        script.decompose()

    # Metinleri al
    text = soup.get_text(separator='\n')

    # Öneri/ilgili içerik bloklarını ve sonrasını kesip at
    split_phrases = [
        "İlginizi Çekebilecek Yazılar", "İlgili Yazılar", "Benzer Yazılar",
        "Benzer Ürünler", "İlginizi Çekebilecek Ürünler", "Önerilen Ürünler"
    ]
    for phrase in split_phrases:
        if phrase in text:
            text = text.split(phrase)[0]

    # Boşlukları temizle
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

@st.cache_data(ttl=3600, show_spinner=False)
def get_available_models(api_key):
    """API'deki kullanılabilir modelleri listeler (1 saat önbelleklenir)."""
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception:
        return []

def extract_json(raw_text):
    """Model yanıtındaki JSON bloğunu güvenli şekilde ayıklar."""
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Yanıtta JSON bulunamadı.")
    return json.loads(cleaned[start:end + 1])

def remove_link_suggestions(analysis):
    """Link önerisi içeren düzeltmeleri sonuçlardan ayıklar (güvenlik ağı)."""
    banned_keywords = ["link eklenmeli", "eksik link", "link verilmeli", "köprü bağlantısı"]
    corrections = analysis.get('corrections', [])
    filtered = []
    for item in corrections:
        combined = " ".join([
            str(item.get('original', '')),
            str(item.get('correction', '')),
            str(item.get('reason', ''))
        ]).lower()
        if not any(kw in combined for kw in banned_keywords):
            filtered.append(item)
    analysis['corrections'] = filtered
    return analysis

def analyze_with_gemini(text, source, api_key):
    """Gemini API ile metni analiz eder."""
    genai.configure(api_key=api_key)

    today_str = datetime.date.today().strftime("%d.%m.%Y")

    prompt = f"""
    Sen Koçtaş (Türkiye'nin lider yapı market ve ev geliştirme perakendecisi) için çalışan kıdemli bir İçerik Kalite Uzmanı ve Editörsün.
    Bugünün tarihi: {today_str}.

    Aşağıdaki metin şu kaynaktan alındı: {source}

    GÖREVİN:
    1. Bu içeriği; güncellik, doğruluk, yazım kuralları ve marka dili açısından denetle.
    2. Sayfanın en üstünde yer alan **Başlık (H1)** ve **'Son Güncelleme Tarihi'** gibi meta bilgileri analize DAHİL ETME. Sadece gövde metninin kalitesine ve güncelliğine odaklan.
    3. Sayfanın alt kısımlarındaki "Benzer ürünler", "İlginizi çekebilecek ürünler" gibi öneri alanlarını görmezden gel.
    4. Ürün/kategori sayfalarındaki fiyat, stok ve kampanya bilgilerini denetleme; bunlar dinamik verilerdir. Açıklama ve tanıtım metinlerine odaklan.

    YAPMA (ÖNEMLİ):
    - Link, köprü veya bağlantı eksikliğiyle ilgili HİÇBİR düzeltme veya öneri yapma.
    - "Link Eklenmeli" türünde notlar ASLA üretme. Metindeki "tıklayınız", "buradan ulaşabilirsiniz" gibi ifadelerin bir yere bağlı olup olmadığını değerlendirme; bu senin görevin dışında.
    - Sadece metnin kendisindeki hatalara odaklan: eski/yanlış bilgi, yazım ve dil bilgisi hataları, marka diline uymayan ifadeler.

    Yanıtını SADECE aşağıdaki JSON formatında ver:
    {{
        "summary": "İçeriğin kısa özeti.",
        "corrections": [
            {{
                "original": "Hatalı kısım",
                "correction": "Düzeltilmiş hali",
                "reason": "Hata sebebi (Eski bilgi, Yazım hatası vb.)"
            }}
        ],
        "missing_topics": ["Eksik konu 1", "Eksik konu 2"],
        "score": 85
    }}

    İÇERİK METNİ (İlk 25.000 karakter):
    {text[:25000]} 
    """

    # Model Seçimi
    available_models = get_available_models(api_key)
    preferred_order = []

    if available_models:
        preferred_order.extend([m for m in available_models if 'flash' in m])
        preferred_order.extend([m for m in available_models if 'pro' in m])
        preferred_order.extend([m for m in available_models if m not in preferred_order])
    else:
        preferred_order = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash',
            'models/gemini-2.5-pro',
            'models/gemini-1.5-flash'
        ]

    last_error = None

    for model_name in preferred_order:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            analysis = extract_json(response.text)
            return remove_link_suggestions(analysis)
        except Exception as e:
            last_error = e
            continue

    raise Exception(f"Analiz hatası. Son hata: {str(last_error)}")

def fetch_url_content(url):
    """URL'den içeriği çeker (Akamai bot korumasını aşmak için tarayıcı TLS taklidi)."""
    try:
        response = requests.get(
            url,
            impersonate="chrome120",  # Akamai JA3 kontrolünü geçmek için ŞART
            timeout=20
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        msg = str(e)
        if "403" in msg:
            raise Exception("Erişim Engellendi (403). Koçtaş bot koruması isteği engelledi.")
        raise Exception(f"Bağlantı hatası: {msg}")

def is_allowed_domain(url):
    """URL'nin koctas.com.tr domainine ait olup olmadığını güvenli şekilde kontrol eder."""
    netloc = urlparse(url).netloc.lower()
    return netloc == ALLOWED_DOMAIN or netloc.endswith("." + ALLOWED_DOMAIN)

# --- ARAYÜZ ---

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(LOGO_URL, width=120)

with col_title:
    st.title("Koçtaş Content Update & Quality Check")
    st.markdown("**İçerik güncelliği, doğruluk ve kalite kontrol aracı.**")

with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("Gemini API Key", type="password")
    st.info("API anahtarı kaydedilmez.")

st.subheader("Analiz Edilecek Sayfalar")
urls_input = st.text_area(
    "URL Listesi (Her satıra bir tane yapıştırın)",
    height=150,
    placeholder="https://www.koctas.com.tr/mobilya/c/109"
)

analyze_button = st.button("Kontrolü Başlat", type="primary")

if analyze_button:
    if not api_key:
        st.error("Lütfen sol menüden API Anahtarını girin.")
    else:
        url_list = [u.strip() for u in urls_input.split('\n') if u.strip()]

        if not url_list:
            st.warning("Lütfen geçerli bir URL girin.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, url in enumerate(url_list):
                # Şemasız girilen URL'leri tamamla (yoksa domain kontrolü boş kalır)
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                # --- DOMAIN KONTROLÜ (Çakallık Engelleyici) ---
                if not is_allowed_domain(url):
                    results.append({
                        "title": url,
                        "error": "Çakallık engelleyici aktif! Sadece koctas.com.tr özelinde analiz yapabilirsin",
                        "status": "error"
                    })
                    progress_bar.progress((i + 1) / len(url_list))
                    continue

                status_text.text(f"İnceleniyor: {url}...")
                try:
                    html = fetch_url_content(url)
                    text = clean_text(html)

                    if len(text) < 200:
                        results.append({"title": url, "error": "İçerik okunamadı (metin çok kısa).", "status": "error"})
                    else:
                        analysis = analyze_with_gemini(text, url, api_key)
                        results.append({"title": url, "data": analysis, "status": "success"})

                except Exception as e:
                    results.append({"title": url, "error": str(e), "status": "error"})

                progress_bar.progress((i + 1) / len(url_list))

            status_text.empty()
            progress_bar.empty()

            for res in results:
                if res['status'] == 'error':
                    st.error(f"❌ {res['title']} - Hata: {res['error']}")
                else:
                    data = res['data']
                    with st.expander(f"📄 {res['title']} (Kalite Skoru: {data.get('score', '?')}/100)", expanded=True):
                        st.info(f"**Özet:** {data.get('summary', 'Özet yok.')}")

                        corrections = data.get('corrections', [])
                        if corrections:
                            st.markdown("### 🛠️ Düzeltilmesi Gerekenler")
                            for item in corrections:
                                with st.container():
                                    c1, c2, c3 = st.columns([3, 3, 2])
                                    with c1:
                                        st.markdown("**🔴 Mevcut Hali**")
                                        st.error(f"\"{item.get('original', '-')}\"")
                                    with c2:
                                        st.markdown("**🟢 Önerilen Hali**")
                                        st.success(f"\"{item.get('correction', '-')}\"")
                                    with c3:
                                        st.markdown("**Sebep**")
                                        st.caption(item.get('reason', '-'))
                                    st.divider()
                        else:
                            st.success("✅ Belirgin bir hata bulunamadı.")

                        missing = data.get('missing_topics', [])
                        if missing:
                            st.markdown("### 💡 Eksik Konular")
                            for m in missing:
                                st.markdown(f"- {m}")
