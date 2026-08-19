# PregnancySafe 🤰

نظام **RAG (Retrieval-Augmented Generation)** لتصنيف أمان الأعراض والأدوية أثناء الحمل، مبني بالكامل على مصادر طبية رسمية معتمدة (WHO، NICE، ACOG، FDA، MotherToBaby/LactMed) — مش مقالات عامة أو معرفة نموذج لغوي غير موثقة.

> ⚠️ هذا مشروع تعليمي/هاكاثون. مش بديل عن استشارة طبية. راجعي قسم [إخلاء المسؤولية](#إخلاء-المسؤولية) بالأسفل.

---

## لماذا هذا الهيكل

المشروع مبني بـ **src-layout** مع فصل حقيقي بين المسؤوليات، مش سكريبتات متلخبطة:

| الطبقة | المسؤولية | ليه منفصلة |
|---|---|---|
| `schemas/` | pydantic models (Disease, Medication, Trimester, Chunk) | أي خطأ في البيانات (تصنيف أمان غلط، رابط تالف) بيتكشف فورًا وقت التحميل، مش وقت العرض قدام اللجنة |
| `ingestion/` | تحميل PDFs + تنظيف + تقطيع | مستقلة عن الفهرسة، قابلة للاختبار بدون ChromaDB |
| `indexing/` | ChromaDB + embeddings | Wrapper نضيف عليه بسهولة (مثلاً نبدل الـ embedding model) |
| `retrieval/` | البحث + فلترة حسب score_threshold + الاقتباسات | لو نتيجة البحث مش واثقة كفاية، بترفض بدل ما تديك إجابة ملفقة |
| `safety/` | تصنيف الأدوية، كشف علامات الخطر، إخلاء المسؤولية | **أهم جزء في المشروع** — منفصل تمامًا وواضح للتقييم |
| `agent/` | المنسّق: يربط كل الطبقات فوق | نقطة دخول واحدة (`PregnancyAgent.ask()`) |

---

## البنية الكاملة

```
PregnancySafe/
├── README.md
├── pyproject.toml                  # pip install -e . — يخلي src/ قابل للـ import من أي مكان
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   └── config.yaml                 # كل المصادر الطبية بحالتها + إعدادات embeddings/chunking
│
├── data/
│   ├── raw/<disease>/               # حطي PDFs هنا (فقط verified_open — راجعي docs/)
│   ├── processed/                   # chunks بعد التقطيع (JSON) — تتولد من run_ingestion.py
│   ├── vectorstore/                 # ChromaDB الفعلي — تتولد من build_index.py
│   └── medication_safety.json       # 35 دواء عبر 8 أمراض، structured، مبني من docs/
│
├── src/pregnancysafe/
│   ├── schemas/                     # pydantic models
│   ├── ingestion/                   # loader.py + chunker.py
│   ├── indexing/                    # vector_store.py (ChromaDB wrapper)
│   ├── retrieval/                   # retriever.py + citation_formatter.py
│   ├── safety/                      # medication_tiers.py + red_flags.py + disclaimers.py
│   ├── agent/                       # pregnancy_agent.py — المنسّق الرئيسي
│   └── utils/                       # config_loader.py + logging_config.py
│
├── scripts/
│   ├── run_ingestion.py             # PDFs -> chunks
│   └── build_index.py               # chunks -> ChromaDB
│
├── app/
│   └── streamlit_app.py             # الواجهة التفاعلية
│
├── evaluation/
│   ├── eval_cases.json              # 10 حالات: red_flag, safe_refusal, medication_tier
│   └── evaluate.py                  # يشغّل الحالات ويطبع pass/fail
│
├── tests/                           # 45 اختبار، كلها pass
│   ├── test_schemas.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_safety.py
│
└── docs/
    ├── PregnancySafe_Medical_Sources.md      # مصادر كل مرض بالحالة (🟢/🟡/🔴)
    ├── PregnancySafe_Medication_Safety.md    # جداول تصنيف الأدوية
    └── PregnancySafe_Medication_Links.md     # روابط كل دواء (MotherToBaby/LactMed)
```

---

## التشغيل

### 1. التثبيت

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .                  # يثبت المشروع + كل الاعتماديات من pyproject.toml
```

### 2. تشغيل الاختبارات (مايحتاجش أي بيانات أو ChromaDB)

```bash
pytest tests/ -v
```

### 3. تحميل الـ PDFs

راجعي `docs/PregnancySafe_Medical_Sources.md` — كل المصادر المؤكدة (🟢) لازم تتحمل يدويًا وتتحط في `data/raw/<disease_id>/`. المصادر بحالة 🟡 محتاجة فتح يدوي من المتصفح (bot detection مش رفض حقيقي). المصادر 🔴 مقفولة فعليًا (عضوية مطلوبة).

لو بتستخدمي سكريبت تحميل دفعة واحدة (batch downloader) بتاعك، الـ ingestion دلوقتي بيدعم:
- **ملفات متداخلة في subfolders** (`guidelines/`, `medications/<drug>/`) — مش لازم تبقى flat في المجلد مباشرة
- **PDF وHTML مع بعض** — صفحات زي ACOG UTI وCochrane وFDA بتترفع كـ HTML وبتتقرا صح
- **ملف log** بصيغة `DOWNLOADED: <path> <- <url>` — لو حطيتيه في `data/raw/download_sources.log`، الـ ingestion هيستخدمه يربط كل ملف بمصدره الحقيقي تلقائيًا للاقتباسات

⚠️ **مهم:** أسماء مجلدات الأمراض لازم تطابق بالظبط الأسماء المستخدمة في `config/config.yaml` و`data/medication_safety.json` (`hypertension`, `anemia`, `thyroid`, `varicose_veins`, `gerd_heartburn`, إلخ)، عشان استرجاع البيانات (retrieval) وتصنيف الأدوية (medication tiers) يشتغلوا على نفس الـ disease_id. لو سكريبت التحميل بتاعك استخدم أسماء تانية (زي `hypertension_preeclampsia` أو `anemia_iron_deficiency`)، لازم تعيدي تسمية المجلدات الأولى:

```powershell
# PowerShell — من جوه مجلد المشروع
Rename-Item "data\raw\hypertension_preeclampsia" "hypertension"
Rename-Item "data\raw\anemia_iron_deficiency" "anemia"
Rename-Item "data\raw\thyroid_disorders" "thyroid"
Rename-Item "data\raw\varicose_veins_leg_edema" "varicose_veins"
Rename-Item "data\raw\heartburn_reflux" "gerd_heartburn"
```

### 4. بناء الـ Knowledge Base

```bash
python scripts/run_ingestion.py     # PDFs + HTML -> data/processed/*.json (يستخدم data/raw/download_sources.log تلقائيًا لو موجود)
python scripts/build_index.py       # data/processed/ -> ChromaDB
```

### 5. تشغيل التقييم

```bash
python evaluation/evaluate.py
```

يشتغل حتى **قبل** بناء الـ index — حالات `red_flag`، `safe_refusal`، و`medication_tier` مايحتجوش retrieval أصلًا.

### 6. تشغيل الواجهة

```bash
streamlit run app/streamlit_app.py
```

---

## منطق الأمان (Safety Pipeline)

كل سؤال بيعدي على `PregnancyAgent.ask()` بالترتيب ده بالظبط:

1. **كشف علامات الخطر** (`safety/red_flags.py`) — لو الأعراض بتوحي بحالة طارئة (تسمم حمل، التهاب كلوي، نزيف، hyperemesis)، الرد بيوقف هنا فورًا ويوجّه لطوارئ — من غير ما يلمس الـ retrieval خالص.
2. **فحص النطاق** — لو السؤال عن مرض مش من الـ 8 المدعومين، رفض آمن.
3. **الاسترجاع** (`retrieval/retriever.py`) — بيرفض أي نتيجة تحت `score_threshold` بدل ما يديك إجابة من نص مش relevant.
4. **تصنيف الدواء** (`safety/medication_tiers.py`) — بيحسب الـ tier الفعلي حسب الـ trimester (مثلاً NSAIDs بتتصعّد لـ 🔴 تلقائيًا بعد الأسبوع 20 حتى لو الدواء نفسه مصنف 🟡 بشكل عام).
5. **إخلاء المسؤولية** — بيتضاف **إجباريًا** لكل رد، من غير استثناء (فيه اختبار مخصص لده: `test_disclaimer_always_appended`).

---

## حالة المصادر الطبية

راجعي `docs/PregnancySafe_Medical_Sources.md` و `docs/PregnancySafe_Medication_Links.md` للتفاصيل الكاملة. ملخص سريع:

- 🟢 **نص كامل مجاني ومؤكد:** WHO (Antenatal Care، Pre-eclampsia)، ACOG (UTI)، Cochrane (Varicose Veins)، 17/18 دواء عبر MotherToBaby
- 🟡 **محتاجة فتح يدوي (bot detection مش رفض):** NICE (NG201، NG133)، ATA Thyroid 2026
- 🔴 **مقفولة فعليًا (عضوية):** ACOG Nausea/Vomiting (189)، ACOG Anemia (233) — استُخدم PubMed abstract كبديل جزئي موثّق

---

## إضافة مفتاح Groq API (اختياري — لردود أذكى بدل النص الجاهز)

بدون أي مفتاح، الـ agent بيرد بنص template بسيط من المقتطفات المسترجعة (كافي للاختبار والتقييم). لو عايزة ردود مصاغة بشكل طبيعي عن طريق نموذج لغوي حقيقي:

1. **اعملي مفتاح مجاني:** [console.groq.com/keys](https://console.groq.com/keys)
2. **انسخي `.env.example` لملف `.env` جديد:**
   ```bash
   cp .env.example .env
   ```
3. **افتحي `.env` وحطي المفتاح جوه** (الملف ده مش بيتبعت لأي حد ومستبعد من git تلقائيًا عبر `.gitignore`):
   ```
   GROQ_API_KEY=gsk_...
   ```
   > ⚠️ متحطيش المفتاح في الشات أو أي مكان عام — خليه في `.env` بتاعك بس.
4. **ثبّتي مكتبة groq:**
   ```bash
   pip install groq
   # أو: pip install -e ".[llm]"
   ```
5. كده خلاص — `app/streamlit_app.py` بيكتشف المفتاح تلقائيًا ويستخدم Groq (`src/pregnancysafe/llm/groq_composer.py`). لو مفيش مفتاح أو حصل خطأ في الـ API، الكود بيرجع تلقائيًا للـ template composer بدل ما يـ crash.

**مهم:** الـ Groq composer بيصيغ بس النص النهائي من المقتطفات المسترجعة (اللي retrieval فلترها مسبقًا). كشف علامات الخطر، فحص النطاق، تصنيف الأدوية، وإخلاء المسؤولية كلهم شغالين قبل/بعد الـ composer بغض النظر عنه — مفيش أي جزء من منطق الأمان بيتأثر باستخدام Groq من عدمه.

---

## إضافة مرض جديد



1. أضيفي المرض في `config/config.yaml` تحت `disease_sources` (label + folder + sources)
2. أضيفي مجلد `data/raw/<disease_id>/`
3. أضيفي أدويته في `data/medication_safety.json` تحت `diseases`
4. شغّلي `run_ingestion.py` و`build_index.py` من جديد

الـ pydantic schemas هترفض أي بيانات ناقصة أو غلط فورًا.

---

## إخلاء المسؤولية

المعلومات في هذا المشروع للأغراض **التعليمية والتوضيحية فقط** (مشروع هاكاثون)، ومبنية على مصادر طبية رسمية، لكنها **ليست** بديلاً عن استشارة طبيب مختص. أي قرار طبي فعلي لازم يكون تحت إشراف طبيب متابع للحالة.
