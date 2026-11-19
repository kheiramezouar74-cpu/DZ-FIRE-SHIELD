# DZ-FIRE SHIELD — واجهة تجريبية عربية
# التشغيل: pip install -r requirements.txt && streamlit run app.py

from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(page_title="DZ-FIRE SHIELD", page_icon="🌲", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
.block-container { max-width: 1100px; padding-top: 1.5rem; }
.hero { padding: 1.2rem; border-radius: 16px; background: linear-gradient(120deg,#103b2d,#1f6846); color: white; }
.small { color: #64748b; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🌲 DZ-FIRE SHIELD</h1>
<h3>نظام تجريبي ذكي للمساعدة في الكشف المبكر عن حرائق الغابات</h3>
<p>نسخة أولية للبحث والتجربة — ليست نظام إنذار رسمي أو معتمد</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("القائمة")
    page = st.radio("انتقل إلى", ["الرئيسية", "تحليل صورة", "سجل التجارب", "حول المشروع"])
    st.divider()
    st.caption("نسخة تجريبية — لا ترسل تنبيهات إلى مصالح الحماية المدنية.")

if "records" not in st.session_state:
    st.session_state.records = []

if page == "الرئيسية":
    st.subheader("لوحة التحكم")
    c1,c2,c3 = st.columns(3)
    c1.metric("حالة النموذج", "غير مربوط" if not Path("best.pt").exists() else "ملف موجود")
    c2.metric("التجارب المسجلة", len(st.session_state.records))
    c3.metric("حالة التنبيه", "يدوي فقط")
    st.info("ابدئي برفع صورة من صفحة «تحليل صورة». يلزم إضافة نموذج YOLO مدرّب باسم best.pt حتى تظهر نتائج كشف آلية.")
    st.markdown("### كيف يعمل النموذج؟")
    st.write("1. رفع صورة من الغابة أو من كاميرا/درون.")
    st.write("2. تحليل الصورة بواسطة نموذج ذكاء اصطناعي مدرّب.")
    st.write("3. عرض الفئات المكتشفة مثل الدخان أو النار، إن كان النموذج مدرباً عليها.")
    st.write("4. مراجعة بشرية قبل اعتبار النتيجة إنذاراً.")

elif page == "تحليل صورة":
    st.subheader("تحليل صورة")
    uploaded = st.file_uploader("اختاري صورة للغابة أو الدخان أو النار", type=["jpg","jpeg","png","webp"])
    conf = st.slider("حد الثقة الأدنى", 0.05, 0.95, 0.25, 0.05)
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left,right = st.columns([1,1])
        with left:
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
        with right:
            model_path = Path("best.pt")
            if not model_path.exists():
                st.warning("لم يتم ربط نموذج مدرّب بعد.")
                st.write("تم استلام الصورة فقط. لا يمكن إصدار نتيجة كشف حقيقية دون ملف النموذج.")
                st.caption("لا تعرض هذه النسخة أي تنبؤ وهمي.")
            else:
                try:
                    from ultralytics import YOLO
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_path = tmp.name
                    model = YOLO(str(model_path))
                    result = model.predict(source=tmp_path, conf=conf, verbose=False)[0]
                    annotated = result.plot()
                    st.image(annotated[:,:,::-1], caption="نتيجة النموذج", use_container_width=True)
                    detections = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            detections.append({
                                "الفئة": model.names.get(class_id, str(class_id)),
                                "الثقة": round(float(box.conf.item()), 3)
                            })
                    if detections:
                        st.dataframe(pd.DataFrame(detections), use_container_width=True)
                        st.warning("نتيجة آلية أولية: يجب على مختص مراجعتها قبل أي إجراء.")
                    else:
                        st.info("لم يكتشف النموذج أجساماً فوق حد الثقة المحدد. هذا لا يثبت غياب الحريق.")
                    st.session_state.records.append({
                        "الصورة": uploaded.name,
                        "عدد الاكتشافات": len(detections),
                        "حد الثقة": conf,
                        "ملاحظة": "تحتاج مراجعة بشرية"
                    })
                except Exception as e:
                    st.error(f"تعذر تشغيل النموذج: {e}")
                    st.caption("تحققي من توافق ملف best.pt مع إصدار مكتبة Ultralytics.")

elif page == "سجل التجارب":
    st.subheader("سجل التجارب")
    if st.session_state.records:
        st.dataframe(pd.DataFrame(st.session_state.records), use_container_width=True)
        csv = pd.DataFrame(st.session_state.records).to_csv(index=False).encode("utf-8-sig")
        st.download_button("تحميل السجل CSV", csv, "dz_fire_shield_trials.csv", "text/csv")
    else:
        st.info("لا توجد تجارب مسجلة في هذه الجلسة.")

elif page == "حول المشروع":
    st.subheader("حول DZ-FIRE SHIELD")
    st.write("مشروع بحثي يجمع بين الهندسة المدنية والذكاء الاصطناعي لدراسة الكشف المبكر عن حرائق الغابات ودعم تقييم الخطر.")
    st.write("المنطقة التجريبية المقترحة: غابة أكفادو، الجزائر.")
    st.write("المكونات المخطط لها: كشف بصري للدخان/النار، بيانات التضاريس، الغطاء النباتي، الطقس، وإمكانية الربط مستقبلاً بمنصة جوية بدون طيار.")
    st.warning("هذه الواجهة نموذج أولي. لا توجد حالياً معايرة ميدانية أو شهادة تشغيل أو تكامل فعلي مع طائرة أو مصالح الطوارئ.")
    st.markdown("### حالة التطوير")
    st.write("- الواجهة العربية: جاهزة")
    st.write("- رفع الصور: جاهز")
    st.write("- تشغيل الكشف: ينتظر نموذج best.pt مدرّب")
    st.write("- بيانات أكفادو والطقس: لم تُربط بعد")
    st.write("- ربط الدرون والتنبيه الرسمي: غير منفذ")
