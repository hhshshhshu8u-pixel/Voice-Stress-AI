import io
import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import torch

from transformers import pipeline


# =========================================================
# إعدادات التطبيق
# =========================================================

st.set_page_config(
    page_title="Voice Stress Analyzer",
    page_icon="🎙️",
    layout="centered"
)

SAMPLE_RATE = 16000
DURATION = 5
MODEL_ID = "Aniemore/wav2vec2-emotion-v1-crosslingual"


# =========================================================
# تحميل نموذج Hugging Face مرة واحدة فقط
# =========================================================

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "audio-classification",
        model=MODEL_ID,
        device=device
    )


# =========================================================
# حساب Pitch
# =========================================================

def calculate_pitch(audio, sr):
    """
    يحسب متوسط وتذبذب التردد الأساسي للصوت F0.
    """

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) == 0:
        return 0.0, 0.0

    mean_pitch = float(np.mean(valid_f0))
    pitch_std = float(np.std(valid_f0))

    return mean_pitch, pitch_std


# =========================================================
# حساب Jitter تقريبي
# =========================================================

def calculate_jitter(audio, sr):
    """
    حساب Jitter تقريبي اعتماداً على تغيرات فترة النغمة
    بين الدورات الصوتية المتتالية.
    """

    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) < 3:
        return 0.0

    periods = 1.0 / f0

    differences = np.abs(np.diff(periods))

    mean_period = np.mean(periods)

    if mean_period == 0:
        return 0.0

    jitter = np.mean(differences) / mean_period

    return float(jitter)


# =========================================================
# تجهيز الصوت
# =========================================================

def prepare_audio(uploaded_audio):
    """
    قراءة ملف WAV وتحويله إلى Mono و16kHz
    ثم أخذ أول 5 ثوانٍ أو padding إذا كان أقصر.
    """

    audio_bytes = uploaded_audio.read()

    audio, sr = sf.read(io.BytesIO(audio_bytes))

    # Stereo -> Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # إعادة أخذ العينات إلى 16kHz
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )
        sr = SAMPLE_RATE

    target_length = SAMPLE_RATE * DURATION

    # أخذ 5 ثوانٍ فقط
    if len(audio) > target_length:
        audio = audio[:target_length]

    # إذا كان التسجيل أقصر من 5 ثوانٍ
    elif len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    return audio, sr


# =========================================================
# تحويل المشاعر إلى مؤشر توتر تجريبي
# =========================================================

def calculate_stress_index(emotions, pitch_std, jitter):
    """
    هذه معادلة تجريبية وليست نموذجاً طبياً.
    """

    emotion_scores = {
        item["label"].lower(): item["score"]
        for item in emotions
    }

    fear = emotion_scores.get("fear", 0.0)
    anger = emotion_scores.get("anger", 0.0)

    emotional_signal = (fear + anger) / 2

    # تحويل Pitch variability إلى نطاق 0-1 تقريبي
    pitch_signal = np.clip(pitch_std / 80.0, 0, 1)

    # Jitter عادة رقم صغير، لذلك نستخدم مقياساً مضخماً
    jitter_signal = np.clip(jitter / 0.05, 0, 1)

    stress = (
        0.60 * emotional_signal
        + 0.20 * pitch_signal
        + 0.20 * jitter_signal
    )

    return float(np.clip(stress * 100, 0, 100))


# =========================================================
# الواجهة
# =========================================================

st.title("🎙️ محلل الصوت ومؤشر التوتر")

st.write(
    """
    سجّل صوتك لمدة تصل إلى 5 ثوانٍ، وسيتم تحليل بعض الخصائص
    الصوتية مثل Pitch وJitter بالإضافة إلى المشاعر الصوتية.
    """
)

st.warning(
    """
    ⚠️ هذا المشروع تجريبي وتعليمي. النتيجة ليست تشخيصاً طبياً
    ولا تثبت أن الشخص متوتر أو غير متوتر.
    """
)

st.markdown("### 🎤 السماح باستخدام الميكروفون")

audio_file = st.audio_input(
    "اضغط هنا واسمح للمتصفح باستخدام الميكروفون",
    sample_rate=SAMPLE_RATE
)


# =========================================================
# معالجة التسجيل
# =========================================================

if audio_file is not None:

    st.success("تم استلام التسجيل.")

    st.audio(audio_file)

    if st.button("🔍 تحليل الصوت", type="primary"):

        with st.spinner("جاري تحليل الصوت..."):

            try:

                # -------------------------------------------------
                # تجهيز الصوت
                # -------------------------------------------------

                audio, sr = prepare_audio(audio_file)

                # -------------------------------------------------
                # Pitch
                # -------------------------------------------------

                mean_pitch, pitch_std = calculate_pitch(
                    audio,
                    sr
                )

                # -------------------------------------------------
                # Jitter
                # -------------------------------------------------

                jitter = calculate_jitter(
                    audio,
                    sr
                )

                # -------------------------------------------------
                # نموذج Hugging Face
                # -------------------------------------------------

                model = load_model()

                emotions = model(
                    {
                        "sampling_rate": sr,
                        "raw": audio
                    },
                    top_k=7
                )

                # -------------------------------------------------
                # مؤشر التوتر التجريبي
                # -------------------------------------------------

                stress_index = calculate_stress_index(
                    emotions,
                    pitch_std,
                    jitter
                )

                # =================================================
                # عرض النتائج
                # =================================================

                st.divider()

                st.subheader("📊 النتائج الصوتية")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "متوسط Pitch",
                        f"{mean_pitch:.1f} Hz"
                    )

                with col2:
                    st.metric(
                        "Pitch Variability",
                        f"{pitch_std:.1f} Hz"
                    )

                with col3:
                    st.metric(
                        "Jitter",
                        f"{jitter:.4f}"
                    )

                # -------------------------------------------------
                # مؤشر التوتر
                # -------------------------------------------------

                st.divider()

                st.subheader("🧠 مؤشر التوتر التجريبي")

                st.progress(
                    int(stress_index)
                )

                st.metric(
                    "Stress Index",
                    f"{stress_index:.1f}%"
                )

                if stress_index < 30:
                    st.info(
                        "المؤشر الصوتي منخفض في هذه العينة."
                    )

                elif stress_index < 60:
                    st.warning(
                        "المؤشر الصوتي متوسط في هذه العينة."
                    )

                else:
                    st.error(
                        "المؤشر الصوتي مرتفع في هذه العينة."
                    )

                # -------------------------------------------------
                # المشاعر التي اكتشفها النموذج
                # -------------------------------------------------

                st.divider()

                st.subheader("🎭 تصنيف المشاعر الصوتية")

                for emotion in emotions:

                    label = emotion["label"]
                    score = emotion["score"]

                    st.write(
                        f"**{label}** — "
                        f"{score * 100:.1f}%"
                    )

                    st.progress(
                        int(score * 100)
                    )

                # -------------------------------------------------
                # معلومات إضافية
                # -------------------------------------------------

                st.caption(
                    """
                    ملاحظة: مؤشر التوتر أعلاه عبارة عن معادلة تجريبية
                    تجمع بين مخرجات نموذج المشاعر وبعض خصائص الصوت.
                    لا ينبغي استخدامه لاتخاذ قرارات صحية أو تشخيصية.
                    """
                )

            except Exception as e:

                st.error(
                    f"حدث خطأ أثناء تحليل الصوت: {e}"
                )
