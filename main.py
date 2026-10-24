import os
import uuid
import tempfile
import base64
import edge_tts
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
أنت معلم وموجه ذكي للطلاب.
أسلوبك مريح ولهجتك "بيضاء شبه عامية" ودودة جداً ومبسطة (مثل معلم يدردش مع طالبه).
القواعد الصارمة:
1. استخرج السؤال من الصورة بدقة.
2. ممنوع تعطي الحل النهائي أو الإجابة المباشرة نهائياً.
3. رحب بالطالب بكلمة خفيفة بالعامية البيضاء، ووضح فكرة السؤال ببساطة.
4. اطلب منه الخطوة الأولى فقط واسأله سؤال يساعده يبدأ الحل.
5. إذا قال "لم أفهم": اشرح له نفس الفكرة بتشبيه من واقع الحياة أو قسّمها لأصغر خطوة.
6. إذا قال "فهمت": شجعه بكلمة لطيفة وانتقل للخطوة اللي بعدها.
7. لا تكتب رموز ومعادلات إنجليزية معقدة لأن الكلام سيتحول لصوت مسموع.
"""

# دالة لاختيار أفضل نموذج متاح وشغال في حسابك تلقائياً
def get_working_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    
    # قائمة النماذج حسب الأفضلية
    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-pro-vision"
    ]
    
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for pref in preferred_models:
            if pref in available_models:
                return genai.GenerativeModel(model_name=pref, system_instruction=SYSTEM_PROMPT)
        # إذا لم يجد في القائمة، يأخذ أول نموذج رؤية متاح
        if available_models:
            return genai.GenerativeModel(model_name=available_models[0], system_instruction=SYSTEM_PROMPT)
    except Exception:
        pass
    
    return genai.GenerativeModel(model_name="gemini-1.5-flash-latest", system_instruction=SYSTEM_PROMPT)

# تحويل النص إلى صوت بشري بلهجة بيضاء مريحة (شخصية حامد)
async def generate_speech(text: str) -> str:
    try:
        communicate = edge_tts.Communicate(text, "ar-SA-HamedNeural", rate="+5%")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        await communicate.save(temp_path)
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(temp_path)
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

sessions = {}

class FeedbackRequest(BaseModel):
    session_id: str
    action: str

@app.post("/analyze")
async def analyze_question(file: UploadFile = File(...)):
    try:
        model = get_working_model()
        session_id = str(uuid.uuid4())
        image_bytes = await file.read()
        
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        chat = model.start_chat(history=[])
        prompt = "شف السؤال الموضح بالصورة، وضح فكرته بالعامية البيضاء ووجهني للخطوة الأولى فقط بدون حل:"
        
        response = chat.send_message([image_part, prompt])
        sessions[session_id] = chat
        
        guidance_text = response.text
        audio_base64 = await generate_speech(guidance_text)
        
        return {
            "session_id": session_id,
            "guidance": guidance_text,
            "audio": audio_base64
        }
    except Exception as e:
        return {"guidance": f"حدث خطأ أثناء الاتصال: {str(e)}", "audio": ""}

@app.post("/feedback")
async def handle_feedback(data: FeedbackRequest):
    try:
        chat = sessions.get(data.session_id)
        if not chat:
            return {"guidance": "انتهت الجلسة، صور السؤال مجدداً ونبدأ من جديد.", "audio": ""}
        
        user_msg = "فهمت عليك، وش نسوي الحين؟" if data.action == "understood" else "ما فهمت، وضحها لي بأسلوب أسهل ومثال من الحياة اليومية."
        response = chat.send_message(user_msg)
        guidance_text = response.text
        audio_base64 = await generate_speech(guidance_text)
        
        return {"guidance": guidance_text, "audio": audio_base64}
    except Exception as e:
        return {"guidance": "تعذر الرد، يرجى المحاولة ثانية.", "audio": ""}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
