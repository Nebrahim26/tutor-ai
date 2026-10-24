import os
import uuid
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
3. رحب بالطالب بكلمة لطيفة بالعامية البيضاء، ووضح له فكرة السؤال باختصار وسهولة.
4. اطلب منه الخطوة الأولى فقط، واسأله سؤال يساعده يبدأ الحل بنفسه.
5. إذا قال "لم أفهم": اشرح له نفس الفكرة بتشبيه من واقع الحياة اليومية أو بسطها لأصغر جزئية.
6. إذا قال "فهمت": شجعه بحماس وانتقل معه للخطوة التالية.
7. اكتب بكلمات عربية واضحة ومباشرة وتجنب الرموز المعقدة حتى ينطقها الصوت بوضوح وسلاسة.
"""

def get_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    
    # استخدام النموذج الموصى به مباشرة
    try:
        return genai.GenerativeModel(model_name="gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)
    except Exception:
        return genai.GenerativeModel(model_name="gemini-2.0-flash-exp", system_instruction=SYSTEM_PROMPT)

sessions = {}

class FeedbackRequest(BaseModel):
    session_id: str
    action: str

@app.post("/analyze")
async def analyze_question(file: UploadFile = File(...)):
    try:
        model = get_model()
        session_id = str(uuid.uuid4())
        image_bytes = await file.read()
        
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        chat = model.start_chat(history=[])
        prompt = "شف المسألة اللي بالصورة، وضح فكرتها بالعامية البيضاء ووجهني للخطوة الأولى فقط بدون حل:"
        
        response = chat.send_message([image_part, prompt])
        sessions[session_id] = chat
        
        return {
            "session_id": session_id,
            "guidance": response.text
        }
    except Exception as e:
        # محاولة أخيرة مع الاسم الموصى به في الخطأ
        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=SYSTEM_PROMPT)
            chat = model.start_chat(history=[])
            response = chat.send_message([image_part, prompt])
            sessions[session_id] = chat
            return {"session_id": session_id, "guidance": response.text}
        except Exception as e2:
            return {"guidance": f"حدث خطأ أثناء الاتصال: {str(e2)}"}

@app.post("/feedback")
async def handle_feedback(data: FeedbackRequest):
    try:
        chat = sessions.get(data.session_id)
        if not chat:
            return {"guidance": "انتهت الجلسة، صور السؤال مرة ثانية ونبدأ سوا."}
        
        user_msg = "فهمت عليك، وش نسوي بالخطوة الجاية؟" if data.action == "understood" else "ما فهمت، وضحها لي بطريقة ثانية أسهل ومثال من الحياة."
        response = chat.send_message(user_msg)
        return {"guidance": response.text}
    except Exception as e:
        return {"guidance": "تعذر الرد، يرجى المحاولة ثانية."}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
