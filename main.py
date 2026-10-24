import os
import uuid
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# قراءة مفتاح الـ API من متغيرات البيئة السحابية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
أنت معلم وموجه تعليمي ذكي وودود للطلاب (Socratic Tutor).
القواعد الصارمة:
1. استخرج المسألة من الصورة بدقة.
2. لا تعطِ الحل النهائي أو الإجابة المباشرة مطلقاً مهما طلب الطالب.
3. التوجيه الأول: اشرح فكرة السؤال باختصار شديد، واطرح سؤالاً بسيطاً يقود الطالب للخطوة الأولى فقط.
4. إذا قال الطالب "لم أفهم":
   - لا تنتقل للخطوة التالية!
   - أعد شرح نفس النقطة السابقة بأسلوب مختلف كلياً وبسيط جداً، أو استخدم تشبيهاً سهلاً من واقع الحياة، ثم اسأله مجدداً.
5. إذا قال الطالب "فهمت":
   - شجّعه بكلمة لطيفة، ثم انتقل معه للخطوة التالية في التفكير حتى يصل للحل بنفسه.
6. اجعل الرد قصيراً ومباشراً وبلسان عفوي مسموع (تجنب رموز المعادلات المعقدة لأن النص سيُنطق بصوت).
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

sessions = {}

class FeedbackRequest(BaseModel):
    session_id: str
    action: str

# دالة إعادة المحاولة التلقائية لتفادي ضغط الطلبات مع 100 طالب
async def send_with_retry(chat, message, retries=3):
    for attempt in range(retries):
        try:
            return chat.send_message(message)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(2)

@app.post("/analyze")
async def analyze_question(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    image_bytes = await file.read()
    image_part = {
        "mime_type": file.content_type or "image/jpeg",
        "data": image_bytes
    }
    
    chat = model.start_chat(history=[])
    prompt = "حلل السؤال الموضح في الصورة، واشرح فكرته باختصار ووجهني للخطوة الأولى فقط:"
    
    response = await send_with_retry(chat, [image_part, prompt])
    sessions[session_id] = chat
    
    return {
        "session_id": session_id,
        "guidance": response.text
    }

@app.post("/feedback")
async def handle_feedback(data: FeedbackRequest):
    chat = sessions.get(data.session_id)
    if not chat:
        return {"guidance": "انتهت الجلسة، يرجى تصوير المسألة مجدداً لنبدأ معاً."}
    
    user_msg = "فهمت النقطة، ما هي الخطوة التالية؟" if data.action == "understood" else "لم أفهم، أرجو توضيحها لي بطريقة أخرى أبسط ومختلفة تماماً."
    
    response = await send_with_retry(chat, user_msg)
    return {"guidance": response.text}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
