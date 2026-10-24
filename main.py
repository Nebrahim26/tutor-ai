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
6. اجعل الرد قصيراً ومباشراً وبلهجة فصحى مبسطة وعفوية لأن النص سيُنطق بصوت مسموع.
"""

def get_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    # استخدام flash-latest يضمن التوافق المباشر
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        system_instruction=SYSTEM_PROMPT
    )

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
        prompt = "حلل السؤال الموضح في الصورة، واشرح فكرته باختصار ووجهني للخطوة الأولى فقط دون حل مباشر:"
        
        response = chat.send_message([image_part, prompt])
        sessions[session_id] = chat
        
        return {
            "session_id": session_id,
            "guidance": response.text
        }
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return {"guidance": f"عذراً، حدث خطأ أثناء المعالجة: {str(e)}"}

@app.post("/feedback")
async def handle_feedback(data: FeedbackRequest):
    try:
        chat = sessions.get(data.session_id)
        if not chat:
            return {"guidance": "انتهت الجلسة، يرجى تصوير المسألة مجدداً لنبدأ معاً."}
        
        user_msg = "فهمت النقطة، ما هي الخطوة التالية؟" if data.action == "understood" else "لم أفهم، أرجو توضيحها لي بطريقة أخرى أبسط ومختلفة تماماً."
        
        response = chat.send_message(user_msg)
        return {"guidance": response.text}
    except Exception as e:
        print(f"Error in /feedback: {str(e)}")
        return {"guidance": "حدث خطأ أثناء الرد، يرجى المحاولة ثانية."}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
