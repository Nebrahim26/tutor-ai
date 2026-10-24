import os
import uuid
import tempfile
import base64
import edge_tts
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
أنت معلم وموجه ذكي للطلاب، أسلوبك ودود ولهجتك "بيضاء شبه عامية" مريحة ومبسطة جداً (مثل أسلوب معلم يدردش مع طالبه).
القواعد:
1. استخرج السؤال من الصورة بدقة.
2. ممنوع تعطي الحل النهائي أو الإجابة أبداً!
3. ابدأ بترحيب خفيف بالعامية البيضاء، ووضّح فكرة المسألة ببساطة بدون تعقيد.
4. اعطه توجيه للخطوة الأولى فقط، واسأله سؤال يساعده يبدأ.
5. إذا قال "لم أفهم": لا تكرر نفس الكلام، اشرح له نفس الفكرة بتشبيه بسيط من الحياة اليومية.
6. إذا قال "فهمت": شجعه بكلمة لطيفة وانتقل للخطوة اللي بعدها.
7. لا تستخدم رموز رياضية إنجليزية معقدة داخل النص حتى ينطقها الصوت بشكل سليم وطبيعي.
"""

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)

# دالة تحويل النص إلى صوت طبيعي بلهجة خليجية/بيضاء
async def text_to_speech_base64(text: str) -> str:
    # صوت طبيعي جداً وعفوي (شخصية حامد - صوت خليجي واضح ومريح)
    communicate = edge_tts.Communicate(text, "ar-SA-HamedNeural", rate="+5%")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
    
    await communicate.save(temp_path)
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(temp_path)
    return base64.b64encode(audio_bytes).decode("utf-8")

sessions = {}

class FeedbackRequest(BaseModel):
    session_id: str
    action: str

@app.post("/analyze")
async def analyze_question(file: UploadFile = File(...)):
    try:
        client = get_client()
        session_id = str(uuid.uuid4())
        image_bytes = await file.read()
        
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7
            )
        )
        
        prompt = "شوف السؤال اللي بالصورة، بسط لي فكرته بالعامية البيضاء وعطني أول خطوة أسويها فقط بدون ما تحل:"
        response = chat.send_message([
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ])
        
        guidance = response.text
        audio_base64 = await text_to_speech_base64(guidance)
        sessions[session_id] = chat
        
        return {
            "session_id": session_id,
            "guidance": guidance,
            "audio": audio_base64
        }
    except Exception as e:
        # محاولة احتياطية مع نموذج 1.5 في حال الرغبة
        try:
            client = get_client()
            chat = client.chats.create(
                model="gemini-1.5-flash",
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
            )
            response = chat.send_message([
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                "شوف المسألة وبسط فكرتها بالعامية البيضاء ووجهني للخطوة الأولى فقط:"
            ])
            guidance = response.text
            audio_base64 = await text_to_speech_base64(guidance)
            sessions[session_id] = chat
            return {"session_id": session_id, "guidance": guidance, "audio": audio_base64}
        except Exception as err:
            return {"guidance": f"حدث خطأ: {str(err)}", "audio": None}

@app.post("/feedback")
async def handle_feedback(data: FeedbackRequest):
    chat = sessions.get(data.session_id)
    if not chat:
        return {"guidance": "انتهت الجلسة، صور السؤال مرة ثانية ونبدأ سوا.", "audio": None}
    
    user_msg = "فهمت عليك، ايش نسوي بعدها؟" if data.action == "understood" else "ما فهمت، ممكن تبسطها لي بمثال أسهل من واقع الحياة؟"
    response = chat.send_message(user_msg)
    guidance = response.text
    audio_base64 = await text_to_speech_base64(guidance)
    
    return {"guidance": guidance, "audio": audio_base64}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
