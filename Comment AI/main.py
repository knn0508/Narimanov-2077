import os
import json
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

# Ətraf mühit dəyişənlərini (env) yüklə
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Loqların tənzimlənməsi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai2")

# İstifadəçidən gələcək (Frontend-dən) məlumatın strukturu
class ComplaintInput(BaseModel):
    user_description: str
    ai1_vision_data: Dict[str, Any]

# FastAPI Tətbiqinin yaradılması
app = FastAPI(title="AI 2 - Narimanov Complaint Analyzer")

# CORS Tənzimləmələri (Brauzer xətalarının qarşısını almaq üçün)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _extract_json_object(text: str) -> Optional[str]:
    """Mətndəki JSON strukturunu təmizləyib çıxaran xüsusi funksiya (Fallback)"""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if not stack:
                return None
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None

@app.on_event("startup")
def startup():
    """Server işə düşəndə API açarını və Modeli yoxlayıb aktiv edir"""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY tapılmadı! Zəhmət olmasa .env faylını yoxlayın.")
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 1. Mövcud modelləri yoxlayırıq və generateContent dəstəkləyən ilk flash və ya pro modelini tapırıq
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Ən uyğun modelin seçilməsi (Əvvəlcə flash, sonra pro axtarır)
        target_model = next((m for m in available_models if "flash" in m), None)
        if not target_model:
            target_model = next((m for m in available_models if "pro" in m), "gemini-pro")
            
        # 2. Tapılan düzgün model adı ilə inisializasiya edirik
        app.state.gmodel = genai.GenerativeModel(target_model)
        logger.info(f"Google Gemini API uğurla qoşuldu. İstifadə olunan model: {target_model}")
        
    except Exception as e:
        app.state.gmodel = None
        logger.exception(f"Modeli işə salarkən xəta baş verdi: {e}")

@app.post("/analyze-complaint")
async def analyze_complaint(payload: ComplaintInput):
    """
    Azərbaycan dilindəki şikayəti və AI 1-in məlumatlarını alıb Gemini ilə analiz edir.
    Gözlənilən nəticə: { consistency: bool, reason: str, text_urgency_score: int }
    """
    if not GEMINI_API_KEY or not getattr(app.state, "gmodel", None):
        raise HTTPException(status_code=500, detail="Serverdə Gemini API konfiqurasiyası tam deyil.")

    # Süni İntellektə veriləcək təlimat (Prompt)
    prompt = (
        "Siz Bələdiyyənin Şikayət Analizi üzrə ekspertisiniz. Yalnız və yalnız JSON formatında cavab verməlisiniz. Heç bir əlavə izah, markdown və ya kod bloku yox.\n"
        "Aşağıdakı Azərbaycan dilində yazılmış istifadəçi şikayəti (user_description) ilə AI 1 (Vision) tərəfindən verilən məlumatlar (ai1_vision_data) arasında uyğunluğu yoxlayın.\n\n"
        
        "QİYMƏTLƏNDİRMƏ ŞKALASI (text_urgency_score üçün 0-100 arası xal verin):\n"
        "- [0 - 20 xal]: Çox aşağı təcililik (Məsələn: Zibil qabı az doludur, heç bir ciddi qoxu və ya ətrafa dağılma yoxdur).\n"
        "- [21 - 50 xal]: Orta təcililik (Məsələn: Zibil qabı tam dolub, sadəcə rutin olaraq boşaldılmalıdır. Ciddi narahatlıq yoxdur).\n"
        "- [51 - 80 xal]: Yüksək təcililik (Məsələn: Zibillər ətrafa dağılıb, pis qoxu var, vətəndaşlar üçün keçidə mane olur və ya estetik cəhətdən çox pis vəziyyətdədir).\n"
        "- [81 - 100 xal]: Kritik / Təcili müdaxilə (Məsələn: Bioloji təhlükə, yanğın riski, heyvanların zibilləri parçalaması, yolu tamamilə bağlaması və ya epidemik risk).\n\n"
        
        "DİQQƏT: 'consistency' (uyğunluq) - əgər istifadəçinin dedikləri ilə AI 1-in gördükləri (ai1_vision_data) üst-üstə düşürsə true, ziddiyyət təşkil edirsə false olmalıdır.\n\n"
        
        "Cavabda yalnız bu üç açar olsun: consistency (boolean), reason (Azərbaycan dilində qısa izah), text_urgency_score (0-100 integer).\n\n"
        
        f"USER_DESCRIPTION:\n{payload.user_description}\n\n"
        f"AI1_VISION_DATA:\n{json.dumps(payload.ai1_vision_data, ensure_ascii=False)}\n\n"
        
        "Cavab nümunəsi: {\"consistency\": true, \"reason\": \"Vətəndaşın dediyi kimi zibil qabı doludur və dağılıb.\", \"text_urgency_score\": 75}\n"
    )

    # 1. Məlumatın Gemini-yə göndərilməsi
    try:
        response = app.state.gmodel.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2  # Təkcə bu qalır. max_tokens və mime_type SİLİNİR!
            ),
        )
    except Exception as e:
        logger.exception(f"Gemini API xətası: {e}")
        raise HTTPException(status_code=502, detail="Süni intellekt modelinə qoşulmaq mümkün olmadı.")

# 2. Cavabın mətn olaraq oxunması (Rəsmi və təmiz üsul)
    try:
        text_output = response.text
        # BİZƏ LAZIM OLAN ƏSAS YER: Modelin nə dediyini terminalda görmək üçün
        logger.info(f"Modelin xam cavabı:\n{text_output}") 
    except Exception as e:
        logger.error(f"Cavabı mətnə çevirmək olmadı: {e}")
        raise HTTPException(status_code=502, detail="Süni intellekt boş və ya xətalı cavab qaytardı.")

    # SÜNİ İNTELLEKTİN YAZDIĞI MARKDOWN (```json) İŞARƏLƏRİNİ SİLİRİK
    text_output = text_output.strip()
    if text_output.startswith("```json"):
        text_output = text_output[7:]
    if text_output.startswith("```"):
        text_output = text_output[3:]
    if text_output.endswith("```"):
        text_output = text_output[:-3]
    text_output = text_output.strip()

    # 3. JSON Məlumatının parçalanması (Parsing)
    parsed = None
    try:
        parsed = json.loads(text_output)
    except json.JSONDecodeError:
        json_sub = _extract_json_object(text_output)
        if json_sub:
            try:
                parsed = json.loads(json_sub)
            except Exception:
                raise HTTPException(status_code=502, detail="Modelin JSON obyekti zədəlidir.")
        else:
            raise HTTPException(status_code=502, detail="Model təmiz JSON formatında cavab qaytarmadı.")

    # 4. Açarların və tiplərin dəqiq yoxlanışı (Validation)
    expected_keys = {"consistency", "reason", "text_urgency_score"}
    if not expected_keys.issubset(set(parsed.keys())):
        raise HTTPException(status_code=502, detail=f"Bəzi açarlar çatışmır. Gələn açarlar: {list(parsed.keys())}")

    try:
        parsed["consistency"] = bool(parsed["consistency"])
        parsed["reason"] = str(parsed["reason"])
        
        # Xalı 0-100 arasında məhdudlaşdırır
        score = int(parsed["text_urgency_score"])
        parsed["text_urgency_score"] = max(0, min(100, score)) 
    except ValueError:
        raise HTTPException(status_code=502, detail="Dataların növündə (type) yanlışlıq var (məsələn, xal rəqəm deyil).")

    # Uğurlu cavabın Frontend-ə qaytarılması
    return parsed

if __name__ == "__main__":
    import uvicorn
    # Veb serverin işə salınması
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
