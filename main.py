from fastapi import FastAPI, HTTPException, Request, Body, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
import re
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost:3000",  # React default dev port
    "http://127.0.0.1:3000", # Also common for local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Whisper model
whisper_model_name = "openai/whisper-small"
tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file. Please set it.")
    # You might want to raise an exception or exit if the key is critical
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API configured and model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        gemini_model = None # Ensure it's None if initialization fails

# 메뉴 아이템과 가격 (원 단위)
menu_items = {
    "돼지국밥": 9000,
    "순대국밥": 10000,
    "내장국밥": 9500,
    "섞어국밥": 9500,
    "수육 반접시": 13000,
    "수육 한접시": 25000
}
menu_list = list(menu_items.keys())

class OrderRequest(BaseModel):
    audio_data: str  # base64 encoded audio data

class CorrectedOrder(BaseModel):
    original_text: str
    corrected_text: str
    order_items: dict

class VoiceCommand(BaseModel):
    text: str

# Korean number mapping
KOREAN_NUM = {
    "한": 1, "하나": 1, "않아": 1, "아나": 1, "일": 1,
    "두": 2, "둘": 2, "이": 2,
    "세": 3, "셋": 3, "삼": 3,
    "네": 4, "넷": 4, "사": 4,
    "다섯": 5, "오": 5,
    "여섯": 6, "육": 6,
    "일곱": 7, "칠": 7,
    "여덟": 8, "팔": 8,
    "아홉": 9, "구": 9,
    "열": 10, "십": 10
}

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def correct_text_with_gpt(text: str) -> str:
    """Use GPT to correct the recognized text for food ordering"""
    try:
        system_prompt = """
        당신은 한국어 음성 주문을 정확한 메뉴명과 수량으로 변환하는 전문가입니다.
        다음 메뉴 중에서 가장 유사한 메뉴로 교정하고, 수량을 정확히 인식해주세요:
        
        [메뉴 목록]
        1. 돼지국밥
        2. 내장국밥
        3. 섞어국밥
        4. 순대국밥
        5. 수육 한접시
        6. 수육 반접시

        [주문 예시]
        - "섞어 5개" → "섞어국밥 5개"
        - "내장 세 그릇" → "내장국밥 3개"
        - "돼지 둘" → "돼지국밥 2개"
        - "순대 하나" → "순대국밥 1개"
        - "수육 한접시 셋" → "수육 한접시 3개"
        - "수육 반 두 개" → "수육 반접시 2개"

        [지시사항]
        1. 수량은 반드시 보존하고, 누락되지 않도록 하세요.
        2. 수량 표현은 다음과 같이 변환하세요:
           - "하나", "한 개", "한 그릇", "일" → "1개"
           - "둘", "두 개", "두 그릇", "이" → "2개"
           - "셋", "세 개", "세 그릇", "서", "삼" → "3개"
           - "넷", "네 개", "네 그릇", "사" → "4개"
           - "다섯", "다섯 개", "오" → "5개"
           - "여섯", "육" → "6개"
           - "일곱", "칠" → "7개"
           - "여덟", "팔" → "8개"
           - "아홉", "구" → "9개"
           - "열", "십" → "10개"
        3. 메뉴 이름이 생략되었거나 줄여서 말해도 정확한 메뉴명으로 변환하세요.
        4. 최종 출력 형식은 반드시 "메뉴이름 수량" 형식으로 하세요.
        5. 수량이 없으면 "1개"로 기본값을 설정하세요.
        """
        
        user_prompt = f"""
        다음 음성 인식 결과를 분석하여 메뉴와 수량을 정확히 인식해주세요.
        수량은 반드시 보존하고, 메뉴는 정확한 이름으로 변환해주세요.
        입력: '{text}'
        """
        
        response = genai_model.generate_content(user_prompt)
        
        # 결과에서 불필요한 설명 제거
        result = response.text.strip()
        
        # "→" 기호 이후의 텍스트만 추출 (예시 형식이 있을 경우)
        if "→" in result:
            result = result.split("→")[-1].strip()
            
        return result
        
    except Exception as e:
        print(f"Error in GPT correction: {e}")
        return text

def extract_menus_with_quantity(text: str) -> dict:
    """Extract menu items and their quantities from text with improved quantity handling"""
    orders = {}
    
    # Korean number to integer mapping (expanded)
    korean_nums = {
        "한": 1, "하나": 1, "일": 1,
        "두": 2, "둘": 2, "이": 2, "두 개": 2,
        "세": 3, "셋": 3, "삼": 3, "서": 3, "세 개": 3,
        "네": 4, "넷": 4, "사": 4, "세 개": 4,
        "다섯": 5, "오": 5, "다섯 개": 5,
        "여섯": 6, "육": 6, "여섯 개": 6,
        "일곱": 7, "칠": 7, "일곱 개": 7,
        "여덟": 8, "팔": 8, "여덟 개": 8,
        "아홉": 9, "구": 9, "아홉 개": 9,
        "열": 10, "십": 10, "열 개": 10,
        "스물": 20, "이십": 20, "스무": 20,
        "서른": 30, "삼십": 30,
        "마흔": 40, "사십": 40,
        "쉰": 50, "오십": 50
    }
    
    # Menu keywords and their variations
    menu_keywords = {
        "돼지국밥": ["돼지국밥", "돼지 국밥", "돼지"],
        "순대국밥": ["순대국밥", "순대 국밥", "순대"],
        "내장국밥": ["내장국밥", "내장 국밥", "내장"],
        "섞어국밥": ["섞어국밥", "섞어 국밥", "섞어"],
        "수육 반접시": ["수육 반접시", "수육 반 접시", "반접시"],
        "수육 한접시": ["수육 한접시", "수육 한 접시", "수육"]
    }
    
    # Process each menu item
    for menu, keywords in menu_keywords.items():
        for keyword in keywords:
            if keyword in text:
                # Find quantity patterns before and after the keyword
                patterns = [
                    # Patterns for numbers with counters (e.g., "두 개", "세 그릇")
                    rf"([가-힣]+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)\s*{keyword}",
                    rf"{keyword}\s*([가-힣]+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)",
                    # Patterns for numbers (e.g., "두", "세")
                    rf"([가-힣]+)\s+{keyword}",
                    rf"{keyword}\s+([가-힣]+)",
                    # Patterns for numeric digits (e.g., "2개", "3 그릇")
                    rf"(\d+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)?\s*{keyword}",
                    rf"{keyword}\s*(\d+)\s*(?:개|그릇|접시|인분|인승|명|병|잔)?"
                ]
                
                qty = 1  # Default quantity
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match and match.group(1):
                        qty_str = match.group(1).strip()
                        
                        # Try to convert to integer
                        try:
                            qty = int(qty_str)
                            break
                        except ValueError:
                            # If not a digit, try Korean number
                            qty = korean_nums.get(qty_str, 1)
                            if qty != 1:  # If found in korean_nums
                                break
                
                # Add to orders
                if menu in orders:
                    orders[menu] += qty
                else:
                    orders[menu] = qty
                
                # Remove the matched text to avoid duplicate processing
                text = text.replace(keyword, "", 1)
                break
    
    return orders

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "menu_list": menu_list})

@app.get("/menu")
async def get_menu():
    """현재 메뉴 목록을 반환합니다."""
    return {"menu_items": menu_items}

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        # Read and process audio
        audio_data = await audio.read()
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Convert to text using Whisper
        input_features = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        generated_ids = whisper_model.generate(input_features)
        recognized_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Correct text using GPT
        corrected_text = correct_text_with_gpt(recognized_text)
        
        # Extract order items
        order_items = extract_menus_with_quantity(corrected_text)
        
        return {
            "original_text": recognized_text,
            "corrected_text": corrected_text,
            "order_items": order_items
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-voice-command")
async def process_voice_command(command_request: VoiceCommand):
    user_text = command_request.text
    print(f"Received text for processing: {user_text}")

    # Construct the prompt for Gemini
    # We want Gemini to classify intent and extract relevant information as JSON.
    menu_list_str = ", ".join(menu_items.keys())
    prompt = f"""
You are a helpful AI assistant for a Korean restaurant kiosk. Your task is to process a customer's voice input and determine their intent.
Available menu items are: {menu_list_str}.

The customer said: "{user_text}"

Analyze the customer's statement and respond in JSON format with the following structure:
{{
  "action": "<intent_type>",
  "payload": {{}}
}}

Possible <intent_type> values are:
1.  "order_processed": If the customer is placing an order. The payload should be an "order_items" object where keys are valid menu items and values are their quantities (e.g., {{"돼지국밥": 1, "소주": 2}}).
2.  "gemini_answer": If the customer is asking a question (e.g., about menu items, ingredients, recommendations, spiciness, or any general food-related question). The payload should contain an "answer" field with a helpful, concise, and friendly response in Korean.
3.  "call_staff": If the customer's request is not an order or a clear question that you can answer, or if they explicitly ask to call a staff member (e.g., "사장님 불러주세요", "계산할게요", "도와주세요"). The payload can be an empty object or contain a "message" like "Calling staff for assistance."
4.  "order_complete": If the customer indicates they are finished with their order or do not want to add more items (e.g., in response to "더 주문하시겠어요?"). Common phrases include "됐어요", "아니요 괜찮아요", "그게 다예요", "없어요", "안 시켜요", "필요 없어요". The payload can be an empty object or contain a confirmation message.
5.  "cancel_order": If the customer EXPLICITLY states they want to CANCEL the current ordering process, GO BACK to the start, STOP ordering, or LEAVE the kiosk. Common phrases include "취소", "주문 취소", "나갈래", "처음으로", "그만할래요", "주문 안 할래요", "됐어요 그만할래요", "안 할래요", "그만둘래요", "빠져나가고 싶어요", "첫 화면으로", "돌아가고 싶어요". This action signifies a desire to ABANDON the current interaction and return to a neutral state.

Important considerations:
-   CRITICAL DISTINCTION FOR "cancel_order" vs "order_complete": 
    * "cancel_order" = Customer wants to ABANDON/EXIT the entire ordering process and go back to the main screen (취소, 처음으로, 그만할래요, 안 할래요, 나갈래)
    * "order_complete" = Customer is saying NO to additional items but wants to CONTINUE with their current order (됐어요, 아니요 괜찮아요, 없어요 when asked about more items)
    
-   Context matters: If the system just asked "더 주문하시겠어요?" and customer says "됐어요" or "아니요", this is typically "order_complete" (no more items needed)
-   If customer says "됐어요 그만할래요" or "취소" or "처음으로", this is "cancel_order" (wants to exit completely)
-   If customer shows STRONG rejection like "안 할래요", "그만할래요", this should be "cancel_order"

-   If it's an order, only include items from the provided menu list.
-   If multiple items are ordered, include all of them.
-   If the quantity is not specified for an order item, assume 1.
-   If the text is unclear or seems like a greeting (e.g., "안녕하세요"), try to infer if it's leading to an order or question. If it's just a greeting with no other intent, AND NOT a cancellation, you can classify it as 'call_staff' with a simple acknowledgement or ask them what they need, or provide a generic greeting response via 'gemini_answer'. For this system, let's default to 'call_staff' for very generic or unclear statements not related to orders/questions/cancellations.
-   For questions, provide a direct and helpful answer in Korean. If the question is about a menu item not on the list, you can say it's not available.
-   Ensure the output is ONLY the JSON object, with no other text before or after it.

Example for an order: "돼지국밥 하나랑 맥주 두병 주세요"
{{
  "action": "order_processed",
  "payload": {{
    "order_items": {{
      "돼지국밥": 1,
      "맥주": 2
    }}
  }}
}}

Example for a question: "돼지국밥 많이 매워요?"
{{
  "action": "gemini_answer",
  "payload": {{
    "answer": "저희 돼지국밥은 기본적으로 맵지 않게 제공됩니다. 원하시면 다대기를 추가해서 맵기를 조절하실 수 있습니다."
  }}
}}

Example for calling staff: "사장님!"
{{
  "action": "call_staff",
  "payload": {{
    "message": "직원을 호출합니다."
  }}
}}

Example for unclear/greeting: "네 안녕하세요"
{{
  "action": "call_staff", 
  "payload": {{ "message": "안녕하세요! 무엇을 도와드릴까요?" }} 
}}

Example for order completion (after being asked for more items): "됐어요"
{{
  "action": "order_complete",
  "payload": {{ "message": "네, 알겠습니다. 주문을 마무리합니다." }}
}}

Example for order completion (negative response): "아니요 괜찮아요"
{{
  "action": "order_complete",
  "payload": {{}}
}}

Example for order completion (nothing more): "없어요"
{{
  "action": "order_complete",
  "payload": {{}}
}}

Example for order completion (that's all): "네 그게 다예요"
{{
  "action": "order_complete",
  "payload": {{ "message": "알겠습니다." }}
}}

Example for cancelling the order/exiting: "주문 취소할래요"
{{
  "action": "cancel_order",
  "payload": {{ "message": "주문을 취소하고 처음 화면으로 돌아갑니다." }}
}}

Example for going back to the main screen: "처음으로"
{{
  "action": "cancel_order",
  "payload": {{ "message": "처음 화면으로 돌아갑니다." }}
}}

Example for stopping the process: "아니요 그만할래요"
{{
  "action": "cancel_order",
  "payload": {{}}
}}

Example for general refusal to order: "안 할래요"
{{
  "action": "cancel_order",
  "payload": {{"message": "알겠습니다. 도움이 필요하시면 다시 호출해주세요."}}
}}

Example for wanting to leave: "나갈래요"
{{
  "action": "cancel_order",
  "payload": {{"message": "첫 화면으로 돌아갑니다."}}
}}

Example for saying no to additional items (when asked "더 주문하시겠어요?"): "아니요 필요 없어요"
{{
  "action": "order_complete",
  "payload": {{"message": "네, 알겠습니다."}}
}}
"""

    try:
        print("Sending prompt to Gemini...")
        # print(f"Prompt: \n{prompt}") # Uncomment for debugging the prompt
        response = gemini_model.generate_content(prompt)
        
        # Debug: Print raw Gemini response
        print(f"Raw Gemini response text: {response.text}")

        # Extract JSON from the response text
        # Gemini might sometimes wrap JSON in ```json ... ``` or have leading/trailing text.
        raw_response_text = response.text.strip()
        json_start_index = raw_response_text.find('{')
        json_end_index = raw_response_text.rfind('}')
        
        if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
            json_str = raw_response_text[json_start_index : json_end_index+1]
            gemini_output = json.loads(json_str)
            print(f"Parsed Gemini JSON output: {gemini_output}")
        else:
            print("Error: Gemini response was not valid JSON or JSON markers not found.")
            # Fallback or default response if JSON parsing fails
            gemini_output = {
                "action": "call_staff",
                "payload": {"message": "죄송합니다, 요청을 처리하는 중 오류가 발생했습니다. 직원을 호출하겠습니다."}
            }

        # Validate and structure the final response to frontend
        action = gemini_output.get("action")
        payload = gemini_output.get("payload", {})

        if action == "order_processed":
            # 주문 처리 로직 ...
            return {"status": "order_processed", "order": payload.get("order_items"), "message": "주문이 처리되었습니다."}
        elif action == "gemini_answer":
            # Gemini 답변 처리 로직 ...
            return {"status": "answered", "message": payload.get("answer")}
        elif action == "call_staff":
            # 직원 호출 로직 ...
            return {"status": "staff_called", "message": payload.get("message", "직원을 호출합니다.")}
        elif action == "order_complete":
            # 주문 완료 로직 ...
            return {"status": "order_completed", "message": payload.get("message", "주문이 완료되었습니다.")}
        elif action == "cancel_order":
            # 주문 취소 로직 ...
            return {"status": "order_cancelled", "message": payload.get("message", "주문이 취소되었습니다. 처음 화면으로 돌아갑니다.")}
        else:
            print(f"Unknown action from Gemini: {action}")
            # 알 수 없는 액션 처리 (예: 기본 메시지 반환 또는 직원 호출)
            return {"status": "error", "message": "알 수 없는 요청입니다. 직원을 호출하시겠습니까?", "original_action": action}

    except Exception as e:
        print(f"Error processing voice command with Gemini: {e}")
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Error processing voice command with Gemini: {str(e)}"})

# Keep the old /order endpoint if it's still used for manual order completion confirmation, 
# or if frontend directly posts structured orders. 
# For now, assuming it might still be relevant for other parts or future use.
orders_db = [] # Placeholder
class OrderItem(BaseModel):
    item: str
    quantity: int
class Order(BaseModel):
    items: list[OrderItem]

@app.post("/order")
async def create_order(order: Order):
    total_price = 0
    order_details = []
    for item_order in order.items:
        item_name = item_order.item
        quantity = item_order.quantity
        if item_name not in menu_items:
            raise JSONResponse(status_code=404, detail=f"Item '{item_name}' not found in menu")
        price = menu_items[item_name] * quantity
        total_price += price
        order_details.append({"item": item_name, "quantity": quantity, "price_per_item": menu_items[item_name], "subtotal": price})
    
    new_order = {"id": len(orders_db) + 1, "details": order_details, "total_price": total_price}
    orders_db.append(new_order)
    return {"message": "Order created successfully", "order": new_order}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
