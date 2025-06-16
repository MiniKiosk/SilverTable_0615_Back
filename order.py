import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sys
import re

# 한글 수량 매핑 (필요시 더 추가 가능)
KOREAN_NUM = {
    "한": 1, "하나": 1,"않아":1,"아나":1,
    "두": 2, "둘": 2,
    "세": 3, "셋": 3,
    "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
    "열": 10
}

def extract_menus_with_quantity(text):
    orders = {}

    for menu in menu_list:
        if menu.startswith("돼지"):
            keywords = ["돼지"]
        elif menu.startswith("내장"):
            keywords = ["내장"]
        elif menu.startswith("섞어"):
            keywords = ["섞어"]
        elif menu.startswith("순대"):
            keywords = ["순대"]
        else:
            keywords = [menu]

        for keyword in keywords:
            if keyword in text:
                # 메뉴 앞뒤로 수량 찾기
                pattern = rf"{keyword}\s*([0-9]+|[가-힣]+)?"
                match = re.search(pattern, text)
                if match:
                    qty_str = match.group(1)
                    if qty_str:
                        qty = KOREAN_NUM.get(qty_str, None)
                        if qty is None:
                            try:
                                qty = int(qty_str)
                            except:
                                qty = 1
                    else:
                        qty = 1
                else:
                    qty = 1
                orders[menu] = orders.get(menu, 0) + qty

    return orders


# ✅ Whisper 모델 변경 (OpenAI 공식)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

menu_list = ["돼지국밥", "내장국밥", "섞어국밥", "순대국밥"]
# 오디오 정규화
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# 음성 녹음
def record_audio(duration=5, sample_rate=16000):
    print(f"🎙️ {duration}초간 녹음 중...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    audio = normalize_audio(audio)
    print("✅ 녹음 완료.")
    return audio

# 음성 → 텍스트 변환
def speech_to_text(audio, sample_rate=16000):
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    generated_ids = model.generate(input_features)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result.strip()

# 유사도 기반 메뉴 추출
def extract_menus(text):
    found_menus = []

    for menu, embedding in zip(menu_list, menu_embeddings):
        score = util.cos_sim(similarity_model.encode(text, convert_to_tensor=True), embedding)[0].item()
        if score > 0.45:  # 임계값 조금 낮춰도 가능
            found_menus.append((menu, round(score, 2)))

    # 중복 방지 및 점수 정렬 (선택)
    found_menus.sort(key=lambda x: -x[1])
    return [menu for menu, _ in found_menus]

# 주문 처리
def place_order(menu_dict):
    if menu_dict:
        for menu, qty in menu_dict.items():
            print(f"🧾 주문 완료: '{menu}' 메뉴 {qty}개가 주문되었습니다.")
    else:
        print("⚠️ 메뉴를 정확히 인식하지 못했습니다. 다시 말씀해주세요.")

# 메인 흐름
def main():
    while True:
        cmd = input("🎧 Enter 키로 녹음 / 'q' 입력 시 종료: ")
        if cmd.lower() == 'q':
            print("👋 프로그램을 종료합니다.")
            break

        audio = record_audio(duration=5)
        text = speech_to_text(audio)
        print(f"📝 인식된 텍스트: {text}")

        menu_dict = extract_menus_with_quantity(text)
        place_order(menu_dict)

if __name__ == "__main__":
    main()

