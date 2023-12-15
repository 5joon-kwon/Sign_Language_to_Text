import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="HandTracking")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 50px; font-weight: bold;'>Handtracking with image <br> 🤚🤞processing👈🤘</h1>
    </div>
""", unsafe_allow_html=True)
          
plt.rcParams['font.size']=3

# 손동작 관련 설정
max_num_hands = 1
gesture = {
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h',
    8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o',
    15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v',
    22:'w', 23:'x', 24:'y', 25:'z', 26:'spacing', 27:'clear',
    28:'okay', 29:'good', 30:'fxxk', 31:"I love you", 32:"Hi",
    33:"bad", 34:"gun"
}

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.2,
    min_tracking_confidence = 0.2)

# KNN 모델 및 데이터 로딩
f = open('test.txt','w')
file = np.genfromtxt("data/dataSet.txt", delimiter = ',')

angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 감마 보정 함수
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 이미지 향상 및 에지 강조 함수
def enhance_and_edge_enhance_image(img):
    gamma = 3.0
    img = adjust_gamma(img, gamma=gamma)

    # CLAHE를 적용하여 대비 향상
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_Lab2BGR)

    # 노이즈 감소 적용
    denoised_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)

    # Sobel 필터를 사용하여 에지 강조
    gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # 에지 이미지를 원본 이미지에 오버레이
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    sobel_colored = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
    final_img = cv2.addWeighted(denoised_img, 0.7, sobel_colored, 0.3, 0)

    return final_img

def blur_hand_region(img, landmarks, padding=20, blur_value=(45, 45)):
    if landmarks:
        x_min = min([lm.x for lm in landmarks]) * img.shape[1]
        x_max = max([lm.x for lm in landmarks]) * img.shape[1]
        y_min = min([lm.y for lm in landmarks]) * img.shape[0]
        y_max = max([lm.y for lm in landmarks]) * img.shape[0]

        x_min = max(0, int(x_min) - padding)
        x_max = min(img.shape[1], int(x_max) + padding)
        y_min = max(0, int(y_min) - padding)
        y_max = min(img.shape[0], int(y_max) + padding)

        # 손 영역에 블러 효과 적용
        img[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(img[y_min:y_max, x_min:x_max], blur_value, 0)

    return img

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 손동작 인식 변수 초기화
startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1
brightness_threshold = 50
image_container = st.empty()

st.markdown("""
    <style>
        .css-15zjjas {
            font-family: 'Gothic', sans-serif;
            font-size: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .divider {
        background-color: black;
        height: 3px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.markdown("<h1 style='font-family: Gothic, sans-serif; font-size: 18px; text-align: center;'>영상처리 전</h1>", unsafe_allow_html=True)
col1.video("data/before_image_processing.mp4")
col2.markdown("<h1 style='font-family: Gothic, sans-serif; font-size: 18px; text-align: center;'>영상처리 후</h1>", unsafe_allow_html=True)
col2.video("data/after_image_processing.mp4")

while True:
    ret, img = cap.read()
    if not ret:
        continue

    # 이미지의 평균 밝기 계산
    avg_brightness = np.mean(img)

    # 밝기가 임계값 이하인 경우에만 향상 및 에지 강조 함수 호출
    if avg_brightness < brightness_threshold:
        img = enhance_and_edge_enhance_image(img)

    # Mediapipe 손동작 인식
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    landmarks = []
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x,lm.y,lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11,
                        0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14 ,15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis = 1)[:, np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 
                           14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                           17, 18, 19], :]
            angle = np.arccos(np.einsum('nt, nt->n', compareV1, compareV2))
            angle = np.degrees(angle)
            if keyboard.is_pressed('a'):
                for num in angle:
                    nim = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write("34.000000")
                f.write('\n')
                print('next')
            data = np.array([angle], dtype = np.float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            index = int(results[0][0])
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += ' '
                        elif index == 27:
                            sentence = ''
                        elif index == 28:
                            sentence = 'okay'
                        elif index == 29:
                            sentence = 'good'
                        elif index == 30:
                            sentence = 'fxxk'
                        elif index == 31:
                            sentence = 'I love you'
                        elif index == 32:
                            sentence = 'HI'
                        elif index == 33:
                            sentence = 'bad'
                        elif index == 34:
                            sentence = 'gun'
                        else:
                            sentence += gesture[index]
                        startTime = time.time()
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),3)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            landmarks.extend([lm for lm in res.landmark])
    cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    if sentence == 'fxxk':
        img = blur_hand_region(img, landmarks)
    
    # 화면에 표시
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # OpenCV 프레임을 Pillow 이미지로 변환
    pil_image = Image.fromarray(frame_rgb)

    # Pillow 이미지를 Streamlit에 업데이트
    image_container.image(pil_image, channels="RGB")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# 자원 해제
cap.release()
f.close()