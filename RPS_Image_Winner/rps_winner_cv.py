import cv2
import numpy as np
import os

IMG_PATH = "images/test.jpg"
OUT_PATH = "results/rps_result.jpg"

def preprocess(img):
    # 피부색 후보 영역 추출(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)

    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def count_fingers_from_contour(cnt):
    # 너무 작으면 실패
    area = cv2.contourArea(cnt)
    if area < 3000:
        return None

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        # 주먹(바위)일 때 defects가 거의 없을 수 있음
        return 0

    finger_gaps = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # 삼각형 각도 기반 필터링(손가락 사이 ‘V’만 카운트)
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        if b * c == 0:
            continue
        angle = np.degrees(np.arccos((b*b + c*c - a*a) / (2*b*c)))

        # 깊이(d)도 같이 필터링(너무 얕은 결함 제외)
        if angle < 90 and d > 5000:
            finger_gaps += 1

    # finger_gaps(손가락 사이 V개수) → 대략 손가락 개수 추정
    # 보(펴진 손): gaps가 3~4 정도
    # 가위: gaps가 1 정도
    # 바위: gaps가 0
    return finger_gaps

def classify_rps(gaps):
    if gaps is None:
        return "UNKNOWN"
    if gaps <= 0:
        return "ROCK"
    elif gaps == 1:
        return "SCISSORS"
    else:
        return "PAPER"

def judge(left, right):
    if left == "UNKNOWN" or right == "UNKNOWN":
        return "UNKNOWN"
    if left == right:
        return "DRAW"
    if (left == "ROCK" and right == "SCISSORS") or \
       (left == "SCISSORS" and right == "PAPER") or \
       (left == "PAPER" and right == "ROCK"):
        return "LEFT WINS"
    return "RIGHT WINS"

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError("images/test.jpg 를 찾을 수 없습니다.")

    mask = preprocess(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 손 2개를 가장 큰 컨투어 2개로 가정
    if len(contours) < 2:
        out = img.copy()
        cv2.putText(out, "Need TWO hands (clear background)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(OUT_PATH, out)
        print("손 2개를 못 찾았습니다. 배경을 단순하게 다시 찍어주세요.")
        return

    hand_cnts = contours[:2]

    # 왼쪽/오른쪽 정렬
    boxes = [cv2.boundingRect(c) for c in hand_cnts]
    hand_cnts = [c for _, c in sorted(zip([b[0] for b in boxes], hand_cnts), key=lambda x: x[0])]

    out = img.copy()
    labels = []
    for idx, cnt in enumerate(hand_cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)

        gaps = count_fingers_from_contour(cnt)
        rps = classify_rps(gaps)
        labels.append(rps)

        cv2.putText(out, rps, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    result_text = judge(labels[0], labels[1])
    cv2.putText(out, result_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    os.makedirs("results", exist_ok=True)
    cv2.imwrite(OUT_PATH, out)
    print(f"저장 완료: {OUT_PATH} / LEFT={labels[0]} RIGHT={labels[1]} RESULT={result_text}")

if __name__ == "__main__":
    main()