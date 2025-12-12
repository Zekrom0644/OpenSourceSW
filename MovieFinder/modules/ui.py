import cv2
import numpy as np
import textwrap

def show_result_card(movie_info):
    """
    OpenCV 창 출력
    """
    # 1. 검은 배경 생성
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # 2. 색상 및 폰트 설정
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    GRAY = (200, 200, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 3. 헤더
    cv2.putText(img, "MOVIE FINDER RESULT", (20, 40), font, 0.8, YELLOW, 2)
    cv2.line(img, (20, 50), (680, 50), YELLOW, 1)

    # 4. 정보 출력
    title_text = f"{movie_info['title']} ({movie_info['year']})"
    cv2.putText(img, title_text, (20, 100), font, 0.9, WHITE, 2)

    score_text = f"Match Score: {movie_info['score']*100:.1f}%"
    cv2.putText(img, score_text, (20, 140), font, 0.6, GREEN, 1)

    # 5. 줄거리 및 특수문자 처리
    plot_summary = movie_info['plot'][:280] + "..."
    plot_summary = plot_summary.replace("—", "-").replace("–", "-")
    wrapped_text = textwrap.wrap(plot_summary, width=65)

    y_pos = 200
    for line in wrapped_text:
        cv2.putText(img, line, (20, y_pos), font, 0.5, GRAY, 1)
        y_pos += 25

    # 6. 하단 안내 문구
    cv2.line(img, (20, 440), (680, 440), (50, 50, 50), 1)
    cv2.putText(img, "Press ANY KEY to close", (20, 470), font, 0.6, YELLOW, 1)

    # 결과창 설정
    window_name = 'Scene Seeker'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    cv2.imshow(window_name, img)
    
    # 아무 키나 누를 때까지 대기
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()