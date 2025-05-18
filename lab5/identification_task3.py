import cv2
import numpy as np

# Функція обробки зображення
def apply_filters(image, brightness=0, contrast=1.0, sharpness=0, blur=0, hue=0):
    img = image.copy()

    # Яскравість і контраст
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # Зміна відтінку (Hue)
    if hue != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Різкість
    if sharpness > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpness, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    # Розмиття
    if blur > 0:
        k = 2 * blur + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def on_apply(img, img_original):
    # Перетворення в сіре і бінаризація
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Морфологічне замикання (з'єднання розірваних контурів)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Пошук контурів
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Вікно і повзунок
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Min Area', 'Contours', 0, 10000, lambda x: None)
    cv2.setTrackbarPos('Min Area', 'Contours', 540)

    while True:
        img_top = img.copy()
        img_bottom = img_original.copy()
        min_area = cv2.getTrackbarPos('Min Area', 'Contours')

        # Фільтрація контурів
        filtered = [cnt for cnt in all_contours if cv2.contourArea(cnt) > min_area]

        # Оброблене зображення (верхнє)
        for cnt in filtered:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_top, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(img_top, filtered, -1, (0, 255, 0), 2)
        cv2.putText(img_top, f'Objects: {len(filtered)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Оригінал з прямокутниками (нижнє зображення)
        for cnt in filtered:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_bottom, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Об’єднання зображень вертикально
        combined = np.vstack((img_top, img_bottom))
        cv2.imshow("Contours", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cv2.destroyWindow("Contours")

if __name__ == "__main__":
    image = cv2.imread('img_task3.png')
    if image is None:
        print("Помилка: не вдалося завантажити зображення. Перевірте шлях до файлу.")
        exit(1)

    # Збільшуємо зображення для кращої видимості
    image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)

    # Створюємо повзунки
    cv2.createTrackbar('Brightness', 'Processed', 50, 100, lambda x: None)     # -50...50
    cv2.setTrackbarPos('Brightness', 'Processed', 85)
    cv2.createTrackbar('Contrast', 'Processed', 10, 50, lambda x: None)        # 1.0...5.0
    cv2.setTrackbarPos('Contrast', 'Processed', 30)
    cv2.createTrackbar('Sharpness', 'Processed', 0, 5, lambda x: None)
    cv2.setTrackbarPos('Sharpness', 'Processed', 1)
    cv2.createTrackbar('Blur', 'Processed', 0, 10, lambda x: None)
    cv2.setTrackbarPos('Blur', 'Processed', 5)
    cv2.createTrackbar('Hue', 'Processed', 90, 180, lambda x: None)            # -90...+90
    cv2.setTrackbarPos('Hue', 'Processed', 86)
    cv2.createTrackbar('Apply', 'Processed', 0, 1, lambda x: None)             # кнопка
    cv2.setTrackbarPos('Apply', 'Processed', 0)

    applied = False

    while True:
        b = cv2.getTrackbarPos('Brightness', 'Processed') - 50
        c = cv2.getTrackbarPos('Contrast', 'Processed') / 10.0
        s = cv2.getTrackbarPos('Sharpness', 'Processed')
        bl = cv2.getTrackbarPos('Blur', 'Processed')
        h = cv2.getTrackbarPos('Hue', 'Processed') - 90
        apply_btn = cv2.getTrackbarPos('Apply', 'Processed')

        processed = apply_filters(image, brightness=b, contrast=c, sharpness=s,
                                  blur=bl, hue=h)

        cv2.imshow("Original", image)
        cv2.imshow("Processed", processed)

        # При натисканні "Apply"
        if apply_btn == 1 and not applied:
            on_apply(processed, image.copy())
            applied = True
        elif apply_btn == 0:
            applied = False  # дозвіл на повторне натискання

        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
