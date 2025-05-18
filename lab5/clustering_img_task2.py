import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # Завантаження зображення
    image = cv2.imread('img_task2.jpg')
    NUMBER_OF_CLUSTERS = 3
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Зменшення розміру для пришвидшення обробки
    resized = cv2.resize(image_rgb, (image_rgb.shape[1] // 4, image_rgb.shape[0] // 4))

    # Фільтрація Гаусса (зменшення шуму)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # CLAHE для підвищення контрасту
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Збільшення насичення кольорів
    hsv = cv2.cvtColor(contrast_img, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], 50)
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Медіанний фільтр для згладжування
    img_blured = cv2.medianBlur(saturated, 15)

    # Глобальне покращення контрасту через гаму
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    final_img = cv2.LUT(img_blured, look_up_table)

    # Кластеризація (KMeans)
    pixels = final_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    clustered = kmeans.cluster_centers_[labels].reshape(final_img.shape).astype(np.uint8)

    # Кластеризація без редагування для порівняння
    original_pixels = image_rgb.reshape(-1, 3)
    kmeans_orig = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
    labels_orig = kmeans_orig.fit_predict(original_pixels)
    clustered_orig = kmeans_orig.cluster_centers_[labels_orig].reshape(image_rgb.shape).astype(np.uint8)

    # Візуалізація без редагування
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Оригінальне зображення')
    axes[0].axis('off')
    axes[1].imshow(clustered_orig)
    axes[1].set_title('Кластеризація без покращень')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Маска для виділення кластерів
    mask = np.zeros_like(final_img)
    for i in range(NUMBER_OF_CLUSTERS):
        mask[labels.reshape(final_img.shape[:2]) == i] = kmeans.cluster_centers_[i]
    mask = mask.astype(np.uint8)

    # Накладення маски на оригінал
    overlay = cv2.addWeighted(cv2.resize(image_rgb, (mask.shape[1], mask.shape[0])), 0.5, mask, 0.5, 0)

    # Візуалізація кластеризації з покращенням
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.resize(image_rgb, (mask.shape[1], mask.shape[0])))
    axes[0].set_title('Оригінальне зображення (зменшене)')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('Кластеризація з редагуванням')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

