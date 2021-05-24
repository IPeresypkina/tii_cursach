import cv2
import matplotlib.pyplot as plt

from Import.ImportCat import cat_files

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalcatface.xml')

# загрузка изображения
img = cv2.imread(cat_files[17])
# преобразовать изображение в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# найти кошек на изображении
cats = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

# вывести количество обнаруженных котов
print('Number of cats detected:', len(cats))

# рисуем рамку вокруг каждой морды
for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# конвертировать изображение BGR в RGB для печати
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# отобразить изображение вместе с ограничительной рамкой
plt.imshow(cv_rgb)
plt.show()