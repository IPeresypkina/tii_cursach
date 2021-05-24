''''
Захват нескольких морд от нескольких животных для хранения в базе данных (каталог набора данных)
==> Морды будут храниться в каталоге: набор данных
==> Каждая морда будет иметь уникальный числовой идентификатор целого числа 1, 2, 3 и т. Д.
'''

import cv2
import matplotlib.pyplot as plt
from Import.ImportCat import cat_files

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalcatface.xml')
nose_Cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_nosecat.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

# Initialize individual sampling face count
count = 0

# загрузка изображения
img = cv2.imread(cat_files[3])
# преобразовать изображение в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# найти кошек на изображении
cats = face_cascade.detectMultiScale(gray)
# вывести количество обнаруженных котов
print('Number of cats detected:', len(cats))

# рисуем рамку вокруг каждой морды
for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face_cate = img[y:y + h, x:x + w]
    count += 1
    # Сохранение захваченного изображение в папку наборов данных
    cv2.imwrite("../dataset/cat." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
    cv2.imshow('image', img)

    nose = nose_Cascade.detectMultiScale(gray)
    for (xx, yy, ww, hh) in nose:
        cv2.rectangle(face_cate, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
    count += 1
    # Сохранение захваченного изображение в папку наборов данных
    cv2.imwrite("../dataset/cat." + str(face_id) + '.' + str(count) + ".jpg", gray[yy:yy + hh, xx:xx + ww])
    cv2.imshow('image', face_cate)


# конвертировать изображение BGR в RGB для печати
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# отобразить изображение вместе с ограничительной рамкой
plt.imshow(cv_rgb)
plt.show()