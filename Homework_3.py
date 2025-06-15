import cv2
import numpy as np

def resize_image(img, width=400): # دالة لجعل كل الصور ذات حجم واحد ليسهل عرضها
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(img, (width, new_height))

image = cv2.imread('Annotation 2025-06-12 101508.png')  # قراءة الصورة
image_resized = resize_image(image) # تصغير الصورة الأصللية

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # تحويل الصورة إلى تدرجات الرمادي
gray_resized = resize_image(gray) # تصغير الصورة الرمادية

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # تحويل الصورة الرمادية إلى صورة ثنائية
binary_resized = resize_image(binary) # تصغير الصورة الثنائية

B, G, R = cv2.split(image_resized) # فصل قنوات الألوان
zeros = np.zeros_like(R)
red = cv2.merge([zeros, zeros, R]) # صورة القناة الحمراء
green = cv2.merge([zeros, G, zeros]) # صورة القناة الخضراء
blue = cv2.merge([B, zeros, zeros]) # صورة القناة الزرقاء

# عرض كل الصور
cv2.imshow('Original Image', image_resized)
cv2.imshow('Grayscale Image', gray_resized)
cv2.imshow('Binary Image', binary_resized)
cv2.imshow('Red Channel', red)
cv2.imshow('Green Channel', green)
cv2.imshow('Blue Channel', blue)

cv2.waitKey(0)
cv2.destroyAllWindows()
