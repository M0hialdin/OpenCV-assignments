import cv2
import numpy 
import glob
import os

# أبعاد لوح الشطرنج
number_of_squares_X = 8  # عدد المربعات رأسياً
number_of_squares_Y = 8   # عدد المربعات أفقياً
nX = number_of_squares_X - 1  # زوايا المربعات الداخلية الرأسية
nY = number_of_squares_Y - 1  # زوايا المربعات الداخلية الأفقية 
# لابد من ايجاد زوايا المربعات الداخلية لأنها نقاط دقيقة ومعروفة مما يسهل عملية المعايرة

square_size = 1.0 # حجم مربع الشطرنج على الحقيقة

input_folder = '.' # ادخال كل الصور الموجودة في كل الملف بشرط أن يكون الكود موجود في نفس الملف
output_folder = 'output' # خلق ملف لتخزين الصور بعد ايجاد الزوايا الداخلية
os.makedirs(output_folder, exist_ok=True) # استخدمت المكتبة هنا للتأكد من وجود الملف
image_pattern = os.path.join(input_folder, '*.jpg')

# تجهيز كل نقاط
objp = np.zeros((nY * nX, 3), np.float32)
objp[:, :2] = np.mgrid[0:nX, 0:nY].T.reshape(-1, 2)
objp *= square_size

# تخزين النقاط من اجل المعايرة
objectpoints = []  # نقاط الجسم
imagepoints = []  # النقاط على مستوي الصورة

# تخزين أبعاد الصور
image_size = None

# أولاً يجب ايجاد زوايا كل الصور
images = glob.glob(image_pattern) # للبحث عن كل الصور بدل ادخالها صورة صورة بشكل مفرد glob استخدمت مكتبة 
for fname in images:
    image = cv2.imread(fname) # قراءة الصور
    if image is None: 
        print(f"could not load the image {fname}") # رسالة في حال وجد خطأ في تحميل الصور
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # تحويل الصور لتدرج الرمادي لسهولة معالجتها
    image_size = gray.shape[::-1]  # حفظ أبعاد الصورة (العرض والارتفاع) لاستخدامها في المعايرة

    success, corners = cv2.findChessboardCorners(gray, (nX, nY)) # ايجاد الزوايا

    if success:
        objectpoints.append(objp) # حفظ مواقع الزوايا في العالم الحقيقي
        imagepoints.append(corners) # حفظ مواقع الزوايا في الصورة

        cv2.drawChessboardCorners(image, (nX, nY), corners, success) # رسم زوايا لوح الشطرنج
        for pt in corners:
            p = tuple(pt[0].astype(int))
            cv2.circle(image, p, 5, (0, 255, 255), -1)
            cv2.circle(image, p, 8, (255, 0, 255), 2)
        name = os.path.basename(fname).split('.')[0]
        cv2.imwrite(os.path.join(output_folder, f"{name}_corners.jpg"), image) # تخزين الصور بعد ايجاد الزوايا

# بداية المعايرة
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, image_size, None, None)

# عرض نتائج المعايرة
print("\n=== Calibration Results ===")
print("Intrinsic Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist.ravel())

for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        T = tvecs[i]
        print(f"\nImage {i + 1}:")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)
