
# 🔧 Toolbox Detection using YOLOv8

Toolbox Detection is a real-time object detection project focused on identifying five commonly used tools — **Drill, Hammer, Pliers, Screwdriver, and Wrench** — using the powerful YOLOv8 deep learning architecture. This project was developed as part of a Vision AI internship and deployed successfully on the **YOLOvX mobile application** by [Wiserli](https://wiserli.com).

---

## 📱 Deployed On

**YOLOvX Mobile App** by Wiserli — Enables real-time on-device detection of tools using a custom-trained YOLOv8 model.

---

## 📁 Dataset

* **Total Images:** 5204

  * Training: 3640
  * Validation: 1034
  * Test: 529
* **Data Sources:**

  * [Kaggle](https://kaggle.com)
  * [Roboflow](https://roboflow.com)
  * Manually collected images
* **Classes Annotated:**

  1. Drill
  2. Hammer
  3. Pliers
  4. Screwdriver
  5. Wrench
* **Annotation Tool:** Roboflow

---

## 🧠 Model Details

* **Model Used:** YOLOv8
* **Framework:** Ultralytics YOLOv8
* **Model Format:** `best.pt`
* **Key Metrics Monitored:**

  * **Precision**
  * **Recall**
  * **mAP\@0.5**
  * **mAP\@0.5:0.95**
  * **Box Loss**
  * **Class Loss**
  * **DFL Loss**

---

## 💻 How to Run the Project

### 🔄 1. Clone the Repository

```bash
git clone https://github.com/mohitxgithub/toolbox_detection.git
cd toolbox_detection
```

### ⚙️ 2. Install Dependencies

Make sure you have Python 3.8+ and pip installed.

```bash
pip install ultralytics
```

> If you face issues, also try:

```bash
pip install -r requirements.txt
```

---

## 🎥 Run Detection on Webcam

Make sure your model (`best.pt`) is inside the project folder.

```bash
yolo task=detect mode=predict model=best.pt source=0
```

* `source=0` refers to your default webcam.
* Change `0` to `1`, `2`, etc., for external webcams.

---

## 📁 Run on Image/Video

```bash
# For an image
yolo task=detect mode=predict model=best.pt source="image.jpg"

# For a video
yolo task=detect mode=predict model=best.pt source="video.mp4"
```


