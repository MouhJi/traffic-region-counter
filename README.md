## 🚦 AI Traffic Region Counter using YOLO & Ultralytics Solutions

An **AI-powered traffic monitoring system** that detects and counts vehicles **within predefined regions** in a video stream using **YOLO11** and **Ultralytics Solutions**.  
The project demonstrates how computer vision can assist in **intelligent traffic management** by tracking vehicles and analyzing regional traffic density in real-time.

Here is my project demo using YOLO library:



https://github.com/user-attachments/assets/ad7f481e-0665-4b62-8e2e-509bb167f8eb



---

## 🧭 Project Overview

This project performs real-time **vehicle detection, tracking, and counting** using AI:

- Detects multiple vehicle types (car, bus, truck, etc.)
- Defines multiple polygon regions (lanes, intersections, entry zones)
- Tracks unique vehicle IDs across frames
- Counts how many vehicles **enter or exit** each region
- Prevents duplicate counting using unique object IDs
- Displays total counts directly on the output video

---

## 🏗️ Project Structure

vehicle_detected/

├── vehicle_detected.py # Main solution class (CounterObject)

└── Rrequirements.txt # Required libraries

---

## 🧠 How It Works

1. YOLO detects all vehicles in each video frame.
2. Each vehicle is assigned a unique **track ID** for multi-frame tracking.
3. The system calculates the **centroid** of each vehicle’s bounding box.
4. If the centroid enters a **user-defined polygon region**, the count for that region increases.
5. The system stores counted IDs to prevent double-counting while the vehicle remains inside the region.

---

## ⚙️ Configuration Example

```python
object_classes = [2, 5, 7]  # car, truck, bus
region_points = [
    [[1805, 455], [1805, 363], [1526, 302], [1400, 336]],
    [[1015, 949], [915, 528], [918, 380], [1007, 382], [1385, 954]],
    ...
]

counter = CounterObject(
    show=True,
    region=region_points,
    model="yolo11n.pt",
    classes=object_classes
)
```

## 🚀 Installation
- Clone Repository

  ```
  git clone https://github.com/MouhJi/traffic-region-counter.git
  cd traffic-region-counter
  ```
- Install Dependencies

  ```
  pip install -r requirements.txt
  ```
- Run the Project

  ```
  python main.py
  ```

---

## 🎥 Output

The processed video will be saved as output.mp4

Each polygon region is drawn with semi-transparent color

The total number of vehicles per region is displayed in real-time
  
---

## 🧩 Dependencies

| Library         | Purpose                           |
| --------------- | --------------------------------- |
| `ultralytics`   | YOLO detection + tracking         |
| `supervision`   | Drawing polygons, visual overlays |
| `shapely`       | Region geometry operations        |
| `opencv-python` | Video input/output                |
| `torch`         | Deep learning backend             |

---

## 🧑‍💻 Author

Developer: Mouth Ji

Email: nguyengiap1802@gmail.com

GitHub: [https://github.com/your_username](https://github.com/MouhJi)
