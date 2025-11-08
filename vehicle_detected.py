import cv2
import torch
import numpy as np
import supervision as sv
from shapely.geometry import Point, Polygon
from ultralytics.solutions.solutions import BaseSolution, SolutionResults, SolutionAnnotator

# Color definitions for monitored regions and vehicles
region_colors = [
    (255, 0, 255),
    (0, 255, 255),
    (86, 0, 254),
    (0, 128, 255),
    (235, 34, 134)
]
vehicle_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 128)
]


class CounterObject(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Retrieve region definitions from configuration
        self.regions = self.CFG.get("region", [])

        # Initialize in/out counters and sets to store counted IDs
        self.in_counts = [0] * len(self.regions)
        self.out_counts = [0] * len(self.regions)
        self.counted_ids = [set() for _ in range(len(self.regions))]

        # Control flags and display settings
        self.region_initialized = False
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

    def initialize_region_geometry(self):
        # Initialize geometric functions from Shapely
        self.Point = Point
        self.Polygon = Polygon

    def count_object_in_region(self, region_idx, region_points, current_centroid, track_id, prev_position):
        """
        Determine if a tracked object has entered a region.
        If yes, increment the region counter and store the object's track ID.
        """

        # Skip if previous position is missing or object was already counted in this region
        if prev_position is None or track_id in self.counted_ids[region_idx]:
            return

        # Create a polygon for the region and check if centroid is inside it
        polygon = self.Polygon(region_points)
        if polygon.contains(self.Point(current_centroid)):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            region_width = max(xs) - min(xs)
            region_height = max(ys) - min(ys)

            # Determine movement direction (entering vs exiting)
            going_in = False
            if region_width < region_height and current_centroid[0] > prev_position[0]:
                going_in = True
            elif region_width >= region_height and current_centroid[1] > prev_position[1]:
                going_in = True

            # Increment corresponding counter
            if going_in:
                self.in_counts[region_idx] += 1
            else:
                self.out_counts[region_idx] += 1

            # Store this ID to avoid double counting
            self.counted_ids[region_idx].add(track_id)

    def display_counts(self, plot_im):
        """
        Display total counts for each region on the video frame.
        """
        for i, region_points in enumerate(self.regions):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))

            # Text to display total count in the region
            text_str = f"{self.in_counts[i] + self.out_counts[i]}"
            cv2.putText(plot_im, text_str, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    def process(self, frame):
        """
        Main per-frame processing method:
        1. Extract YOLO detections and tracking info.
        2. Draw regions and bounding boxes.
        3. Count objects inside regions.
        4. Display results.
        """
        # Initialize region geometry once
        if not self.region_initialized:
            self.initialize_region_geometry()
            self.region_initialized = True

        # Perform object detection + tracking using Ultralytics base method
        self.extract_tracks(frame)

        # Annotator object used for drawing bounding boxes and labels
        self.annotator = SolutionAnnotator(frame, line_width=self.line_width)

        # Draw all predefined monitoring regions
        for idx, region_points in enumerate(self.regions):
            color = region_colors[idx % len(region_colors)]
            self.annotator.draw_region(reg_pts=region_points, color=color, thickness=self.line_width * 2)
            b, g, r = color
            frame = sv.draw_filled_polygon(
                scene=frame,
                polygon=np.array(region_points),
                color=sv.Color(r=r, g=g, b=b),
                opacity=0.25
            )

        # Loop over all detected objects (bounding boxes, track IDs, and class labels)
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            cls = int(cls)  # Convert YOLO float class index to integer

            # Choose color based on vehicle type (class)
            color = vehicle_colors[cls % len(vehicle_colors)]

            # Draw bounding box with label (class name)
            self.annotator.box_label(box, label=self.names[cls], color=color)

            # Update tracking history for the object
            self.store_tracking_history(track_id, box)

            # Calculate current bounding box centroid (center point)
            current_centroid = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            )

            # Retrieve previous centroid position for direction calculation
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]

            # For each region, check if the current object is inside it
            for r_idx, region_points in enumerate(self.regions):
                self.count_object_in_region(
                    region_idx=r_idx,
                    region_points=region_points,
                    current_centroid=current_centroid,
                    track_id=track_id,
                    prev_position=prev_position
                )

        # Render annotations and display total counts
        plot_im = self.annotator.result()
        self.display_counts(plot_im)

        # Display frame (if enabled)
        self.display_output(plot_im)

        # Return processed frame and number of tracked objects
        return SolutionResults(
            plot_im=plot_im,
            total_track=len(self.track_ids)
        )


if __name__ == '__main__':
    # Object class IDs: car=2, truck=5, bus=7 (COCO dataset IDs)
    object_classes = [2, 5, 7]

    # Open input video file
    cap = cv2.VideoCapture("vehicle_video.mp4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve video metadata
    w, h, fps = (int(cap.get(x)) for x in
                 (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Define multiple polygonal regions of interest (ROI) for counting
    region_points = [
        [[1805, 455], [1805, 363], [1526, 302], [1400, 336]],
        [[1015, 949], [915, 528], [918, 380], [1007, 382], [1385, 954]],
        [[661, 949], [755, 542], [833, 385], [918, 387], [915, 532], [1019, 957]],
        [[279, 948], [577, 548], [736, 385], [831, 387], [757, 546], [663, 951]],
    ]

    # Initialize counter solution with YOLO model and defined regions
    counter = CounterObject(
        show=True,
        region=region_points,
        model="yolo11n.pt",
        classes=object_classes
    )

    # Main processing loop: read video frame-by-frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run the counter on each frame
        results = counter(frame)
        frame = results.plot_im

        # Write annotated frame to output video
        video_writer.write(frame)

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
