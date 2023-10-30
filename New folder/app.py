from flask import Flask, request, render_template, send_file
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image, ImageEnhance
from rembg import remove, new_session
import os
import psutil
import time

app = Flask(__name__)

# Initialize the REMBG session
session = new_session(model_name='u2net_human_seg')




def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size (RSS) in bytes





def sharpen_img(image):
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)

    # Calculate the unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.3, blurred, -0.7, 0)

    return unsharp_mask

def crop_head_and_shoulders(image_path, new_height):
    img = cv2.imread(image_path)

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(img)

    if results:
        # Extract the bounding box from the first detected face
        x_f, y_f, w_f, h_f = results[0]['box']

        # Calculate the cropping dimensions based on face size
        face_size = max(w_f, h_f)
        width_crop = int(1 * face_size)  # Adjust this proportion based on your preference
        height_crop = int(1 * face_size)  # Adjust this proportion based on your preference

        # Calculate the center of the detected face
        center_x = x_f + w_f // 2
        center_y = y_f + h_f // 2

        # Calculate the cropping coordinates
        x1 = max(center_x - width_crop, 0)
        x2 = min(center_x + width_crop, img.shape[1])
        y1 = max(center_y - height_crop, 0)
        y2 = min(center_y + height_crop, img.shape[0])

        # Crop the head and shoulders region
        img = img[y1:y2, x1:x2]

        # Resize to maintain the aspect ratio with the user-specified height
        img = cv2.resize(img, (int(new_height * img.shape[1] / img.shape[0]), new_height), interpolation=cv2.INTER_AREA)

        image = sharpen_img(img)

        return image

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Record the start time
        start_time = time.time()

        # Check if a file is uploaded
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        bg_color = tuple(map(int, request.form['bg_color'].split(',')))
        new_height = int(request.form['new_height'])

        # Check if the file is empty
        if file.filename == "":
            return "No selected file"

        if file:
            # Generate a unique filename based on the current timestamp
            #timestamp = time.datetime.now().strftime("%Y%m%d%H%M%S")
            uploaded_file_path = "static/image.jpg"
            file.save(uploaded_file_path)

            # Record the start time for image processing
            processing_start_time = time.time()

            # Perform image processing
            cropped_image = crop_head_and_shoulders(uploaded_file_path, new_height)
            output = remove(cropped_image, session=session, alpha_matting=True,
                            alpha_matting_foreground_threshold=255, alpha_matting_background_threshold=255,
                            alpha_matting_erode_size=20, bgcolor=bg_color)

            # Record the end time for image processing
            processing_end_time = time.time()

            # Calculate the elapsed time for image processing
            processing_time = processing_end_time - processing_start_time

            # Save the processed image
            output_path = "static/passport.jpg"
            cv2.imwrite(output_path, output)

            # Record the end time
            end_time = time.time()

            # Calculate the total elapsed time
            total_time = end_time - start_time

            print(f"Image processing time: {processing_time:.2f} seconds")
            print(f"Total time: {total_time:.2f} seconds")

            return render_template("result.html", image_path=output_path)

    return render_template("upload.html")


@app.route("/download")
def download_image():
    image_path = "static/passport.jpg"
    org_image_path = 'uploads/image.jpg'
    return send_file(image_path, as_attachment=True)

if __name__ == "__main__":

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True)

    memory_used = memory_usage()
    print(f"Memory used by the program: {memory_used} bytes")
