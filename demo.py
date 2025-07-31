import cv2
import os

# Path to the folder containing images
image_folder = './default/samples'
video_path = 'ddpm_sampling_video.mp4'

# Sort images numerically
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")],
                key=lambda x: int(x.split("_")[1].split(".")[0]) , reverse=True)

# Read the first image to get frame size
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
size = (width, height)

# Create VideoWriter object
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'),15, size)

for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print("Video saved to:", video_path)
