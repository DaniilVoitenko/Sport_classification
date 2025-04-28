
!pip install yt-dlp scipy opencv-python tensorflow matplotlib
!apt-get install -y ffmpeg


import os
import zipfile
import yt_dlp
import cv2
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

 
zip_path = "/content/CPSM_GT.zip"
extract_path = "/content/CPSM_GT"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extracted to:", extract_path)

 
mat = scipy.io.loadmat(os.path.join(extract_path, "pp_CPSM_Shared.mat"))
data = mat['D'][0]  # Contains 487 entries
print("Example entry keys:\n", data[0].dtype) 
vid_data = data['vidData']
print("vidData shape:", vid_data.shape)
print("Type of first element:", type(vid_data[0]))
print("First entry shape (if it's an array):", getattr(vid_data[0], 'shape', 'not an array'))


videos = []
for entry in data:
    url = str(entry['url'][0]) if entry['url'].size > 0 else ''
    label = str(entry['label'][0]) if entry['label'].size > 0 else ''
    if url.startswith("http") and label:
        videos.append((url, label))

# Save sample video links
os.makedirs("/content/links", exist_ok=True)
video_list_path = "/content/links/video_links.txt"
with open(video_list_path, "w") as f:
    for url, label in videos[:20]:  # Limit to 20 for faster training
        f.write(f"{url},{label}\n")

# --- STEP 5: Download videos ---
def download_videos(link_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(link_file, 'r') as f:
        links = [line.strip().split(',') for line in f if line.strip()]
    ydl_opts = {
        'quiet': True,
        'format': 'mp4',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url, label in links:
            try:
                ydl.download([url])
            except Exception as e:
                print(f"Failed to download {url}: {e}")

 
def extract_frames(video_path, label, output_root, max_frames=100):
    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (64, 64))
        frame_path = os.path.join(out_dir, f"{os.path.basename(video_path)}_f{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()

def process_downloaded_videos(video_dir, link_file, frame_output_dir):
    with open(link_file, 'r') as f:
        links = [line.strip().split(',') for line in f if line.strip()]
    for file in os.listdir(video_dir):
        if not file.endswith('.mp4'):
            continue
        video_id = os.path.splitext(file)[0]
        label = next((lbl for url, lbl in links if video_id in url), None)
        if label:
            extract_frames(os.path.join(video_dir, file), label, frame_output_dir)

 
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(frame_dir, epochs=75):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
    train_gen = datagen.flow_from_directory(
        frame_dir, target_size=(64, 64), batch_size=32,
        color_mode='grayscale', class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(
        frame_dir, target_size=(64, 64), batch_size=32,
        color_mode='grayscale', class_mode='categorical', subset='validation')
    
    model = build_cnn((64, 64, 1), train_gen.num_classes)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Loss")
    plt.legend()
    plt.show()

    return model

 
video_output_dir = "/content/videos"
frame_output_dir = "/content/frames"

print("Downloading videos...")
download_videos(video_list_path, video_output_dir)

print("Extracting frames...")
process_downloaded_videos(video_output_dir, video_list_path, frame_output_dir)

print("Training model...")
model = train_model(frame_output_dir, epochs=75)


model.save("/content/cpsm_sports_model.h5")
print(" Model saved to /content/cpsm_sports_model.h5")
