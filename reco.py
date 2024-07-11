import cv2
import numpy as np
import os
from datetime import datetime
import requests
import time
from threading import Thread
from pyfirmata import Arduino, util
import keyboard

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Lights...')
print("*" * 50)

# Connect to Arduino board
board = Arduino("COM5")  # Adjust the port based on your system

# Define pin for the relay
relay_pin = 7  # Adjust the pin based on your Arduino setup
board.digital[relay_pin].mode = 1  # Set the pin as OUTPUT

def control_relay(value):
    board.digital[relay_pin].write(value)

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
(width, height) = (130, 100)  # Define width and height here
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (width, height))  # Resize images to a consistent size
            images.append(img)
            labels.append(int(label))
        id += 1

def markattendance(name):
    cwd = os.getcwd()
    path = os.path.join(cwd, 'Attendance.csv')
    if not os.path.isdir(path):
        with open("Attendance.csv", "w+"):
            pass
    with open('Attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            tmstring = now.strftime('%H:%M:%S   %d-%m-%y')
            f.writelines(f'\n{name},"present at",{tmstring}')



(images, labels) = [np.array(lst) for lst in [images, labels]]

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.write("trainer.yml")
model.read("trainer.yml")

# Use Recognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
# video_path = 'salman.mp4'  # Replace with the path to your video file
# webcam = cv2.VideoCapture(video_path)
webcam = cv2.VideoCapture(0)

# Set the time interval (in seconds) between consecutive email notifications
notification_interval = 300  # 5 minutes

# Initialize a dictionary to keep track of the last detection time for each registered person
last_detection_time = {name: 0 for name in names}

update_offset = 0
unrecognized_person_timer = 0
unrecognized_person_cooldown = 300  # Set the interval in seconds (e.g., 300 seconds = 5 minutes)

def send_unrecognized_person_message(photo_path):
    global unrecognized_person_timer
    current_time = time.time()

    if current_time - unrecognized_person_timer >= unrecognized_person_cooldown:
        unrecognized_person_timer = current_time

        # Send a message with the image to Telegram
        telegram_message = "Unrecognized person detected!"
        send_telegram_message(telegram_message, photo_path)

def send_telegram_message(message, photo_path=None):
    telegram_api_url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
    params = {'chat_id': telegram_chat_id, 'text': message}

    try:
        response = requests.post(telegram_api_url, params=params)
        if response.status_code == 200:
            print("Telegram message sent successfully!")
        else:
            print("Error sending Telegram message. Status code:", response.status_code)
    except Exception as e:
        print("Error sending Telegram message:", str(e))

    if photo_path:
        send_telegram_photo(photo_path)

def send_telegram_photo(photo_path):
    telegram_api_url = f'https://api.telegram.org/bot{telegram_bot_token}/sendPhoto'
    files = {'photo': open(photo_path, 'rb')}
    params = {'chat_id': telegram_chat_id}

    try:
        response = requests.post(telegram_api_url, params=params, files=files)
        if response.status_code == 200:
            print("Telegram photo sent successfully!")
        else:
            print("Error sending Telegram photo. Status code:", response.status_code)
    except Exception as e:
        print("Error sending Telegram photo:", str(e))

def check_telegram_messages(update_offset):
    while True:
        try:
            response = requests.get(
                f'https://api.telegram.org/bot{telegram_bot_token}/getUpdates?offset={update_offset + 1}'
            )
            if response.status_code == 200:
                updates = response.json().get('result', [])
                for update in updates:
                    update_id = update.get('update_id')
                    message = update.get('message', {}).get('text')
                    if message:
                        print(f"Received message: {message}")
                        if message == "Allow":
                            control_relay(0)
                        else:
                            control_relay(1)
                        # Add your logic to handle the 'Allow' or 'Deny' response here
                if updates:
                    update_offset = max(update_id for update in updates)
            time.sleep(2)  # Adjust the sleep time based on your needs
        except Exception as e:
            print("Error checking Telegram messages:", str(e))
            time.sleep(10)


# Set your Telegram bot token and chat ID
telegram_bot_token = '6885852215:AAHqdS64iv4ss77EMcHJzGUt6sI6wqxijbA'
telegram_chat_id = '1576022880'


# Start a thread to check for incoming Telegram messages
telegram_thread = Thread(target=check_telegram_messages, args=(update_offset,))
telegram_thread.start()

def get_location(latitude, longitude):
    # Use OpenCage Geocoding API to get a detailed address from latitude and longitude
    try:
        response = requests.get(
            f"https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key=YOUR_OPENCAGE_API_KEY"
        )
        data = response.json()
        if data.get("results"):
            result = data["results"][0]
            formatted_address = result.get("formatted")
            return formatted_address
    except Exception as e:
        print("Error getting location:", str(e))
    return "Unknown Location"

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    if len(faces)==0:

        if keyboard.is_pressed('o'):
            control_relay(0)
        elif keyboard.is_pressed('c'):
            control_relay(1)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 100:
            detected_person = names.get(prediction[0], None)
            if detected_person is not None:
                cv2.putText(im, '% s - %.0f' %
                            (detected_person, prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                markattendance(detected_person)

                control_relay(0)

                current_time = time.time()
        else:
            control_relay(1)
            cv2.putText(im, 'not recognized', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Save the image of the unrecognized person
            unrecognized_face_path = "unrecognized_face.jpg"
            cv2.imwrite(unrecognized_face_path, im)

            # Send a message with the image to Telegram (controlled by the cooldown)
            send_unrecognized_person_message(unrecognized_face_path)


    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        control_relay(0)
        break

webcam.release()
cv2.destroyAllWindows()
