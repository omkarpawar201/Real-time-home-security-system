# Real-Time Home Security System

## Overview

This project is a real-time home security system using OpenCV, Arduino, and Telegram. The system detects faces using a pre-trained Haar Cascade model and recognizes them using LBPH face recognizer. If an unrecognized face is detected, the system sends an alert message and photo to a Telegram chat. The system can also control a relay connected to an Arduino board, which can be operated via Telegram messages or keyboard inputs.

## Features

- Real-time face detection and recognition
- Attendance marking for recognized faces
- Relay control via Arduino
- Telegram notifications for unrecognized faces
- Manual relay control via keyboard
- Threading for continuous Telegram message checking

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Arduino board (e.g., Arduino Uno)
- Webcam
- Libraries:
  - OpenCV
  - NumPy
  - Requests
  - PyFirmata
  - Keyboard

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/omkarpawar201/Real-time-home-security-system.git
    cd real-time-home-security-system
    ```

2. **Install the required libraries:**
    ```sh
    pip install opencv-python numpy requests pyfirmata keyboard
    ```

3. **Upload Arduino code:**
    - Connect your Arduino board to your computer.
    - Open the Arduino IDE and upload the code to set up the relay pin:


4. **Prepare the datasets:**
    - Create a folder named `datasets` and add subfolders for each person.
    - Add images of each person in their respective subfolders.

## Usage

1. **Run the Python script:**
    ```sh
    python reco.py
    ```

2. **Telegram Bot Setup:**
    - Create a Telegram bot and obtain the bot token.
    - Replace `telegram_bot_token` and `telegram_chat_id` in the script with your bot token and chat ID.

3. **Face Recognition and Relay Control:**
    - The system will start recognizing faces and controlling the relay.
    - If an unrecognized face is detected, an alert message and photo will be sent to the Telegram chat.
    - You can manually control the relay via Telegram messages (`Allow` to open, `Deny` to close) or keyboard inputs (`o` to open, `c` to close).


