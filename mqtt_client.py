#!/usr/bin/env python
import base64
import pathlib
import subprocess
import imageio
import os
import json
import requests
import paho.mqtt.client as mqtt
import io
from PIL import Image
import tempfile
import logging
from multiprocessing import Process
import yaml
from datetime import datetime
import atexit
from functools import partial
import signal


logging.basicConfig(level=logging.INFO, format="%(processName)s: %(message)s")

ongoing_tasks = {}

amblegpt_enabled = True

config = yaml.safe_load(open("config.yml", "r"))

# Define the MQTT server settings
MQTT_FRIGATE_TOPIC = "frigate/reviews"
MQTT_SUMMARY_TOPIC = "frigate/reviews/summary"
MQTT_HA_SWITCH_TOPIC = "homeassistant/switch/amblegpt"
MQTT_HA_SWITCH_CONFIG_TOPIC = MQTT_HA_SWITCH_TOPIC + "/config"
MQTT_HA_SWITCH_COMMAND_TOPIC = MQTT_HA_SWITCH_TOPIC + "/set"
MQTT_HA_SWITCH_STATE_TOPIC = MQTT_HA_SWITCH_TOPIC + "/state"
MQTT_BROKER = config["mqtt_broker"]
MQTT_PORT = config.get("mqtt_port", 1883)
MQTT_USERNAME = config.get("mqtt_username", "")
MQTT_PASSWORD = config.get("mqtt_password", "")

# Define Frigate server details for thumbnail retrieval
FRIGATE_SERVER_IP = config["frigate_server_ip"]
FRIGATE_SERVER_PORT = config.get("frigate_server_port", 5000)
THUMBNAIL_ENDPOINT = "/api/review/{}/thumbnail.jpg"
CLIP_ENDPOINT = "/api/review/{}/clip.mp4"

# Video frame sampling settings
GAP_SECS = 3

# GPT config
DEFAULT_PROMPT = """
You're a helpful assistant helping to label a video for machine learning training
You are reviewing some continuous frames of a video footage as of {EVENT_START_TIME}. Frames are {GAP_SECS} second(s) apart from each other in the chronological order.
{CAMERA_PROMPT}
Please describe what happend in the video in json format. Do not print any markdown syntax!
Answer like the following:
{{
    "num_persons" : 2,
    "persons" : [
    {{
        "height_in_meters": 1.75,
        "duration_of_stay_in_seconds": 15,
        "gender": "female",
        "age": 50
    }},
    {{
        "height_in_meters": 1.60,
        "duration_of_stay_in_seconds": 15,
        "gender": "unknown",
        "age": 36
    }},
    "summary": "SUMMARY"
    "title": "TITLE"
}}

You can guess their height and gender . It is 100 percent fine to be inaccurate.

You can measure their duration of stay given the time gap between frames.

You should take the time of event into account.
For example, if someone is trying to open the door in the middle of the night, it would be suspicious. Be sure to mention it in the SUMMARY.

Mostly importantly, be sure to mention any unusualness considering all the context.

Some example SUMMARIES are
    1. One person walked by towards right corner with her dog without paying attention towards the camera's direction.
    2. One Amazon delivery person (in blue vest) dropped off a package.
    3. A female is waiting, facing the door.
    4. Suspicious: A person is wandering without obvious purpose in the middle of the night, which seems suspicious.
    5. Suspicious: A person walked into the frame from outside, picked up a package, and left.
       The person didn't wear any uniform so this doesn't look like a routine package pickup. Be aware of potential package theft!

TITLE is a one sentence summary of the event. Use no more than 10 words.

Write your answer in {RESULT_LANGUAGE} language.
"""

PROMPT_TEMPLATE = config.get("prompt", DEFAULT_PROMPT)

RESULT_LANGUAGE = config.get("result_language", "english")

PER_CAMERA_CONFIG = config.get("per_camera_configuration", {})

VERBOSE_SUMMARY_MODE = config.get("verbose_summary_mode", True)

ADD_HA_SWITCH = config.get("add_ha_switch", False)

REVIEW_TYPES = config.get("review_types", None)

MODEL = config.get("model", "gpt-4o")


def get_camera_prompt(camera_name):
    # Retrieve custom prompt for a specific camera
    camera_config = PER_CAMERA_CONFIG.get(camera_name)
    if camera_config and "custom_prompt" in camera_config:
        return camera_config["custom_prompt"]
    return ""


def generate_prompt(gap_secs, event_start_time, camera_name):
    return PROMPT_TEMPLATE.format(
        GAP_SECS=gap_secs,
        EVENT_START_TIME=event_start_time,
        RESULT_LANGUAGE=RESULT_LANGUAGE,
        CAMERA_PROMPT=get_camera_prompt(camera_name),
    )


def get_local_time_str(ts: float):
    # convert the timestamp to a datetime object in the local timezone
    dt_object = datetime.fromtimestamp(ts)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def prompt_gpt4_with_video_frames(prompt, base64_frames, low_detail=True):
    logging.info("prompting GPT")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(
                    lambda frame: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low" if low_detail else "high",
                        },
                    },
                    base64_frames,
                ),
            ],
        },
    ]
    payload = {
        "model": MODEL,
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    return requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )


def is_ffmpeg_available():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_frames_imagio(video_path, gap_secs):
    logging.info("Extrating frames from video")
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    frames = []

    for i, frame in enumerate(reader):
        # Extract a frame every {gap_secs} seconds
        if i % (int(gap_secs * fps)) == 0:
            # Convert to PIL Image to resize
            image = Image.fromarray(frame)

            # Calculate the new size, maintaining the aspect ratio
            ratio = min(480 / image.size[0], 480 / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # Resize the image
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Cache frames locally for debug
            # Extract the video file name and create a directory for frames
            video_name = pathlib.Path(video_path).stem
            frames_dir = os.path.join("cache_video", video_name)
            os.makedirs(frames_dir, exist_ok=True)
            # Frame file name
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            # Save the frame
            resized_image.save(frame_file, "JPEG")

            # Convert back to bytes
            with io.BytesIO() as output:
                resized_image.save(output, format="JPEG")
                frame_bytes = output.getvalue()
            frames.append(base64.b64encode(frame_bytes).decode("utf-8"))
    reader.close()
    logging.info(f"Got {len(frames)} frames from video")
    return frames


def extract_frames_ffmpeg(video_path, gap_secs):
    logging.info("Extracting frames from video using FFmpeg")

    # Extract the video file name and create a directory for frames
    video_name = pathlib.Path(video_path).stem
    frames_dir = os.path.join("cache_video", video_name)
    os.makedirs(frames_dir, exist_ok=True)

    # FFmpeg command to extract frames every gap_secs seconds and resize them
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{gap_secs},scale=480:480:force_original_aspect_ratio=decrease",
        "-q:v",
        "2",  # Quality level for JPEG
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ]

    # Execute FFmpeg command
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read and encode the extracted frames
    frames = []
    for frame_file in sorted(os.listdir(frames_dir)):
        if not frame_file.endswith(".jpg"):  # to exclude .DS_Store etc
            continue
        with open(os.path.join(frames_dir, frame_file), "rb") as file:
            frame_bytes = file.read()
            frames.append(base64.b64encode(frame_bytes).decode("utf-8"))

    logging.info(f"Got {len(frames)} frames from video")
    return frames


def extract_frames(video_path, gap_secs):
    if is_ffmpeg_available():
        return extract_frames_ffmpeg(video_path, gap_secs)
    else:
        return extract_frames_imagio(video_path, gap_secs)


# Function to download video clip and extract frames
def download_video_clip_and_extract_frames(review_id, gap_secs):
    clip_url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{CLIP_ENDPOINT.format(review_id)}"
    response = requests.get(clip_url)

    if response.status_code == 200:
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        clip_filename = os.path.join(temp_dir.name, f"clip_{review_id}.mp4")

        # clip_filename = "cache_video/" + f"clip_{review_id}.mp4"

        with open(clip_filename, "wb") as f:
            f.write(response.content)
        logging.info(f"Video clip for review {review_id} saved as {clip_filename}.")

        # After downloading, extract frames
        return extract_frames(clip_filename, gap_secs)
    else:
        logging.error(
            f"Failed to retrieve video clip for review {review_id}. Status code: {response.status_code}"
        )
        return []


def process_message(payload):
    try:
        review_id = payload["after"]["id"]
        video_base64_frames = download_video_clip_and_extract_frames(
            review_id, gap_secs=GAP_SECS
        )

        if len(video_base64_frames) == 0:
            return

        local_time_str = get_local_time_str(ts=payload["after"]["start_time"])
        prompt = generate_prompt(
            GAP_SECS, local_time_str, camera_name=payload["after"]["camera"]
        )
        response = prompt_gpt4_with_video_frames(prompt, video_base64_frames)
        logging.info(f"GPT response {response.json()}")
        json_str = response.json()["choices"][0]["message"]["content"]
        result = json.loads(json_str)

        # Set the summary to the 'after' field
        payload["after"]["summary"] = "| GPT: " + (
            result["summary"] if VERBOSE_SUMMARY_MODE else result["title"]
        )

        # Convert the updated payload back to a JSON string
        updated_payload_json = json.dumps(payload)

        # Publish the updated payload back to the MQTT topic
        # Create a new MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if MQTT_USERNAME is not None:
            client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)

        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.publish(MQTT_SUMMARY_TOPIC, updated_payload_json)
        logging.info("Published updated payload with summary back to MQTT topic.")
    except Exception:
        logging.exception(f"Error processing video for event {review_id}")
    finally:
        # Cleanup: remove the task from the ongoing_tasks dict
        if review_id in ongoing_tasks:
            del ongoing_tasks[review_id]


# Define what to do when the client connects to the broker
def on_connect(client, userdata, flags, reason_code, properties):
    logging.info("Connected with reason code " + str(reason_code))
    if reason_code > 0:
        print("Connected with reason code", reason_code)
        return
    TOPICS_TO_SUBSCRIBE = [MQTT_FRIGATE_TOPIC]

    # MQTT discovery configuration
    if ADD_HA_SWITCH:
        TOPICS_TO_SUBSCRIBE.append(MQTT_HA_SWITCH_COMMAND_TOPIC)
        config_message = {
            "name": "AmbleGPT",
            "command_topic": MQTT_HA_SWITCH_COMMAND_TOPIC,
            "state_topic": MQTT_HA_SWITCH_STATE_TOPIC,
            "unique_id": "amblegptd",
            "device": {"identifiers": ["amblegpt0a"], "name": "AmbleGPT"},
        }
        client.publish(
            MQTT_HA_SWITCH_CONFIG_TOPIC,
            json.dumps(config_message),
        )
        client.publish(MQTT_HA_SWITCH_STATE_TOPIC, "ON")

    for topic in TOPICS_TO_SUBSCRIBE:
        client.subscribe(topic)
    print("Subscribed to topic:", TOPICS_TO_SUBSCRIBE)


# Define what to do when a message is received
def on_message(client, userdata, msg):
    global ongoing_tasks, amblegpt_enabled

    if msg.topic == MQTT_HA_SWITCH_COMMAND_TOPIC:
        amblegpt_enabled = msg.payload.decode("utf-8").upper() == "ON"
        logging.info(f"AmbleGPT enabled: {amblegpt_enabled}")
        client.publish(MQTT_HA_SWITCH_STATE_TOPIC, "ON" if amblegpt_enabled else "OFF")
        return
    if msg.topic != MQTT_FRIGATE_TOPIC:
        return
    if not amblegpt_enabled:
        logging.info(f"Ignored Frigate review because AmbleGPT is disabled")
        return
    # Parse the message payload as JSON
    review_id = None
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if REVIEW_TYPES is not None and payload["after"]["label"] not in REVIEW_TYPES:
            # Skip if the event type is not in the list of allowed event types
            logging.info(f"Skipping review because of review type {payload['after']['label']}")
            return
        if "summary" in payload["after"] and payload["after"]["summary"]:
            # Skip if this message has already been processed. To prevent echo loops
            logging.info("Skipping message that has already been processed")
            return
        if (
            payload["before"].get("snapshot_time")
            == payload["after"].get("snapshot_time")
            and (payload["type"] != "end")
            and (review_id in ongoing_tasks)
        ):
            # Skip if this snapshot has already been processed
            logging.info(
                "Skipping because the message with this snapshot is already (being) processed"
            )
            return
        if not payload["after"]["has_clip"]:
            # Skip if this snapshot has already been processed
            logging.info("Skipping because of no available video clip yet")
            return
        review_id = payload["after"]["id"]
        logging.info(f"Review ID: {review_id}")

        # If there's an ongoing task for the same event, terminate it
        if review_id in ongoing_tasks:
            ongoing_tasks[review_id].terminate()
            ongoing_tasks[review_id].join()  # Wait for process to terminate
            logging.info(f"Terminated ongoing task for event {review_id}")

        # Start a new task for the new message
        processing_task = Process(target=process_message, args=(payload,))
        processing_task.start()
        ongoing_tasks[review_id] = processing_task

    except json.JSONDecodeError:
        logging.exception("Error decoding JSON")
    except KeyError:
        logging.exception("Key not found in JSON payload")


def cleanup(client):
    logging.info("Exiting. Cleaning up")
    client.publish(MQTT_HA_SWITCH_CONFIG_TOPIC, "")
    client.disconnect()

def handle_sigterm(client, signum, frame):
    cleanup(client)
    exit(0)


if __name__ == "__main__":
    # Create a client instance
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    # Assign event callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    if MQTT_USERNAME is not None:
        client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)
    # Connect to the broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    atexit.register(lambda: cleanup(client))
    signal.signal(signal.SIGTERM, partial(handle_sigterm, client))

    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()
