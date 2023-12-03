#!/usr/bin/env python
import base64
import pathlib
import subprocess
from json import JSONDecodeError

import imageio
import os
import json
import requests
import paho.mqtt.client as mqtt
import io
from PIL import Image
import tempfile
import logging
import yaml
from datetime import datetime, timedelta
import time
from cachetools import TTLCache
from string import Template
import threading
import queue

logging.basicConfig(level=logging.INFO, format="%(levelname)-s %(asctime)s %(processName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
config = yaml.safe_load(open("config.yml", "r"))

# Define the MQTT server settings
MQTT_FRIGATE_TOPIC = "frigate/events"
MQTT_SUMMARY_TOPIC = "frigate/events/summary"
MQTT_BROKER = config["mqtt_broker"]
MQTT_PORT = config.get("mqtt_port", 1883)
MQTT_USERNAME = config.get("mqtt_username", "")
MQTT_PASSWORD = config.get("mqtt_password", "")

# Define Frigate server details for thumbnail retrieval
FRIGATE_SERVER_IP = config["frigate_server_ip"]
FRIGATE_SERVER_PORT = config.get("frigate_server_port", 5000)
EVENTS_ENDPOINT = "/api/events/{}"
CLIP_ENDPOINT = "/api/events/{}/clip.mp4"
SNAPSHOT_ENDPOINT = "/api/events/{}/snapshot.jpg?quality=100&crop=0&bbox=0&timestamp=0&h={}"
SUB_LABEL_ENDPOINT = "/api/events/{}/sub_label"
VALID_LABELS = config.get("objects", ["person"])

# Video frame sampling settings
GAP_SECS = config.get("internal_between_frames", 3)
FRAME_RESIZE = config.get("resized_frame_size", 512)
MAX_RESPONSE_TOKEN = config.get("max_response_token", 500)
MAX_FRAMES_TO_PROCESS = config.get("max_frames_to_process", 10)
MIN_FRAMES_TO_PROCESS = config.get("min_frames_to_process", 2)
LOW_DETAIL = config.get("low_detail", True)
PROCESS_CLIP = config.get("process_clip", True)
PROCESS_SNAPSHOT = config.get("process_snapshot", False if PROCESS_CLIP else True)
UPDATE_FRIGATE_SUBLABEL = config.get("update_frigate_sublabel", False)
PROCESS_WHEN_COMPLETE = config.get("process_when_complete", False)
WAIT_FOR_EVENT_TIME = config.get("event_wait_time", GAP_SECS * 3 if PROCESS_CLIP else GAP_SECS)
CAMERA_DEBOUNCE_TIME = config.get("camera_retrigger_time", 30)

RUNNING_TASKS = []
outgoing_message_queue = queue.Queue(maxsize=10)
ongoing_tasks = TTLCache(maxsize=1000, ttl=timedelta(hours=4), timer=datetime.now)
recent_triggered_cameras = TTLCache(maxsize=20, ttl=CAMERA_DEBOUNCE_TIME)


SYSTEM_PROMPT = """
You are an assistant helping to label videos from security cameras for machine learning training.
Never include triple backticks in your response.
Never include ```json in your response.
You only returns replies with valid, iterable RFC8259 compliant JSON in your response.
"""

# GPT config
DEFAULT_PROMPT = """
You are reviewing some continuous frames of video footage from a camera as of $EVENT_START_TIME. 
Frames are $GAP_SECS second(s) apart from each other in chronological order.

Information about the camera:
$CAMERA_PROMPT

Please describe what happened in the video in json format. Do not print any markdown syntax!
Please respond with the following JSON:
{
    "num_persons" : 2,
    "persons" : [
    {
        "height_in_meters": 1.75,
        "duration_of_stay_in_seconds": 15,
        "gender": "female",
        "age": 50
    },
    {
        "height_in_meters": 1.60,
        "duration_of_stay_in_seconds": 15,
        "gender": "unknown",
        "age": 36
    },
    "summary": "SUMMARY",
    "title": "TITLE"
}

Even if you are not certain you need to make a guess for their height and gender. It is 100 percent fine to be inaccurate.
You can measure their duration of stay given the time gap between frames.

You should take the time of event into account.
For example, if someone is trying to open the door in the middle of the night, it would be suspicious. Be sure to mention it in the SUMMARY.
Mostly importantly, be sure to mention any unusual activity considering all the context.

Some example SUMMARIES are
    1. One person walked by towards right corner with her dog without paying attention towards the camera's direction.
    2. One Amazon delivery person (in blue vest) dropped off a package.
    3. A female is waiting, facing the door.
    4. Suspicious: A person is wandering without obvious purpose in the middle of the night, which seems suspicious.
    5. Suspicious: A person walked into the frame from outside, picked up a package, and left.
       The person didn't wear any uniform so this doesn't look like a routine package pickup. Be aware of potential package theft!

TITLE is a one sentence summary of the event. Use no more than 100 characters.
Write your answer in $RESULT_LANGUAGE language. Never include triple backticks in your response.
"""


PROMPT_TEMPLATE = config.get("prompt", DEFAULT_PROMPT)

RESULT_LANGUAGE = config.get("result_language", "english")

PER_CAMERA_CONFIG = config.get("per_camera_configuration", {})

VERBOSE_SUMMARY_MODE = config.get("verbose_summary_mode", True)

C

def get_camera_prompt(camera_name):
    # Retrieve custom prompt for a specific camera
    camera_config = PER_CAMERA_CONFIG.get(camera_name)
    if camera_config and "custom_prompt" in camera_config:
        return camera_config["custom_prompt"]
    return ""


def generate_prompt(gap_secs, event_start_time, camera_name):
    templated_prompt = Template(PROMPT_TEMPLATE)
    return templated_prompt.substitute(
        RESULT_LANGUAGE=RESULT_LANGUAGE,
        GAP_SECS=gap_secs,
        EVENT_START_TIME=event_start_time,
        CAMERA_PROMPT=get_camera_prompt(camera_name)
    )


def get_local_time_str(ts: float):
    # convert the timestamp to a datetime object in the local timezone
    dt_object = datetime.fromtimestamp(ts)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def prompt_gpt4_with_video_frames(prompt, base64_frames, low_detail=True):
    logging.info("prompting GPT-4v")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }
    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
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
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": MAX_RESPONSE_TOKEN,
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
    logging.info("Extracting frames from video")
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
    with tempfile.TemporaryDirectory(prefix=video_name, suffix="-cache") as frames_dir:
        logging.info(f"Using temporary clip directory {frames_dir}")
        if not LOW_DETAIL or FRAME_RESIZE < 512:
            resize = FRAME_RESIZE
        else:
            resize = 512
        # FFmpeg command to extract frames every gap_secs seconds and resize them
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{gap_secs},scale={resize}:{resize}:force_original_aspect_ratio=decrease",
            "-q:v",
            "2",  # Quality level for JPEG
            os.path.join(frames_dir, "frame_%04d.jpg"),
        ]

        # Execute FFmpeg command
        subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read and encode the extracted frames
        frames = []
        for frame_file in sorted(os.listdir(frames_dir)):
            while len(frames) < MAX_FRAMES_TO_PROCESS - 1:
                with open(os.path.join(frames_dir, frame_file), "rb") as file:
                    frame_bytes = file.read()
                    frames.append(base64.b64encode(frame_bytes).decode("utf-8"))
            continue

        logging.info(f"Got {len(frames)} frames from video")
    return frames


def extract_frames(video_path, gap_secs):
    if is_ffmpeg_available():
        return extract_frames_ffmpeg(video_path, gap_secs)
    else:
        return extract_frames_imagio(video_path, gap_secs)


def download_snapshot_and_combine_frames(event_id: str, gap_secs: int) -> list:
    """

    :param event_id:
    :param gap_secs:
    :return:
    """
    snapshot_url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{SNAPSHOT_ENDPOINT.format(event_id, FRAME_RESIZE)}"

    frame_count = 0
    retry_attempts = 0
    frame_list = []
    start_time = time.monotonic()
    while frame_count < MAX_FRAMES_TO_PROCESS:
        res = requests.get(snapshot_url)
        if res.status_code == 200:
            frame = base64.b64encode(res.content).decode("utf-8")
            frame_list.append(frame)
            logging.info(f"Snapshot successfully Downloaded: frame_{frame_count}.jpg for event {event_id}")
            frame_count += 1
        else:
            retry_attempts += 1
            if retry_attempts > 5:
                logging.info("Image Could not be retrieved")
                return []
        time.sleep(float(gap_secs) - ((time.monotonic() - start_time) % float(gap_secs)))
    return frame_list


def download_video_clip_and_extract_frames(event_id: str, gap_secs: int) -> list:
    f"""
    download video clip for event id and extract frames every gap_secs
    :param event_id: frigate event id to fetch
    :param gap_secs: period in seconds to wait between still images to extract from clip
    :return: list of extracted
    """
    clip_url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{CLIP_ENDPOINT.format(event_id)}"
    event_data = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{EVENTS_ENDPOINT.format(event_id)}"

    event_response = requests.get(event_data).json()
    if not event_response['has_clip']:
        logging.error(f"Video clip for event {event_id}, is not yet ready.")
        return []
    response = requests.get(clip_url)

    if response.status_code == 200:
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        clip_filename = os.path.join(temp_dir.name, f"clip_{event_id}.mp4")
        logging.info(f"Creating temporary directory for video clip: {clip_filename}")

        with open(clip_filename, "wb") as f:
            f.write(response.content)
        logging.info(f"Video clip for event {event_id} saved as {clip_filename}.")

        # After downloading, extract frames
        return extract_frames(clip_filename, gap_secs)
    else:
        logging.error(
            f"Failed to retrieve video clip for event {event_id}. Status code: {response.status_code}"
        )
        return []


def process_message(payload):
    global ongoing_tasks
    try:
        event_id = payload["after"]["id"]

        # If the event is complete, we have no reason to wait
        if not PROCESS_WHEN_COMPLETE:
            time.sleep(WAIT_FOR_EVENT_TIME)

        if PROCESS_SNAPSHOT:
            video_base64_frames = download_snapshot_and_combine_frames(
                event_id, gap_secs=GAP_SECS
            )
        elif PROCESS_CLIP:
            video_base64_frames = download_snapshot_and_combine_frames(
                event_id, gap_secs=GAP_SECS
            )
        else:
            raise RuntimeError("Error, both PROCESS_SNAPSHOT and PROCESS_CLIP cannot be disable")

        if len(video_base64_frames) < MIN_FRAMES_TO_PROCESS:
            if event_id in ongoing_tasks:
                logging.info(f"Canceling message processing, clip is still too short: {event_id}")
                with threading.Lock():
                    ongoing_tasks.pop(event_id)
                if event_id in ongoing_tasks:
                    raise Exception("Failed to remove event_id from cache")
            return

        local_time_str = get_local_time_str(ts=payload["after"]["start_time"])
        prompt = generate_prompt(
            GAP_SECS, local_time_str, camera_name=payload["after"]["camera"]
        )

        response = prompt_gpt4_with_video_frames(prompt, video_base64_frames, LOW_DETAIL)
        logging.info(f"GPT response {response.json()}")
        json_str = response.json()["choices"][0]["message"]["content"]
        result = json.loads(json_str)

        # Set the summary to the 'after' field
        payload["after"]["summary"] = (result["summary"] if VERBOSE_SUMMARY_MODE else result["title"])
        if VERBOSE_SUMMARY_MODE:
            payload["after"]["summary_title"] = result["title"]
        if result["num_persons"] > 0:
            payload["after"]["summary_details"] = result["persons"]


        if VERBOSE_SUMMARY_MODE and UPDATE_FRIGATE_SUBLABEL:
            url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{SUB_LABEL_ENDPOINT.format(event_id)}"
            request = requests.post(url, json={"subLabel": f"{result['title']}"})
            if request.status_code == 200:
                logging.debug(f"Frigate sublabel updated with summary for event: {event_id}")
            else:
                logging.error(f"Failed updating Frigate sublabel for event: {event_id}")
        # Convert the updated payload back to a JSON string and put in the outgoing queue
        updated_payload_json = json.dumps(payload)
        outgoing_message_queue.put(updated_payload_json)
    except JSONDecodeError as ex:
        logging.error(f"Exception: {ex}")
        logging.error(f"Json response from GPT was received with incorrect structure.")
    except Exception as ex:
        logging.exception(f"Error processing video for event {event_id}")
        logging.error(f"Exception: {ex}")


# Define what to do when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code " + str(rc))
    if rc > 0:
        logging.debug(f"Connected with result code {str(rc)}")
        return
    # Subscribe to the topic
    client.subscribe(MQTT_FRIGATE_TOPIC)
    logging.info(f"Subscribed to topic: {MQTT_FRIGATE_TOPIC}")


def on_connect_writer(client, userdata, flags, rc):
    logging.info(f"Connected with result code " + str(rc))
    if rc > 0:
        logging.debug(f"Connected with result code {str(rc)}")
        return
    # Subscribe to the topic
    client.subscribe(MQTT_SUMMARY_TOPIC)
    logging.info(f"Subscribed to topic: {MQTT_SUMMARY_TOPIC}")

def on_publish(client, userdata, result):
    logging.info(f"on_publish, result: {result}")


def on_message(client, userdata, msg):
    """
    on_message mqtt callback
    :param client:
    :param userdata:
    :param msg:
    :return:
    """
    global ongoing_tasks
    event_id = None
    try:
        # Parse the message payload as JSON
        payload = json.loads(msg.payload.decode("utf-8"))
        event_id = payload["before"]["id"]
        logging.debug(f"Received MQTT message for event: {event_id}")

        if event_id in ongoing_tasks:
            logging.debug(f"Not processing running event: {event_id}")
            return

        if PROCESS_WHEN_COMPLETE and payload["type"] != "end":
            logging.debug(f"Configured to only process event when it ends, event is still running")
            return

        if "summary" in payload["after"] and payload["after"]["summary"]:
            # Skip if this message has already been processed. To prevent echo loops
            logging.debug("Skipping message that has already been processed")
            return
        if (
                payload["before"].get("snapshot_time")
                == payload["after"].get("snapshot_time")
                and (payload["type"] != "end")
                and (event_id in ongoing_tasks)
        ):
            # Skip if this snapshot has already been processed
            logging.debug(
                "Skipping because the message with this snapshot is already (being) processed"
            )
            return

        if not payload["after"]["has_clip"] and PROCESS_CLIP:
            # Skip if this snapshot has already been processed
            logging.debug("Skipping because of no available video clip yet, and configured for processing stream")
            return

        if payload["before"]["stationary"] or payload["after"]["stationary"]:
            logging.debug("Skipping event for stationary object")
            return

        if payload["before"]["label"] not in VALID_LABELS or payload["after"]["label"] not in VALID_LABELS:
            logging.debug(f"Skipping event for object not in {*VALID_LABELS,}")
            return

        if payload["before"]["camera"] in recent_triggered_cameras:
            with threading.Lock():
                ongoing_tasks[event_id] = False
                return


        # Start a new thread for the new message
        with threading.Lock():
            ongoing_tasks[event_id] = threading.Thread(target=process_message, args=(payload,), name=payload["before"]["camera"])
            recent_triggered_cameras[payload["before"]["camera"]] = True
        ongoing_tasks[event_id].start()

    except json.JSONDecodeError:
        logging.exception("Error decoding JSON")
    except KeyError:
        logging.exception("Key not found in JSON payload")


def publish_message():
    publish_client = mqtt.Client("amblegpt_publish_client")
    if MQTT_USERNAME is not None:
        publish_client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)
    logging.debug("Opening connection for mqtt publish client")
    publish_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    logging.debug(f"mqtt publish client is connected: {publish_client.is_connected()}")
    publish_client.loop_start()
    while True:
        message = outgoing_message_queue.get()
        publish_result = publish_client.publish(MQTT_SUMMARY_TOPIC, message, qos=1)
        publish_result.wait_for_publish()
        logging.info(f"Publishing updated payload with summary back to MQTT topic. "
                     f"Published: {publish_result.is_published()}")


def main():
    logging.info(f"ffmpeg for video processing is enabled: {is_ffmpeg_available()}")

    mqtt_publish_thread = threading.Thread(target=publish_message, daemon=True)
    mqtt_publish_thread.start()
    # Create a client instance
    client = mqtt.Client()
    if MQTT_USERNAME is not None:
        client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)
    # Assign event callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_publish = on_publish
    # Connect to the broker
    client.connect(MQTT_BROKER, MQTT_PORT)

    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()


if __name__ == "__main__":
    main()
