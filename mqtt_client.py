import base64
import pathlib
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

logging.basicConfig(level=logging.INFO, format="%(processName)s: %(message)s")

ongoing_tasks = {}

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
THUMBNAIL_ENDPOINT = "/api/events/{}/thumbnail.jpg"
CLIP_ENDPOINT = "/api/events/{}/clip.mp4"

# Video frame sampling settings
GAP_SECS = 3

# GPT config
DEFAULT_PROMPT = """
You're a helpful assistant helping to label a video for machine learning training
You are reviewing some continuous frames of a video footage as of {EVENT_START_TIME}. Frames are {GAP_SECS} second(s) apart from each other in the chronological order…
Please describe what happend in the video in json format. Do not print any markdown syntax!
Answer like the following:
{{
    num_persons : 2,
    persons : [
    {{
        height_in_meters: 1.75,
        duration_of_stay_in_seconds: 15,
        gender: "female",
        age: 50
    }},
    {{
        height_in_meters: 1.60,
        duration_of_stay_in_seconds: 15,
        gender: "unknown",
        age: 36
    }},
    summary: SUMMARY
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
    4. A person is wandering without obvious purpose in the middle of the night, which seems suspicious.
    5. A person walked into the frame from outside, picked up a package, and left.
       The person didn't wear any uniform so this doesn't look like a routine package pickup. Be aware of potential package theft!
"""

PROMPT_TEMPLATE = config.get("prompt", DEFAULT_PROMPT)


def generate_prompt(gap_secs, event_start_time):
    return PROMPT_TEMPLATE.format(GAP_SECS=gap_secs, EVENT_START_TIME=event_start_time)


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
        "max_tokens": 200,
    }

    return requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )


def extract_frames(video_path, gap_secs):
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


# Function to download video clip and extract frames
def download_video_clip_and_extract_frames(event_id, gap_secs):
    clip_url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{CLIP_ENDPOINT.format(event_id)}"
    response = requests.get(clip_url)

    if response.status_code == 200:
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        clip_filename = os.path.join(temp_dir.name, f"clip_{event_id}.mp4")

        # clip_filename = "cache_video/" + f"clip_{event_id}.mp4"
        try:
            with open(clip_filename, "wb") as f:
                f.write(response.content)
            logging.info(f"Video clip for event {event_id} saved as {clip_filename}.")

            # After downloading, extract frames
            frames = extract_frames(clip_filename, gap_secs)
            os.remove(clip_filename)
        except OSError:
            logging.error(
                f"Failed to save video clip for event {event_id}. Error: {OSError.strerror}"
            )
            frames = []
        return frames
    else:
        logging.error(
            f"Failed to retrieve video clip for event {event_id}. Status code: {response.status_code}"
        )
        return []


def process_message(payload):
    try:
        event_id = payload["after"]["id"]
        video_base64_frames = download_video_clip_and_extract_frames(
            event_id, gap_secs=GAP_SECS
        )

        if len(video_base64_frames) == 0:
            return

        local_time_str = get_local_time_str(ts=payload["after"]["start_time"])
        prompt = generate_prompt(GAP_SECS, local_time_str)
        response = prompt_gpt4_with_video_frames(prompt, video_base64_frames)
        logging.info(f"GPT response {response.json()}")
        json_str = response.json()["choices"][0]["message"]["content"]
        result = json.loads(json_str)

        # Set the summary to the 'after' field
        payload["after"]["summary"] = "| GPT: " + result["summary"]

        # Convert the updated payload back to a JSON string
        updated_payload_json = json.dumps(payload)

        # Publish the updated payload back to the MQTT topic
        # Create a new MQTT client
        client = mqtt.Client()
        if MQTT_USERNAME is not None:
            client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)

        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.publish(MQTT_SUMMARY_TOPIC, updated_payload_json)
        logging.info("Published updated payload with summary back to MQTT topic.")
    except Exception:
        logging.exception(f"Error processing video for event {event_id}")
    finally:
        # Cleanup: remove the task from the ongoing_tasks dict
        if event_id in ongoing_tasks:
            del ongoing_tasks[event_id]


# Define what to do when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code " + str(rc))
    if rc > 0:
        print("Connected with result code", rc)  # Print the result code for debugging
        return
    client.subscribe(MQTT_FRIGATE_TOPIC)  # Subscribe to the topic
    print(
        "Subscribed to topic:", MQTT_FRIGATE_TOPIC
    )  # Print the subscribed topic for debugging


# Define what to do when a message is received
def on_message(client, userdata, msg):
    global ongoing_tasks

    # Parse the message payload as JSON
    event_id = None
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if "summary" in payload["after"] and payload["after"]["summary"]:
            # Skip if this message has already been processed. To prevent echo loops
            logging.info("Skipping message that has already been processed")
            return
        if (
            payload["before"].get("snapshot_time")
            == payload["after"].get("snapshot_time")
            and (payload["type"] != "end")
            and (event_id in ongoing_tasks)
        ):
            # Skip if this snapshot has already been processed
            logging.info(
                "Skipping because the message with this snapshot is already (being) processed"
            )
            return
        if not payload["before"]["has_clip"]:
            # Skip if this snapshot has already been processed
            logging.info("Skipping because of no available video clip yet")
            return
        event_id = payload["after"]["id"]
        logging.info(f"Event ID: {event_id}")

        # If there's an ongoing task for the same event, terminate it
        if event_id in ongoing_tasks:
            ongoing_tasks[event_id].terminate()
            ongoing_tasks[event_id].join()  # Wait for process to terminate
            logging.info(f"Terminated ongoing task for event {event_id}")

        # Start a new task for the new message
        processing_task = Process(target=process_message, args=(payload,))
        processing_task.start()
        ongoing_tasks[event_id] = processing_task

    except json.JSONDecodeError:
        logging.exception("Error decoding JSON")
    except KeyError:
        logging.exception("Key not found in JSON payload")


if __name__ == "__main__":
    # Create a client instance
    client = mqtt.Client()

    # Assign event callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    if MQTT_USERNAME is not None:
        client.username_pw_set(MQTT_USERNAME, password=MQTT_PASSWORD)
    # Connect to the broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()
