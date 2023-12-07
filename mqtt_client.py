#!/usr/bin/env python
import base64
from email.mime import message
import pathlib
from re import A
import subprocess
from json import JSONDecodeError
from typing import Any, Dict, List

import imageio
import os
import sys
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
from uuid import getnode as get_mac
from typing import MutableMapping, Any, List, Dict
from dataclasses import dataclass, field
from config import AmbleConfig, CameraConfig, init_config
from const import (
    MQTT_HA_SWITCH_COMMAND_TOPIC,
    MQTT_HA_SWITCH_CONFIG_TOPIC,
    MQTT_HA_SWITCH_STATE_TOPIC,
    SNAPSHOT_ENDPOINT, 
    CLIP_ENDPOINT, 
    EVENTS_ENDPOINT, 
    DESCRIPTION_ENDPOINT, 
    SUB_LABEL_ENDPOINT,
    MQTT_FRIGATE_TOPIC,
    HOMEASSISTANT_DISCOVERY,
    MQTT_HA_SENSOR_CONFIG_TOPIC,
    MQTT_HA_SWITCH_TOPIC,
    OPENAI_ENDPOINT
)
from version import VERSION

logging.basicConfig(level=logging.DEBUG, format=
                    "%(levelname)-s %(asctime)s %(threadName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

config: AmbleConfig = init_config()

outgoing_message_queue = queue.PriorityQueue(maxsize=20)
ongoing_tasks: MutableMapping[str, threading.Thread] = TTLCache(maxsize=1000, ttl=timedelta(hours=4), timer=datetime.now)
recent_triggered_cameras: MutableMapping[str, bool] = TTLCache(maxsize=20, ttl=config.event.debounce)

stop_event: threading.Event = threading.Event()
client: mqtt.Client = mqtt.Client(client_id="amblegpt-mqtt")
usage_dict: Dict = {}
valid_camera_list: List = []
amblegpt_enabled = True

def generate_prompt(event_start_time: str, camera_name: str) -> str:
    templated_prompt = Template(config.prompt.prompt)
    return templated_prompt.substitute(
        RESULT_LANGUAGE=config.prompt.language,
        GAP_SECS=config.image.interval,
        EVENT_START_TIME=event_start_time,
        CAMERA_PROMPT=config.cameras[camera_name].prompt
    )

def get_unique_identifier() -> str:
    return str(get_mac())

def append_ha_availability_payload(payload: dict):
    payload["availability"] = [
        {"topic": f"{config.mqtt.topic_prefix}/status"},
    ]
    payload["device"] = {
        "name": "AmbleGPT",
        "sw_version": VERSION,
        "identifiers": [ f"{config.mqtt.ha_unique_id}" ],
        }
    payload["unique_id"] = payload["unique_id"].format(config.mqtt.ha_unique_id)

    if "{}" in payload["state_topic"] and MQTT_HA_SWITCH_TOPIC not in payload["state_topic"]:
        payload["state_topic"] = payload["state_topic"].format(
            config.mqtt.topic_prefix
        )

    if "{}" in payload["state_topic"] and MQTT_HA_SWITCH_TOPIC in payload["state_topic"]:
        payload["state_topic"] = payload["state_topic"].format(
            payload["unique_id"]
        ) 

    if "command_topic" in payload:
        payload["command_topic"] = payload["command_topic"].format(
            payload["unique_id"]
        )
    return payload

def get_local_time_str(ts: float):
    # convert the timestamp to a datetime object in the local timezone
    dt_object = datetime.fromtimestamp(ts)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def prompt_gpt4_with_video_frames(prompt, base64_frames,
                                   detail) -> requests.Response:
    logging.info("prompting GPT-4v")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }
    prompt_messages = [
        {
            "role": "system",
            "content": config.prompt.system
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
                            "detail": detail,
                        },
                    },
                    base64_frames,
                ),
            ],
        },
    ]
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": config.prompt.max_tokens,
    }

    return requests.post(
        OPENAI_ENDPOINT, headers=headers, json=payload
    )


def is_ffmpeg_available() -> None:
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
        resize = config.image.frame_size
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
            while len(frames) < config.image.max_frames - 1:
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


def download_snapshot_and_combine_frames(event_id, gap_secs):
    snapshot_url = f"http://{config.frigate.host}:{config.frigate.port}{SNAPSHOT_ENDPOINT.format(event_id, config.image.frame_size)}"

    frame_count = 0
    retry_attempts = 0
    frame_list = []
    start_time = time.monotonic()
    while frame_count < config.image.max_frames and not stop_event.is_set():
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


def download_video_clip_and_extract_frames(event_id, gap_secs):
    f"""
    download video clip for event id and extract frames every gap_secs
    :param event_id: frigate event id to fetch
    :param gap_secs: period in seconds to wait between still images to extract from clip
    :return: list of extracted
    """
    clip_url = f"http://{config.frigate.host}:{config.frigate.port}{CLIP_ENDPOINT.format(event_id)}"
    event_data = f"http://{config.frigate.host}:{config.frigate.port}{EVENTS_ENDPOINT.format(event_id)}"

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
    global usage_dict
    try:
        event_id = payload["after"]["id"]

        # If the event is complete, we have no reason to wait
        if not config.event.process_complete:
            time.sleep(config.event.pre_wait_time)

        if config.event.snapshots:
            video_base64_frames = download_snapshot_and_combine_frames(
                event_id, gap_secs=config.image.interval
            )
        elif config.event.clips:
            video_base64_frames = download_video_clip_and_extract_frames(
                event_id, gap_secs=config.image.interval
            )
        else:
            raise RuntimeError("Error, both PROCESS_SNAPSHOT and PROCESS_CLIP cannot be disable")

        if len(video_base64_frames) < config.image.min_frames:
            if event_id in ongoing_tasks:
                logging.info(f"Canceling message processing, clip is still too short: {event_id}")
                with threading.Lock():
                    ongoing_tasks.pop(event_id)
                if event_id in ongoing_tasks:
                    raise Exception("Failed to remove event_id from cache")
            return

        local_time_str = get_local_time_str(ts=payload["after"]["start_time"])
        prompt = generate_prompt(
            event_start_time=local_time_str, camera_name=payload["after"]["camera"]
        )

        if stop_event.is_set():
            return
        try:
            response = prompt_gpt4_with_video_frames(prompt, video_base64_frames, config.image.image_quality)
            logging.debug(f"GPT response: {response.json()}")
            logging.debug(f"Response headers: {response.headers}")
        except Exception as ex:
            logging.error(f"GPT API produced an error: {ex}")
            return

        json_str = response.json()["choices"][0]["message"]["content"]
        result = json.loads(json_str)
        usage = response.json()["usage"]

        # Set the summary to the 'after' field
        payload["after"]["summary"] = result["summary"]
        payload["after"]["summary_title"] = result["title"]

        if result["num_persons"] > 0:
            payload["after"]["summary_details"] = result["persons"]

        if config.frigate.update_description:
            description = f"{result['summary']}"
            if result["persons"] and int(result["num_persons"]) > 0:
                for index, person in enumerate(result["persons"]):
                    description += f" Person {index+1}: "
                    for key in person:
                        description += f" {key}: {person[key]}, "

                    description_stripped = description.strip()
                    description = description_stripped + "."
            trimmed_description = description.strip()
            payload["after"]["description"] = trimmed_description
            url = f"http://{config.frigate.host}:{config.frigate.port}{DESCRIPTION_ENDPOINT.format(event_id)}"
            request = requests.post(url, json={"description": trimmed_description})
            if request.status_code == 200:
                logging.debug(f"Frigate description updated with summary for event: {event_id}")
            else:
                logging.error(f"Failed updating Frigate description for event: {event_id}")

        if config.frigate.update_label:
            sub_label = f"{result['title']}"
            url = f"http://{config.frigate.host}:{config.frigate.port}{SUB_LABEL_ENDPOINT.format(event_id)}"
            request = requests.post(url, json={"sub_label": sub_label})
            if request.status_code == 200:
                logging.debug(f"Frigate sub_label updated with summary for event: {event_id}")
            else:
                logging.error(f"Failed sub_label Frigate description for event: {event_id}")

        outgoing_message_queue.put(
            PrioritizedItem(20, (f"{config.mqtt.topic_prefix}/events", payload))
            )
        if usage_dict:
            usage_dict["prompt_tokens"] += usage["prompt_tokens"]
            usage_dict["completion_tokens"] += usage["completion_tokens"]
            usage_dict["total_tokens"] += usage["total_tokens"]
        else:
            usage_dict = usage

        outgoing_message_queue.put(
            PrioritizedItem(1,
                            (f"{config.mqtt.topic_prefix}/token_usage",
                              usage_dict))
            )

    except JSONDecodeError as ex:
        logging.error(f"Exception: {ex}")
        logging.error(f"Json response from GPT was received with incorrect structure.")
    except Exception as ex:
        logging.exception(f"Error processing video for event {event_id}")
        logging.error(f"Exception: {ex}")


# Define what to do when the client connects to the broker
def on_connect(client: mqtt.Client, userdata, flags, rc):
    logging.info(f"Connected with result code " + str(rc))
    if rc > 0:
        logging.debug(f"Connected with result code {str(rc)}")
    # Subscribe to the topic
    client.subscribe(MQTT_FRIGATE_TOPIC)
    client.subscribe(f"{config.mqtt.topic_prefix}/token_usage", qos=1)
    logging.info("Subscribed to topic: frigate/events")
    client.publish(f"{config.mqtt.topic_prefix}/status", "online", retain=True)

    if config.homeassistant.autodiscovery:
        ha_device_list = [k  for  k in  HOMEASSISTANT_DISCOVERY.keys()]
        for device in ha_device_list:
            if device == "switch":
                #formated_payload = append_ha_availability_payload(HOMEASSISTANT_DISCOVERY[device])
                #formatted_topic = MQTT_HA_SWITCH_CONFIG_TOPIC.format(formated_payload["unique_id"])
                client.publish(topic=config.mqtt.switch_config_topic, 
                               payload=json.dumps(append_ha_availability_payload(HOMEASSISTANT_DISCOVERY[device])), 
                               retain=True)
            elif device.endswith("_tokens") or device.endswith("_cost"):
                #formated_payload = append_ha_availability_payload(HOMEASSISTANT_DISCOVERY[device])
                #formatted_topic = MQTT_HA_SENSOR_CONFIG_TOPIC.format(formated_payload["unique_id"])
                client.publish(topic=MQTT_HA_SENSOR_CONFIG_TOPIC.format(f"{config.mqtt.ha_unique_id}_{device}"),
                       payload=json.dumps(append_ha_availability_payload(HOMEASSISTANT_DISCOVERY[device])),
                       retain=True)
        
        client.publish(config.mqtt.switch_state_topic, "ON")
        client.subscribe(config.mqtt.switch_command_topic)
        #client.publish(MQTT_HA_SWITCH_STATE_TOPIC.format(f"{get_unique_identifier()}_switch"), "ON")
        #client.subscribe(MQTT_HA_SWITCH_COMMAND_TOPIC.format(f"{get_unique_identifier()}_switch"))

    

def on_publish(client, userdata, mid):
    logging.debug(f"mqtt published message id {mid}")


def on_message(client, userdata, msg):
    global ongoing_tasks
    global valid_camera_list
    try:
        # Parse the message payload as JSON
        payload = json.loads(msg.payload.decode("utf-8"))
        event_id = payload["before"]["id"]
        valid_objects = config.frigate.tracked_objects
        camera = payload["before"]["camera"]

        logging.debug(f"Received MQTT message for event: {event_id}")
        
        if camera not in valid_camera_list:
            logging.debug(f"{camera} is not a valid camera for analysis, not in {valid_camera_list}")

        if event_id in ongoing_tasks:
            logging.debug(f"Not processing running event: {event_id}")
            return

        if config.event.process_complete and payload["type"] != "end":
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

        if not payload["after"]["has_clip"] and config.event.clips:
            # Skip if this snapshot has already been processed
            logging.debug("Skipping because of no available video clip yet, and configured for processing stream")
            return

        if payload["before"]["stationary"] or payload["after"]["stationary"]:
            logging.debug("Skipping event for stationary object")
            return

        if payload["before"]["label"] not in valid_objects or payload["after"]["label"] not in valid_objects:
            logging.debug(f"Skipping event for object not in {*valid_objects,}")
            return

        if camera in recent_triggered_cameras:
            logging.debug(f"Skipping event due to debounce interval")
            return
        

        # Start a new thread for the new message
        with threading.Lock():
            ongoing_tasks[event_id] = threading.Thread(target=process_message, args=(payload,), name=f"Thread-{threading.active_count()}-{camera}")
            recent_triggered_cameras[camera] = True
        ongoing_tasks[event_id].start()

    except json.JSONDecodeError:
        logging.exception("Error decoding JSON")
    except KeyError:
        logging.exception("Key not found in JSON payload")


def publish_message():
    retainFlag = False
    qosFlag = 1

    while True:
        priority_message = outgoing_message_queue.get()
        message = priority_message.item

        if message[0] == "" and message[1] == "":
            logging.warning("Publishing thread received exit signal, terminating...")
            break
        elif message[0] == f"{config.mqtt.topic_prefix}/token_usage":
            retainFlag = True
        
        publish_results = client.publish(message[0], json.dumps(message[1]), qos=qosFlag, retain=retainFlag)
        logging.info(f"Publishing updated payload with summary back to MQTT topic.")
        publish_results.wait_for_publish()
        logging.info(f"MQTT Published: {publish_results.is_published()}")


def handle_token_usage(client, userdata, msg):
    if msg.payload:
        payload = json.loads(msg.payload.decode("utf-8"))
    if not usage_dict and payload:
        usage_dict["prompt_tokens"] = int(payload["prompt_tokens"])
        usage_dict["completion_tokens"] = int(payload["completion_tokens"])
        usage_dict["total_tokens"] = int(payload["total_tokens"])
    logging.info("unsubscribing from history")
    client.unsubscribe(f"{config.mqtt.topic_prefix}/token_usage")


def handle_ha_mmessage(client, userdata, msg):
    global amblegpt_enabled
    command = msg.payload.decode("utf-8").upper()
    logging.info(f"received ha command state: {command}")
    if command == "OFF" and amblegpt_enabled:
        client.unsubscribe(MQTT_FRIGATE_TOPIC)
        client.publish(f"{MQTT_HA_SWITCH_STATE_TOPIC.format(f'{config.mqtt.ha_unique_id}_switch')}",
                       "OFF")
        amblegpt_enabled = False
        logging.info("home assistant switch disabled processing, unsubscribing")
    if command == "ON" and not amblegpt_enabled:
        client.subscribe(MQTT_FRIGATE_TOPIC)
        amblegpt_enabled = True
        client.publish(f"{MQTT_HA_SWITCH_STATE_TOPIC.format(f'{config.mqtt.ha_unique_id}_switch')}",
                       "ON")
        logging.info("home assistant switch enabled processing, subscribing")


def main():
    global client
    global mqtt_publish_thread
    global valid_camera_list
    logging.info(f"ffmpeg for video processing is enabled: {is_ffmpeg_available()}")
    # Create a client instance
    #client = mqtt.Client()
    if config.mqtt.user is not None:
        client.username_pw_set(username=config.mqtt.user,
                            password=config.mqtt.password)
    # Assign event callbacks
    client.message_callback_add(sub=f"{config.mqtt.topic_prefix}/token_usage",
                                callback=handle_token_usage)
    
    if config.homeassistant.autodiscovery:
        client.message_callback_add(sub=f"{MQTT_HA_SWITCH_COMMAND_TOPIC.format(f'{config.mqtt.ha_unique_id}_switch')}",
                                    callback=handle_ha_mmessage)
        
    client.message_callback_add(
        sub=MQTT_FRIGATE_TOPIC,
        callback=on_message
    )
        
    client.on_connect = on_connect
    #client.on_message = on_message
    client.on_publish = on_publish
    # Set last will
    client.will_set(
        topic=f"{config.mqtt.topic_prefix}/status",
        payload="offline",
        qos=1,
        retain=True
    )

    for name, camera in config.cameras.items():
        if camera.enabled:
            valid_camera_list.append(name)
    # mqtt connect
    try:
        client.connect(config.mqtt.host, config.mqtt.port)
    except TimeoutError as ex:
        logging.error("Unable to connect to MQTT broker, exiting...")
        sys.exit(1)
    
    mqtt_publish_thread = threading.Thread(target=publish_message, name="mqtt-publish-thread")
    mqtt_publish_thread.start()
    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt called, closing all threads")
        #outgoing_message_queue.put((0, (MQTT_TOKEN_USAGE_TOPIC, usage_dict)), False)
        outgoing_message_queue.put(PrioritizedItem(0,("","")))
    
        stop_event.set()
        for event_thread in ongoing_tasks:
            if ongoing_tasks[event_thread].is_alive():
                ongoing_tasks[event_thread].join()

        sys.exit(1)
