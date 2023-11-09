import base64
import imageio
import os
from dotenv import load_dotenv
import json
import requests
import paho.mqtt.client as mqtt
import io
from PIL import Image


# Load environment variables from .env file
load_dotenv()


# Define the MQTT server settings
MQTT_BROKER = "100.116.240.70"
MQTT_PORT = 1883
MQTT_TOPIC = "frigate/events"


# Define Frigate server details for thumbnail retrieval
FRIGATE_SERVER_IP = "100.116.240.70"
FRIGATE_SERVER_PORT = 5000
THUMBNAIL_ENDPOINT = "/api/events/{}/thumbnail.jpg"
CLIP_ENDPOINT = "/api/events/{}/clip.mp4"

# Video frame sampling settings
GAP_SECS = 3

# GPT configs
PROMPT = """
You're a helpful assistant helping to label a video for machine learning training
You are reviewing some continuous frames of a video footage. Frames are %d second(s) apart from each other in the chronological order…
Please describe what happend in the video in json format. Do not print any markdown syntax!
Answer like the following:
{
    num_persons : 2,
    persons : [
    { 
        height_in_meters: 1.75,
        duration_of_stay_in_seconds: 15,
        gender: "female",
        age: 50
    },
    {
        height_in_meters: 1.60,
        duration_of_stay_in_seconds: 15,
        gender: "unknown",
        age: 36
    },
    summary: SUMMARY
}

You can guess their height and gender . It is 100 percent fine to be inaccurate.
You can measure their duration of stay given the time gap between frames.
Some example SUMMARIES are
    1. One person walked by towards right corner with her dog without paying attention towards the camera's direction.
    2. One Amazon delivery person (in blue vest) dropped off a package
    3. A female is waiting, facing the door
    4. A person is wandering without obvious purpose
""" % (
    GAP_SECS
)


def prompt_gpt4(filename: str):
    prompt = "Please describe this image as if you are a security guard responsible for protecting a high-value vehicle. \
    You're reviewing an image from a security camera mounted outside on one side of the vehicle's roof rack. \
    Describe any potential security threats or suspicious activities you observe in the image.\
    "
    # Getting the base64 string
    with open(filename, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    print(response.json())


def prompt_gpt4_with_video_frames(prompt, base64_frames):
    print("prompting GPT-4v")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(
                    lambda frame: {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
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


# Function to download video clip and extract frames
def download_video_clip_and_extract_frames(event_id, gap_secs):
    clip_url = f"http://{FRIGATE_SERVER_IP}:{FRIGATE_SERVER_PORT}{CLIP_ENDPOINT.format(event_id)}"
    response = requests.get(clip_url)

    if response.status_code == 200:
        clip_filename = f"clip_{event_id}.mp4"
        with open(clip_filename, "wb") as f:
            f.write(response.content)
        print(f"Video clip for event {event_id} saved as {clip_filename}.")

        # After downloading, extract frames
        return extract_frames(clip_filename, gap_secs)
    else:
        print(
            f"Failed to retrieve video clip for event {event_id}. Status code: {response.status_code}"
        )
        return []


def extract_frames(video_path, gap_secs):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    frames = []

    for i, frame in enumerate(reader):
        # Extract a frame every {gap_secs} seconds
        if i % (int(gap_secs * fps)) == 0:
            # Convert to PIL Image to resize
            image = Image.fromarray(frame)
            # Calculate the new size, maintaining the aspect ratio
            ratio = min(500 / image.size[0], 500 / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # Resize the image
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            # Convert back to bytes
            with io.BytesIO() as output:
                resized_image.save(output, format="JPEG")
                frame_bytes = output.getvalue()
            frames.append(base64.b64encode(frame_bytes).decode("utf-8"))
    reader.close()
    print(f"Got {len(frames)} frames from video")
    return frames


# Define what to do when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)  # Subscribe to the topic


# Define what to do when a message is received
def on_message(client, userdata, msg):
    # Parse the message payload as JSON
    event_id = None
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if payload["after"]["summary"]:
            # Skip if this message has already been processed
            print("Skipping message that has already been processed")
            return
        event_id = payload["after"]["id"]
        print(f"Event ID from 'after': {event_id}")

        video_base64_frames = download_video_clip_and_extract_frames(
            event_id, gap_secs=GAP_SECS
        )

        response = prompt_gpt4_with_video_frames(PROMPT, video_base64_frames)
        print("response.json()", response.json())
        json_str = response.json()["choices"][0]["message"]["content"]
        result = json.loads(json_str)
        print("summary", result["summary"])

        # Set the summary to the 'after' field
        payload["after"]["summary"] = "| GPT: " + result["summary"]

        # Convert the updated payload back to a JSON string
        updated_payload_json = json.dumps(payload)

        # Publish the updated payload back to the MQTT topic
        client.publish(MQTT_TOPIC, updated_payload_json)
        print("Published updated payload with summary back to MQTT topic.")

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
    except KeyError as e:
        print(f"Key {e} not found in JSON payload")


if __name__ == "__main__":
    # Create a client instance
    client = mqtt.Client()

    # Assign event callbacks
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()
