YAML_EXT = (".yaml", ".yml")
REGEX_CAMERA_NAME = r"^[a-zA-Z0-9_-]+$"
MQTT_FRIGATE_TOPIC = "frigate/events"


EVENTS_ENDPOINT = "/api/events/{}"
CLIP_ENDPOINT = "/api/events/{}/clip.mp4"
SNAPSHOT_ENDPOINT = "/api/events/{}/snapshot.jpg?quality=100&crop=0&bbox=0&timestamp=0&h={}"
SUB_LABEL_ENDPOINT = "/api/events/{}/sub_label"
DESCRIPTION_ENDPOINT = "/api/events/{}/description"

