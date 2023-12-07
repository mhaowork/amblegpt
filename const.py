OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

YAML_EXT = (".yaml", ".yml")
MQTT_FRIGATE_TOPIC = "frigate/events"

MQTT_HA_SWITCH_TOPIC = "homeassistant/switch/{}"
MQTT_HA_BASE_SENSOR_TOPIC = "homeassistant/sensor/{}"

MQTT_HA_SENSOR_CONFIG_TOPIC = MQTT_HA_BASE_SENSOR_TOPIC + "/config"

MQTT_HA_SWITCH_CONFIG_TOPIC = MQTT_HA_SWITCH_TOPIC + "/config"
MQTT_HA_SWITCH_COMMAND_TOPIC = MQTT_HA_SWITCH_TOPIC + "/set"
MQTT_HA_SWITCH_STATE_TOPIC = MQTT_HA_SWITCH_TOPIC + "/state"


EVENTS_ENDPOINT = "/api/events/{}"
CLIP_ENDPOINT = "/api/events/{}/clip.mp4"
SNAPSHOT_ENDPOINT = "/api/events/{}/snapshot.jpg?quality=100&crop=0&bbox=0&timestamp=0&h={}"
SUB_LABEL_ENDPOINT = "/api/events/{}/sub_label"
DESCRIPTION_ENDPOINT = "/api/events/{}/description"

COST_TEMPLATE_STRING =  "{{ ((value_json.prompt_tokens / 1000) * 0.01) | round(4) + ((value_json.completion_tokens / 1000) * 0.03) | round(4) }}"
HOMEASSISTANT_DISCOVERY = {
"switch": {
    "name": "AmbleGPT",
    "command_topic": MQTT_HA_SWITCH_COMMAND_TOPIC,
    "state_topic": MQTT_HA_SWITCH_STATE_TOPIC,
    "unique_id": "{}_switch",
    "icon": "mdi:head-sync-outline",
    },
"prompt_tokens": {
    "name": "Prompt Tokens Used",
    "state_class": "measurement",
    "icon": "mdi:counter",
    "value_template": "{{ value_json.prompt_tokens }}",
    "unique_id": "{}_prompt_tokens",
    "state_class": "total_increasing",
    "state_topic": "{}/token_usage",
},
"completion_tokens": {
    "name": "Completion Tokens Used",
    "state_class": "measurement",
    "icon": "mdi:counter",
    "value_template": "{{ value_json.completion_tokens }}",
    "unique_id": "{}_completion_tokens",
    "state_class": "total_increasing",
    "state_topic": "{}/token_usage",
},
"total_tokens": {
    "name": "Total Tokens Used",
    "state_class": "measurement",
    "icon": "mdi:counter",
    "value_template": "{{ value_json.total_tokens }}",
    "unique_id": "{}_total_tokens",
    "state_class": "total_increasing",
    "state_topic": "{}/token_usage",
},
"total_cost": {
    "name": "Approximate Cost",
    "state_class": "measurement",
    "device_class": "monetary",
    "icon": "mdi:currency-usd",
    "unit_of_measurement": "$",
    "unique_id": "{}_total_cost",
    "state_class": "total_increasing",
    "state_topic": "{}/token_usage",
    "value_template": COST_TEMPLATE_STRING,
}
}