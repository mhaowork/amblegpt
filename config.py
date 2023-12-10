from re import A
import yaml
from enum import Enum
from pydantic import BaseModel, Field, validator, computed_field
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
from const import YAML_EXT
from prompt import SYSTEM_PROMPT, DEFAULT_PROMPT
import json
import os
from uuid import getnode as get_mac

from const import (
    MQTT_HA_SWITCH_COMMAND_TOPIC,
    MQTT_HA_SWITCH_CONFIG_TOPIC,
    MQTT_HA_SWITCH_STATE_TOPIC,
    MQTT_HA_TEXT_CONFIG_TOPIC,
    MQTT_HA_TEXT_COMMAND_TOPIC,
    MQTT_HA_TEXT_STATE_TOPIC,
)


def load_config_with_no_duplicates(raw_config) -> dict:
    """Get config ensuring duplicate keys are not allowed."""

    # https://stackoverflow.com/a/71751051
    # important to use SafeLoader here to avoid RCE
    class PreserveDuplicatesLoader(yaml.loader.SafeLoader):
        pass

    def map_constructor(loader, node, deep=False):
        keys = [loader.construct_object(node, deep=deep) for node, _ in node.value]
        vals = [loader.construct_object(node, deep=deep) for _, node in node.value]
        key_count = Counter(keys)
        data = {}
        for key, val in zip(keys, vals):
            if key_count[key] > 1:
                raise ValueError(
                    f"Config input {key} is defined multiple times for the same field, this is not allowed."
                )
            else:
                data[key] = val
        return data

    PreserveDuplicatesLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, map_constructor
    )
    return yaml.load(raw_config, PreserveDuplicatesLoader)


class AmbleBaseModel(BaseModel):
    class Config:
        extra = "allow"

class MQTTConfig(BaseModel):
    host: str = Field(default="", title="MQTT Host")
    port: int = Field(default=1883, title="MQTT Port")
    topic_prefix: str = Field(default="amblegpt", title="MQTT Topic Prefix")
    client_id: str = Field(default="amblegpt-client", title="MQTT Client ID")
    user: Optional[str] = Field(title="MQTT Username")
    password: Optional[str] = Field(title="MQTT Password")

    @validator("password", pre=True, always=True)
    def validate_password(cls, v, values):
        if (v is None) != (values["user"] is None):
            raise ValueError("Password must be provided with username.")
        return v
    
    @computed_field
    @property
    def ha_unique_id(self) -> str:
        return str(get_mac())
    
    @computed_field(alias="switch_command_topic", description="MQTT switch command topic for HA autodiscovery")
    @property
    def switch_command_topic(self) -> str:
        return MQTT_HA_SWITCH_COMMAND_TOPIC.format(f"{self.ha_unique_id}_switch")
    
    @computed_field(alias="switch_config_topic", description="MQTT switch config topic for HA autodiscovery")
    @property
    def switch_config_topic(self) -> str:
        return MQTT_HA_SWITCH_CONFIG_TOPIC.format(f"{self.ha_unique_id}_switch")

    @computed_field(alias="switch_state_topic", description="MQTT switch state topic for HA autodiscovery")
    @property
    def switch_state_topic(self) -> str:
        return MQTT_HA_SWITCH_STATE_TOPIC.format(f"{self.ha_unique_id}_switch")

    @computed_field(alias="text_state_topic", description="MQTT text state topic for HA autodiscovery")
    @property
    def text_state_topic(self) -> str:
        return MQTT_HA_TEXT_STATE_TOPIC.format(f"{self.ha_unique_id}_last_summary")


class FrigateConfig(AmbleBaseModel):
    host: str = Field(default="", title="Frigate Host")
    port: int = Field(default=5000, title="Frigate Port")
    tracked_objects: List[str] = Field(default=["person"], title="Objects to track.")
    update_label:  bool = Field(default=False, title="Update the frigate sublabel")
    update_description: bool = Field(default=False, title="Update the frigate description field, currently not available in released frigate")

class ImageQualityEnum(str, Enum):
    low = "low"
    high = "high"

class ImageConfig(AmbleBaseModel):
    interval: int = Field(
        default=3, title="Number of seconds between frames to use for submission"
    )
    frame_size: int = Field(
        default=512, title="Resolution of use for resizing the image, on largest side"
    )
    max_frames: int = Field(
        default=10, title="Maximum number of frames to capture per event"
    )
    min_frames: int = Field(
        default=2, title="Minimum number of frames to capture per event"
    )
    image_quality: ImageQualityEnum = Field(
        default=ImageQualityEnum.low, title="Image quality chatGPT will use for processing"
    )

class EventConfig(AmbleBaseModel):
    clips: bool = Field(default=True, title="Enable using video clips for image capture")
    snapshots: bool = Field(default=False, title="Enable using snapshots for image capture")
    debounce: int = Field(default=30, title="Time in seconds to not process any events from a triggered camera")
    process_complete: bool = Field(default=False, title="Only process clips after an event is done, this is ignored for snapshots")
    pre_wait_time: int = Field(default=3, title="Seconds to pause before processing an event")

    @validator("snapshots")
    def mutually_exclusive(cls, v, values):
        if values["clips"] is True and v is True:
            raise ValueError("'clips' and 'snapshots' are mutually exclusive, only one can be used for capture.")
        return v

class CameraConfig(AmbleBaseModel):
    name: str = Field(default=None, title="Camera name.")
    enabled: bool = Field(default=True, title="Enable camera, trigger on events.")
    prompt: Optional[str] = Field(title="Custom prompt for camera.")


class LogLevelEnum(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"



class PromptConfig(AmbleBaseModel):
    system: str = Field(default=SYSTEM_PROMPT, title="System prompt template")
    prompt: str = Field(default=DEFAULT_PROMPT, title="Default prompt template")
    language: str = Field(default="english", title="GPT language")
    max_tokens: int = Field(default=500, title="Max tokens to allow per message")

class HAConfig(AmbleBaseModel):
    autodiscovery: bool = Field(default=False, title="Enable Home Assistant Auto Discovery")


class LoggerConfig(AmbleBaseModel):
    default: LogLevelEnum = Field(
        default=LogLevelEnum.debug, title="Default logging level."
    )

class AmbleConfig(AmbleBaseModel):
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig, 
                             title="MQTT Configuration.")
    frigate: FrigateConfig = Field(default_factory=FrigateConfig,
                                   title="Frigate Configuration.")
    image: ImageConfig = Field(default_factory=ImageConfig,
                               title="Image/Video frame Configuration.")
    event: EventConfig = Field(default_factory=EventConfig,
                               title="Event/Trigger Configurations.")
    cameras: Dict[str, CameraConfig] = Field(default_factory=CameraConfig, title="Camera configuration.")
    prompt: PromptConfig = Field(default_factory=PromptConfig,
                                 title="Prompt template configurations")
    homeassistant: HAConfig = Field(default_factory=HAConfig,
                                    title="Home Assistant configurations")
    logger: LoggerConfig = Field(
        default_factory=LoggerConfig, title="Logging configuration."
    )
    
    @classmethod
    def parse_file(cls, config_file="config.yml"):
        with open(config_file) as f:
            raw_config = f.read()

        if config_file.endswith(YAML_EXT):
            config = load_config_with_no_duplicates(raw_config)
        elif config_file.endswith(".json"):
            config = json.loads(raw_config)

        return cls.model_validate(config)
    # @classmethod
    # def parse_raw(cls, raw_config):
    #     config = load_config_with_no_duplicates(raw_config)
    #     return cls.parse_obj(config)
    

def init_config() -> AmbleConfig:
    config_file = os.environ.get("CONFIG_FILE", "config.yml")
    config_file_yaml = config_file.replace(".yml", ".yaml")
    if os.path.isfile(config_file_yaml):
        config_file = config_file_yaml
    user_config = AmbleConfig.parse_file(config_file)
    return user_config