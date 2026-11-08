"""Read a trial bag and its trial.yaml, shared by analyze_trials and replay_bag."""

import os

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml


def read_bag(path, topics):
    """
    Read the given topics from an mcap bag, as {topic: [(stamp_s, msg), ...]}.

    The stamp is the message's header.stamp, not the recorder's receive time:
    every node stamps with the wall clock, so all topics share one clock.

    :param path: bag folder
    :param topics: topic names to read
    :return: messages per topic in order
    :rtype: dict
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    messages = {topic: [] for topic in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        messages[topic].append((stamp, msg))
    return messages


def read_trial_yaml(path):
    """Return the trial.yaml the director writes into each bag folder."""
    with open(os.path.join(path, 'trial.yaml')) as handle:
        return yaml.safe_load(handle)
