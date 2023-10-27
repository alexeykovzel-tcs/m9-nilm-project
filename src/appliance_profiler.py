from src.feature_extractor import extract_power_features
from src.cycle_detector import detect_cycles
from src.meter_data import MeterData
from src.appliance import Appliance


def profile_appliances(data: MeterData):
    tag_powers = [(tag, _truncate_power_cycle(data, tag[2], tag[3])) for tag in data.tags]
    tag_features = [(tag, extract_power_features(power)) for tag, power in tag_powers if power is not None]
    return [Appliance(tag[0], tag[1], features) for tag, features in tag_features]


def _truncate_power_cycle(data: MeterData, on_time, off_time):
    # match cycles for each power phase with ON and OFF times
    l1_cycle, l1_match = _match_cycle(detect_cycles(data.l1), on_time, off_time)
    l2_cycle, l2_match = _match_cycle(detect_cycles(data.l2), on_time, off_time)

    # find and extract a power cycle with the best match score
    if l1_cycle is None and l2_cycle is None: return None
    return data.l1.truncate(l1_cycle) if l1_match > l2_match \
        else data.l2.truncate(l2_cycle)


def _match_cycle(cycles, on_time, off_time):
    if not cycles: return None, -1

    # increase the off time, as appliances don't turn off immediately
    real_off_time = off_time + (off_time - on_time) * 0.5
    real_uptime = (on_time, real_off_time)

    # find a cycle that best matches the appliance real uptime
    cycle_overlaps = [(cycle, _overlap_score(cycle, real_uptime)) for cycle in cycles]
    return max(cycle_overlaps, key=lambda x: x[1])


def _overlap_score(r1, r2):
    overlap_len = max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))
    combined_len = (r1[1] - r1[0]) + (r2[1] - r2[0]) - overlap_len
    return overlap_len / combined_len if combined_len > 0 else 0
