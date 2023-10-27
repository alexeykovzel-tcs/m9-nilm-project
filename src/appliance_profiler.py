from src.cycle_detector import detect_cycles
from src.meter_data import MeterData
from src.appliance import Appliance


def profile_appliances(data: MeterData):
    appliances = []

    for tag in data.tags:
        phase, cycle = _detect_cycle(data, tag)

        if cycle is not None:
            appliance = Appliance.profile(data, tag, cycle, phase)
            appliances.append(appliance)

    return appliances


def _detect_cycle(data: MeterData, tag):
    # detect cycles for each power phase
    l1_cycles = detect_cycles(data.l1, data.hf)
    l2_cycles = detect_cycles(data.l2, data.hf)

    # match those cycles with the tag ON and OFF times
    l1_cycle, l1_match = _match_cycle(l1_cycles, tag)
    l2_cycle, l2_match = _match_cycle(l2_cycles, tag)

    # find a cycle and its phase with the best match
    return ('l1', l1_cycle) if l1_match > l2_match else ('l2', l2_cycle)


def _match_cycle(cycles, tag):
    if not cycles:
        return None, -1

    on_time, off_time = tag[2], tag[3]

    # increase the off time, as appliances don't turn off immediately
    real_off_time = off_time + (off_time - on_time) * 0.5
    real_uptime = (on_time, real_off_time)

    # find a cycle that best matches the appliance real uptime
    cycle_similarities = [(cycle, _range_similarity(cycle, real_uptime)) for cycle in cycles]

    return max(cycle_similarities, key=lambda x: x[1])


def _range_similarity(r1, r2):
    overlap_length = max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))
    combined_length = (r1[1] - r1[0]) + (r2[1] - r2[0]) - overlap_length
    return overlap_length / combined_length if combined_length > 0 else 0
