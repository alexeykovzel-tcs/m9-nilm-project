from src.cycle_detector import detect_cycles
from src.cycle_profiler import profile_cycle
from src.meter_data import MeterData


# def profile_tagged_appliances(data_per_day: [MeterData]):
#     for data in data_per_day:
#         for tag, (cycle, phase) in _detect_tag_cycles(data):
#             # profile_cycle()
#             # Implement this.
#             pass
#         pass
#     return []


def _detect_tag_cycles(data: MeterData):
    return [(tag, _detect_tag_cycle(data, tag)) for tag in data.tags]


def _detect_tag_cycle(data: MeterData, tag):
    # detect cycles for each power phase
    cycles = [detect_cycles(p, data.hf) for p in data.powers()]

    # match those cycles with the tag ON and OFF times
    matches = [(_match_tag_cycle(c, tag), i) for i, c in enumerate(cycles)]

    # find a cycle and its phase with the best match
    (best_match, phase) = max(matches, key=lambda x: x[0][0])
    return best_match[1], phase


def _match_tag_cycle(cycles, tag):
    on_time, off_time = tag[2], tag[3]

    # increase the off time, as appliances don't turn off immediately
    real_off_time = off_time + (off_time - on_time) * 0.5
    real_uptime = (on_time, real_off_time)

    # find a cycle that best matches the appliance real uptime
    matches = [(_range_similarity(cycle, real_uptime), cycle) for cycle in cycles]
    best_match = max(matches, key=lambda x: x[0])
    return best_match


def _range_similarity(r1, r2):
    overlap_start = max(r1[0], r2[0])
    overlap_stop = min(r1[1], r2[1])
    overlap_length = max(0, overlap_stop - overlap_start)
    combined_length = (r1[1] - r1[0]) + (r2[1] - r2[0]) - overlap_length
    similarity = overlap_length / combined_length if combined_length > 0 else 0
    return similarity
