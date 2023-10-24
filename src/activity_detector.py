from signals import Signal


def detect_activities(signals: [Signal]):
    return []


# align event pairs if there are matches across multiple signals
def _align_activities(signals: [Signal], event_data: [[int]]):
    return []


# pair ON/OFF events as appliance activities
def _find_activities(signal: Signal, events: [int]):
    return []


# detects extreme changes in the signal (potentially appliance ON/OFF)
def _detect_events(signal: Signal):
    return []
