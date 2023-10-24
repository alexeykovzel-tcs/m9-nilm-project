from signal import Signal


class EventDetector:
    def __init__(self, signal: Signal):
        self.signal = signal

    def run(self):
        times = self.signal.times
        return [times[0], times[1]]
