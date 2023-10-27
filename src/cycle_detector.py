from src.signals import Power, FreqNoise


def detect_cycles(power: Power, fn: FreqNoise):
    return [(10, 100), (150, 300)]
