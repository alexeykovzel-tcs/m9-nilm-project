from src.meter_data import MeterData


def tag_activities(data: MeterData):
    return [(tag, _truncate_tag(data, tag[2], tag[3])) for tag in data.tags]


def _truncate_tag(data: MeterData, tag_start, tag_stop):
    return []
