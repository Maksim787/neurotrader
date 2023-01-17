import datetime

local_tz = datetime.timezone(offset=datetime.timedelta(hours=3))  # Russia/Moscow


def utc_to_local(time: datetime.datetime) -> datetime.datetime:
    assert time.tzinfo == datetime.timezone.utc, f'incorrect {time} with {time.tzinfo=}'
    return time.astimezone(local_tz)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def local_now() -> datetime.datetime:
    return datetime.datetime.now(local_tz)
