import uuid


def derive_point_id(mongo_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, mongo_id))