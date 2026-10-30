from dataclasses import dataclass

from app.schemas.contracts import Decision


@dataclass(frozen=True)
class ScenarioEvent:
    frame_index: int
    timestamp_ms: int
    action_id: str
    action_label: str
    confidence: float
    decision: Decision
    message: str | None


SCENARIOS: dict[str, list[ScenarioEvent]] = {
    "cup_interaction": [
        ScenarioEvent(0, 0, "observing_workspace", "Đang quan sát khu vực làm việc", 0.58, Decision.WAIT, None),
        ScenarioEvent(1, 1200, "reaching_for_cup", "Đang đưa tay về phía cốc", 0.66, Decision.WAIT, None),
        ScenarioEvent(2, 2400, "picking_up_cup", "Đang nhấc chiếc cốc", 0.87, Decision.SPEAK, "Đang nhấc chiếc cốc."),
        ScenarioEvent(3, 3600, "moving_cup", "Đưa cốc về phía bàn", 0.82, Decision.UPDATE, "Cập nhật: đang đưa cốc về phía bàn."),
    ],
    "door_interaction": [
        ScenarioEvent(0, 0, "approaching_door", "Đang tiến gần cửa", 0.61, Decision.WAIT, None),
        ScenarioEvent(1, 1500, "opening_door", "Đang mở cửa", 0.9, Decision.SPEAK, "Đang mở cửa."),
    ],
    "uncertain_action": [
        ScenarioEvent(0, 0, "uncertain_motion", "Chuyển động chưa rõ", 0.34, Decision.WAIT, None),
        ScenarioEvent(1, 1000, "uncertain_motion", "Tiếp tục quan sát hành động chưa rõ", 0.41, Decision.WAIT, None),
        ScenarioEvent(2, 2000, "reaching_object", "Có thể đang với tới đồ vật", 0.55, Decision.WAIT, None),
        ScenarioEvent(3, 3200, "placing_object", "Đang đặt đồ vật xuống", 0.81, Decision.SPEAK, "Đang đặt đồ vật xuống."),
    ],
}


def scenario_ids() -> list[str]:
    return sorted(SCENARIOS)

