from typing import Dict, List, Optional, Union
from pydantic import BaseModel


class Mask(BaseModel):
    size: List[int]
    counts: str


class BaseRequest(BaseModel):
    type: str


class StartSessionRequest(BaseRequest):
    path: str
    session_id: Optional[str] = None


class SaveSessionRequest(BaseRequest):
    session_id: str


class LoadSessionRequest(BaseRequest):
    session_id: str


class RenewSessionRequest(BaseRequest):
    session_id: str


class CloseSessionRequest(BaseRequest):
    session_id: str


class AddPointsRequest(BaseRequest):
    session_id: str
    frame_index: int
    clear_old_points: bool
    object_id: int
    labels: List[int]
    points: List[List[float]]


class AddMaskRequest(BaseRequest):
    session_id: str
    frame_index: int
    object_id: int
    mask: Mask


class ClearPointsInFrameRequest(BaseRequest):
    session_id: str
    frame_index: int
    object_id: int


class ClearPointsInVideoRequest(BaseRequest):
    session_id: str


class RemoveObjectRequest(BaseRequest):
    session_id: str
    object_id: int


class PropagateInVideoRequest(BaseRequest):
    session_id: str
    start_frame_index: int


class CancelPropagateInVideoRequest(BaseRequest):
    session_id: str


class StartSessionResponse(BaseModel):
    session_id: str


class SaveSessionResponse(BaseModel):
    session_id: str


class LoadSessionResponse(BaseModel):
    session_id: str


class RenewSessionResponse(BaseModel):
    session_id: str


class CloseSessionResponse(BaseModel):
    success: bool


class ClearPointsInVideoResponse(BaseModel):
    success: bool


class PropagateDataValue(BaseModel):
    object_id: int
    mask: Mask


class PropagateDataResponse(BaseModel):
    frame_index: int
    results: List[PropagateDataValue]


class RemoveObjectResponse(BaseModel):
    results: List[PropagateDataResponse]


class CancelPorpagateResponse(BaseModel):
    success: bool


class InferenceSession(BaseModel):
    start_time: float
    last_use_time: float
    session_id: str
    state: Dict[str, Dict[str, Union[Dict[int, List[float]], List[float]]]]  # Adjust Tensor to appropriate structure
