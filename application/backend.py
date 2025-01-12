from fastapi import FastAPI, HTTPException
from backend.data_types import (
    StartSessionRequest,
    StartSessionResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    AddPointsRequest,
    AddMaskRequest,
    ClearPointsInFrameRequest,
    ClearPointsInVideoRequest,
    ClearPointsInVideoResponse,
    RemoveObjectRequest,
    RemoveObjectResponse,
    PropagateInVideoRequest,
    PropagateDataResponse,
    CancelPropagateInVideoRequest,
    CancelPorpagateResponse,
)
from backend.inference import InferenceAPI
from typing import Dict
from uuid import uuid4
app = FastAPI()

# Global session store in server memory
sessions: Dict[str, dict] = {}


# Instantiate the InferenceAPI class
inference_api = InferenceAPI()



# Dependency to get session data
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.post("/start_session", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    try:
        # Generate a new session ID
        session_id = str(uuid4())
        # Initialize session data
        sessions[session_id] = {"data": {}, "inference_api": inference_api.start_session(request)}
        return StartSessionResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close_session", response_model=CloseSessionResponse)
async def close_session(request: CloseSessionRequest):
    try:
        session_id = request.session_id
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        # Close the session in inference_api
        response = inference_api.close_session(request)
        # Remove session from the store
        del sessions[session_id]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_points", response_model=PropagateDataResponse)
async def add_points(request: AddPointsRequest):
    try:
        return inference_api.add_points(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_mask", response_model=PropagateDataResponse)
async def add_mask(request: AddMaskRequest):
    try:
        return inference_api.add_mask(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_points_in_frame", response_model=PropagateDataResponse)
async def clear_points_in_frame(request: ClearPointsInFrameRequest):
    try:
        return inference_api.clear_points_in_frame(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_points_in_video", response_model=ClearPointsInVideoResponse)
async def clear_points_in_video(request: ClearPointsInVideoRequest):
    try:
        return inference_api.clear_points_in_video(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_object", response_model=RemoveObjectResponse)
async def remove_object(request: RemoveObjectRequest):
    try:
        return inference_api.remove_object(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/propagate_in_video", response_model=PropagateDataResponse)
async def propagate_in_video(request: PropagateInVideoRequest):
    try:
        # Use a generator to stream results
        return inference_api.propagate_in_video(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel_propagate_in_video", response_model=CancelPorpagateResponse)
async def cancel_propagate_in_video(request: CancelPropagateInVideoRequest):
    try:
        return inference_api.cancel_propagate_in_video(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
