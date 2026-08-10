import json
from typing import Any, Dict

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from app.websocket.manager import dashboard_manager
from app.core.security import decode_token

# Re-export dashboard_manager so it can be imported from here as well
__all__ = ["router", "dashboard_manager"]

router = APIRouter()


@router.get("/ws/dashboard/contract", response_model=Dict[str, Any])
async def dashboard_ws_contract():
    return {
        "version": 1,
        "auth": {
            "type": "queryToken",
            "param": "token",
            "description": "Use the access token from /api/v1/auth/login.",
        },
        "endpoints": {
            "dashboard": "/api/v1/ws/dashboard?token=<accessToken>",
            "cameraStream": "/api/v1/ws/camera/{cameraId}/stream?token=<accessToken>",
            "dashboardByCamera": "/api/v1/ws/dashboard?cameraId={cameraId}&token=<accessToken>",
        },
        "clientMessages": [
            {"type": "ping"},
            {"type": "switch_camera", "cameraId": "<cameraId>"},
        ],
        "serverMessages": {
            "pong": {"type": "pong"},
            "violationAlert": {
                "type": "violation_alert",
                "version": 1,
                "priority": "high",
                "data": {
                    "id": "string",
                    "cameraId": "string",
                    "cameraName": "string",
                    "violationType": "speeding",
                    "vehicleType": "string",
                    "licensePlate": "string|null",
                    "confidence": "number",
                    "evidenceUrl": "string",
                    "status": "pending|verified|rejected",
                    "createdAt": "ISO-8601 datetime",
                },
            },
        },
    }

@router.websocket("/ws/dashboard")
async def dashboard_ws(
    websocket: WebSocket,
    token: str = Query(default=None),
    camera_id: str = Query(default=None, alias="cameraId")
):
    """WebSocket for monitoring dashboards."""
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return
        
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await dashboard_manager.connect(websocket, camera_filter=camera_id)

    try:
        while True:
            # Keep connection alive and listen for client commands
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                
                if command.get("type") == "switch_camera":
                    new_camera = command.get("camera_id") or command.get("cameraId")
                    dashboard_manager.disconnect(websocket)
                    await dashboard_manager.connect(websocket, camera_filter=new_camera)
                    
                elif command.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except ValueError:
                pass
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)

@router.websocket("/ws/camera/{camera_id}/stream")
async def camera_stream_ws(
    websocket: WebSocket,
    camera_id: str,
    token: str = Query(default=None)
):
    """WebSocket stream of detections for a single camera."""
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return
        
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await dashboard_manager.connect(websocket, camera_filter=camera_id)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)
