#!/usr/bin/env python3
"""End-to-end smoke test for the WebRTC transport and online session API."""

from __future__ import annotations

import asyncio
import argparse
import json

import aiohttp
import av
import numpy as np
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection, VideoStreamTrack

import server


class SyntheticVideoTrack(VideoStreamTrack):
    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        shade = int((pts / 3000) % 255)
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[:, :, 0] = shade
        image[:, :, 1] = 120
        image[:, :, 2] = 255 - shade
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def run(real_model: bool) -> None:
    server.MOCK_MODEL = not real_model
    runner = web.AppRunner(server.build_app())
    await runner.setup()
    port = 17861
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    base_url = f"http://127.0.0.1:{port}"
    peer = RTCPeerConnection()
    channel = peer.createDataChannel("telemetry")
    peer.addTrack(SyntheticVideoTrack())

    try:
        async with aiohttp.ClientSession() as client:
            response = await client.post(
                f"{base_url}/api/online/sessions",
                json={
                    "prompt": "Describe the observed colors.",
                    "gpu": 0,
                    "window_seconds": 4,
                    "interval_seconds": 4,
                    "max_new_tokens": 32,
                    "auto_query": True,
                },
            )
            response.raise_for_status()
            session = await response.json()

            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            response = await client.post(
                f"{base_url}/api/rtc/offer",
                json={
                    "session_id": session["id"],
                    "sdp": peer.localDescription.sdp,
                    "type": peer.localDescription.type,
                },
            )
            response.raise_for_status()
            answer = await response.json()
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )

            attempts = 600 if real_model else 100
            for _ in range(attempts):
                if channel.readyState == "open":
                    channel.send(
                        json.dumps(
                            {
                                "type": "telemetry",
                                "sent_at": server.now_ms(),
                                "duration": 8,
                                "fps": 30,
                                "bitrate_kbps": 500,
                                "packets_sent": 100,
                            }
                        )
                    )
                await asyncio.sleep(0.1)
                response = await client.get(
                    f"{base_url}/api/online/sessions/{session['id']}"
                )
                response.raise_for_status()
                status = await response.json()
                if status["responses"]:
                    break

            assert status["transport_state"] == "connected", status
            assert status["metrics"]["received_frames"] > 30, status
            assert status["metrics"]["buffered_frames"] >= 12, status
            assert status["responses"], status
            assert status["responses"][0]["media_end"] >= 4, status
            preview = await client.get(
                f"{base_url}/api/online/sessions/{session['id']}/preview.jpg"
            )
            preview.raise_for_status()
            preview_body = await preview.read()
            assert preview.content_type == "image/jpeg", preview.content_type
            assert preview_body.startswith(b"\xff\xd8"), "invalid JPEG preview"
            print(
                json.dumps(
                    {
                        "session": status["id"],
                        "transport": status["transport_state"],
                        "received_frames": status["metrics"]["received_frames"],
                        "buffered_frames": status["metrics"]["buffered_frames"],
                        "responses": len(status["responses"]),
                        "latency_ms": status["responses"][0][
                            "total_latency_ms"
                        ],
                        "preview_bytes": len(preview_body),
                        "mode": "real" if real_model else "mock",
                        "answer": status["responses"][0]["answer"],
                    },
                    ensure_ascii=False,
                )
            )
            await client.post(
                f"{base_url}/api/online/sessions/{session['id']}/stop"
            )
    finally:
        await peer.close()
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-model", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.real_model))
