import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.post('http://127.0.0.1:8000/reset', timeout=10.0)
        print("RESET:", resp.status_code, resp.text)
        resp2 = await client.post('http://127.0.0.1:8000/step', json={'action': {'signal': 1}}, timeout=10.0)
        print("STEP:", resp2.status_code, resp2.text)

asyncio.run(test())
