# mcp_bus.py

import asyncio
from typing import Callable, Dict, Any

class MCPBus:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.subscribers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def subscribe(self, agent_name: str, handler: Callable):
        self.subscribers[agent_name] = handler

    async def send(self, message: Dict[str, Any]):
        await self.queue.put(message)

    async def run(self):
        while True:
            message = await self.queue.get()
            receiver = message.get("receiver")
            if receiver in self.subscribers:
                await self.subscribers[receiver](message)
            else:
                print(f"[MCP] No subscriber for '{receiver}' — message dropped.")