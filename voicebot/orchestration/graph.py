from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Awaitable, Callable, TypedDict

from voicebot.domain.interfaces import AudioSink, ChatClient, Chunker, TTSProvider
from voicebot.domain.models import ChatTurn, SessionState

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


OnDelta = Callable[[str], Awaitable[None] | None]


class GraphState(TypedDict):
    user_text: str
    session: SessionState
    chunker: Chunker
    on_delta: OnDelta | None
    assistant_text: str
    tts_text_chunks: list[str]
    tts_queue: asyncio.Queue[str | None]
    synthesis_task: asyncio.Task[None] | None
    first_token_at: float | None
    first_audio_at: float | None
    started_at: float


class ConversationOrchestrator:
    def __init__(
        self,
        chat_client: ChatClient,
        tts_provider: TTSProvider,
        audio_sink: AudioSink,
    ) -> None:
        self.chat_client = chat_client
        self.tts_provider = tts_provider
        self.audio_sink = audio_sink
        self.using_langgraph = LANGGRAPH_AVAILABLE
        self._compiled = self._build_graph() if LANGGRAPH_AVAILABLE else None

    async def run(self, state: GraphState) -> GraphState:
        try:
            if self._compiled is None:
                return await self._run_without_langgraph(state)
            return await self._compiled.ainvoke(state)
        except Exception:
            await self._cancel_synthesis_task(state)
            raise

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("accept_input", self._accept_input)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("stream_llm", self._stream_llm)
        graph.add_node("chunk_text", self._chunk_text)
        graph.add_node("playback", self._playback)
        graph.add_node("persist_turn", self._persist_turn)

        graph.add_edge(START, "accept_input")
        graph.add_edge("accept_input", "synthesize")
        graph.add_edge("synthesize", "stream_llm")
        graph.add_edge("stream_llm", "chunk_text")
        graph.add_edge("chunk_text", "playback")
        graph.add_edge("playback", "persist_turn")
        graph.add_edge("persist_turn", END)
        return graph.compile()

    async def _run_without_langgraph(self, state: GraphState) -> GraphState:
        state = await self._accept_input(state)
        state = await self._synthesize(state)
        state = await self._stream_llm(state)
        state = await self._chunk_text(state)
        state = await self._playback(state)
        state = await self._persist_turn(state)
        return state

    async def _accept_input(self, state: GraphState) -> GraphState:
        state["started_at"] = perf_counter()
        state["assistant_text"] = ""
        state["tts_text_chunks"] = []
        state["tts_queue"] = asyncio.Queue(maxsize=8)
        state["synthesis_task"] = None
        state["first_token_at"] = None
        state["first_audio_at"] = None
        return state

    async def _synthesize(self, state: GraphState) -> GraphState:
        async def _worker() -> None:
            while True:
                text_chunk = await state["tts_queue"].get()
                try:
                    if text_chunk is None:
                        return
                    async for audio_chunk in self.tts_provider.synthesize_stream(text_chunk):
                        if state["first_audio_at"] is None:
                            state["first_audio_at"] = perf_counter()
                        self.audio_sink.enqueue(audio_chunk)
                finally:
                    state["tts_queue"].task_done()

        state["synthesis_task"] = asyncio.create_task(_worker(), name="voicebot_synthesis")
        return state

    async def _stream_llm(self, state: GraphState) -> GraphState:
        parts: list[str] = []
        on_delta = state.get("on_delta")
        async for delta in self.chat_client.stream_reply(state["user_text"], state["session"].history):
            if state["first_token_at"] is None:
                state["first_token_at"] = perf_counter()
            parts.append(delta)
            chunked = state["chunker"].feed(delta)
            for piece in chunked:
                state["tts_text_chunks"].append(piece)
                await state["tts_queue"].put(piece)
            if on_delta is not None:
                maybe_coro = on_delta(delta)
                if maybe_coro is not None:
                    await maybe_coro
        state["assistant_text"] = "".join(parts).strip()
        if not state["assistant_text"]:
            raise RuntimeError("LLM returned an empty streamed response.")
        return state

    async def _chunk_text(self, state: GraphState) -> GraphState:
        tail = state["chunker"].flush()
        if tail:
            state["tts_text_chunks"].append(tail)
            await state["tts_queue"].put(tail)
        await state["tts_queue"].put(None)
        return state

    async def _playback(self, state: GraphState) -> GraphState:
        task = state.get("synthesis_task")
        if task is not None:
            await task
        self.audio_sink.wait_until_idle()
        return state

    async def _persist_turn(self, state: GraphState) -> GraphState:
        state["session"].history.append(
            ChatTurn(
                user_text=state["user_text"],
                assistant_text=state["assistant_text"],
            )
        )
        return state

    async def _cancel_synthesis_task(self, state: GraphState) -> None:
        task = state.get("synthesis_task")
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
