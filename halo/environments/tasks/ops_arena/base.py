"""The abstract base of the Ops Arena environments: one support chat per episode, a simulated customer behind
``message_customer``, per-episode state kept off the trajectory, and a grade read from a subclass's ``assess``."""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from typing import Any

from halo.environments.episodes.episode_scoring import EpisodeGrade
from halo.environments.episodes.messages import Trajectory
from halo.environments.protocols.native_tool_use import AsyncNativeToolUseEnvironment
from halo.environments.tasks.ops_arena.agents import END_TOKEN, CustomerAgent
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter


class ChatEpisode:
    """State the tool handlers of one episode share; subclasses add their backend."""

    def __init__(self, scenario: dict[str, Any]):
        self.scenario = scenario
        self.chat: list[dict[str, str]] = [{"role": "customer", "text": scenario["customer"]["opening"]}]
        self.customer_left = False
        self.lock = asyncio.Lock()


class SupportChatEnvironmentBase(AsyncNativeToolUseEnvironment):
    DEFAULT_MAX_TURNS = 20
    # Paying per call would make a lookup spree worth more than resolving the case; errors still cost a little.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.0
    SYSTEM_PROMPT: str

    def __init__(
        self,
        customer_backend: str = "scripted",
        customer_model: str | None = None,
        customer_base_url: str | None = None,
        customer_api_key_env: str = "OPENROUTER_API_KEY",
        customer_temperature: float = 0.7,
        max_customer_turns: int = 8,
        **kwargs,
    ):
        self.customer = CustomerAgent(
            backend=customer_backend,
            model=customer_model,
            base_url=customer_base_url,
            api_key_env=customer_api_key_env,
            temperature=customer_temperature,
            max_customer_turns=max_customer_turns,
        )
        self._episodes_state: dict[int, ChatEpisode] = {}
        kwargs.setdefault("system_prompt", self.SYSTEM_PROMPT)
        registry = self._build_tools()
        registry.register(self._message_customer_tool())
        super().__init__(tool_registry=registry, **kwargs)

    def verify_backend(self) -> None:
        super().verify_backend()
        asyncio.run(self.customer.verify_backend())

    @abstractmethod
    def assess(self, trajectory: Trajectory) -> dict[str, Any]:
        """The graded episode: at least ``objective``, ``success``, ``scenario_id``, ``family``, ``split``, ``cue``
        and ``chat``. Public so evaluation drivers read the same grade training does."""

    @abstractmethod
    def _build_episode(self, scenario: dict[str, Any]) -> ChatEpisode: ...

    @abstractmethod
    def _build_tools(self) -> NativeToolRegistry:
        """Every tool but ``message_customer``, which the base adds."""

    @abstractmethod
    def _persona(self, state: ChatEpisode) -> tuple[str, list[str]]:
        """The customer's LLM system prompt and its scripted turns."""

    @abstractmethod
    def _shaping(self, result: dict[str, Any]) -> dict[str, float]: ...

    def _reset_single(self, prompt, context=None) -> Trajectory:
        ctx = context or {}
        raw = ctx.get("scenario")
        if raw is None:
            raise ValueError(f"{type(self).__name__} needs a 'scenario' column passed through context_fields")
        scenario = json.loads(raw) if isinstance(raw, str) else raw
        trajectory = super()._reset_single(prompt, ctx)
        self._episodes_state[id(trajectory)] = self._build_episode(scenario)
        trajectory.info["slices"] = {"family": scenario["family"], "cue": scenario["cue"]}
        trajectory.info["ops_scenario_id"] = scenario["scenario_id"]
        return trajectory

    def _release_episode(self, episode_id: int) -> None:
        trajectory = self._trajectories.get(episode_id)
        if trajectory is not None:
            self._episodes_state.pop(id(trajectory), None)

    def _state(self) -> ChatEpisode:
        trajectory = self.active_trajectory()
        if trajectory is None or id(trajectory) not in self._episodes_state:
            raise RuntimeError(f"{type(self).__name__} tool called outside an episode")
        return self._episodes_state[id(trajectory)]

    def _message_customer_tool(self) -> NativeTool:
        async def message_customer(message: str) -> str:
            st = self._state()
            if st.customer_left:
                return "The customer has left the chat. Write your final summary."
            st.chat.append({"role": "agent", "text": message})
            persona_prompt, script = self._persona(st)
            reply = await self.customer.reply(persona_prompt, script, st.chat)
            left = END_TOKEN in reply
            reply = reply.replace(END_TOKEN, "").strip()
            st.chat.append({"role": "customer", "text": reply})
            if left:
                st.customer_left = True
                return f"Customer: {reply}\n(The customer has left the chat. Write your final summary.)"
            return f"Customer: {reply}"

        return NativeTool(
            name="message_customer",
            description="Send a chat message to the customer and wait for their reply.",
            parameters=[ToolParameter(name="message", type="string", description="Your message")],
            async_handler=message_customer,
        )

    def _grade_episode(self, trajectory: Trajectory, context=None) -> EpisodeGrade:
        result = self.assess(trajectory)
        trajectory.info["ops"] = result
        return EpisodeGrade(result["objective"], shaping=self._shaping(result))
