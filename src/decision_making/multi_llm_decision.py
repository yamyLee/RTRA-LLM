import os
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod

try:
    from langchain_openai import ChatOpenAI
    CHAT_OPENAI_AVAILABLE = True
except ImportError:
    CHAT_OPENAI_AVAILABLE = False

@dataclass
class VesselState:
    """Represents the state of a vessel encounter"""
    risk: float
    distance: float
    bearing: float
    dcpa: float
    tcpa: float

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from the LLM"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass


class ChatOpenAIProvider(LLMProvider):
    """Base provider for OpenAI-compatible endpoints."""

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: Optional[str] = None,
    ):
        self.provider_name = provider_name.lower()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.client = None

        if self.is_available():
            client_kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": api_key,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = ChatOpenAI(**client_kwargs)

    def is_available(self) -> bool:
        """Check if ChatOpenAI and the provider API key are available."""
        return CHAT_OPENAI_AVAILABLE and bool(self.api_key)

    def generate_response(self, prompt: str) -> str:
        """Generate response through an OpenAI-compatible client."""
        if not self.client:
            return f"{self.provider_name} not available"

        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"{self.provider_name} error: {str(e)}"


class OpenAIProvider(LLMProvider):
    """OpenAI provider using OPENAI_* environment variables."""

    def __init__(self):
        self.provider = ChatOpenAIProvider(
            provider_name="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def is_available(self) -> bool:
        return self.provider.is_available()

    def generate_response(self, prompt: str) -> str:
        return self.provider.generate_response(prompt)


class OtherProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible endpoint except OpenAI itself."""

    def __init__(self, provider_name: str):
        prefix = provider_name.upper()
        self.provider_name = provider_name.lower()
        self.provider = ChatOpenAIProvider(
            provider_name=self.provider_name,
            api_key=os.getenv(f"{prefix}_API_KEY"),
            model=os.getenv(f"{prefix}_MODEL", ""),
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "1000")),
            base_url=os.getenv(f"{prefix}_BASE_URL"),
        )

    def is_available(self) -> bool:
        return self.provider.is_available() and bool(self.provider.model)

    def generate_response(self, prompt: str) -> str:
        if not self.provider.model:
            return f"{self.provider_name} not configured: missing model"
        return self.provider.generate_response(prompt)

class MultiLLMCOLREGSInterpreter:
    """COLREGs interpreter that can use multiple LLM providers"""

    def __init__(self, provider: str = None):
        self.provider_name = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.provider = self._initialize_provider()
        self.system_prompt = self._load_system_prompt()

    def _get_prompt_file_path(self) -> str:
        """Return the absolute path to the unified prompt file."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "prompt", "prompt.txt")

    def _load_system_prompt(self) -> str:
        """Load system prompt from the unified offline prompt file."""
        default_prompt = """You are a ship navigation officer. Based on the situation, give a simple decision in this format:
Action: [Stand on / Give-way, turn to starboard / Give-way, turn to port]

Situation:"""

        try:
            prompt_path = self._get_prompt_file_path()

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            if prompt:
                return prompt

            print(f"Warning: Prompt file is empty: {prompt_path}")
            return default_prompt

        except Exception as e:
            print(f"Warning: Could not load system prompt from prompt/prompt.txt: {str(e)}")
            return default_prompt
    
    def _initialize_provider(self) -> Optional[LLMProvider]:
        """Initialize the requested provider without extra fallback chains."""
        provider = OpenAIProvider() if self.provider_name == "openai" else OtherProvider(self.provider_name)
        return provider if provider.is_available() else None
    
    def _format_bearing_degrees(self, bearing: float) -> float:
        """Normalize relative bearing to degrees for prompt display."""
        if abs(bearing) <= 2 * 3.141592653589793:
            bearing = bearing * 180.0 / 3.141592653589793
        return ((bearing + 180.0) % 360.0) - 180.0

    def _format_situation_description(
        self,
        vessels: List[VesselState],
        memory_context: Optional[str] = None,
        encounter_type: Optional[str] = None,
        key_vessel_index: Optional[int] = None,
    ) -> str:
        """Format situation description for LLM"""
        if not vessels:
            return "No vessels detected."

        if key_vessel_index is None:
            key_vessel_index = max(range(len(vessels)), key=lambda idx: vessels[idx].risk)
        highest_risk_vessel = vessels[key_vessel_index]

        # Add more context about the situation
        if highest_risk_vessel.tcpa < 0:
            time_status = "vessel is astern (already passed)"
        elif highest_risk_vessel.tcpa > 300:
            time_status = "vessel is far, no immediate action needed"
        else:
            time_status = f"vessel will reach CPA in {highest_risk_vessel.tcpa:.1f} seconds"

        vessel_lines = []
        for idx, vessel in enumerate(vessels, start=1):
            marker = " (key target)" if idx - 1 == key_vessel_index else ""
            bearing_deg = self._format_bearing_degrees(vessel.bearing)
            vessel_lines.append(
                f"- Target vessel {idx}{marker}: q={vessel.risk:.3f}, "
                f"R={vessel.distance:.2f} nmi, relative bearing={bearing_deg:.1f} deg, "
                f"d_CPA={vessel.dcpa:.2f} nmi, t_CPA={vessel.tcpa:.1f} s"
            )

        memory_block = f"\n{memory_context.strip()}\n" if memory_context else ""
        prompt_ablation_variant = os.getenv("CORALL_PROMPT_ABLATION_VARIANT")
        include_encounter_hint = prompt_ablation_variant is None
        encounter_block = (
            f"- Encounter type: {encounter_type}\n"
            if encounter_type and include_encounter_hint
            else ""
        )

        description = f"""
Maritime Situation Analysis:
- Number of vessels: {len(vessels)}
- Key target vessel: {key_vessel_index + 1}
{encounter_block}- Key target timing: {time_status}

Current vessel states:
{os.linesep.join(vessel_lines)}
{memory_block}

What action should the own ship take next?"""

        return description.strip()
    
    def make_decision(
        self,
        vessels: List[VesselState],
        time_idx: int = 0,
        memory_context: Optional[str] = None,
        encounter_type: Optional[str] = None,
        key_vessel_index: Optional[int] = None,
    ) -> str:
        """Make a COLREGs-compliant decision"""
        if not self.provider:
            return "No LLM provider available"

        if not vessels:
            return "No vessels detected - maintain course and speed"

        # Format the situation
        situation_description = self._format_situation_description(
            vessels,
            memory_context=memory_context,
            encounter_type=encounter_type,
            key_vessel_index=key_vessel_index,
        )

        # Create full prompt
        full_prompt = f"{self.system_prompt}\n\nDecision step: {time_idx}\n\n{situation_description}"

        if os.getenv("SHOW_LLM_DEBUG", "false").lower() == "true":
            print(f"\n[DEBUG] Full prompt sent to {self.provider_name.upper()}:")
            print("=" * 60)
            print(full_prompt)
            print("=" * 60)

        # Get response from LLM
        response = self.provider.generate_response(full_prompt)

        if os.getenv("SHOW_LLM_DEBUG", "false").lower() == "true":
            print(f"\n[DEBUG] Raw response received from {self.provider_name.upper()}:")
            print("=" * 60)
            print(repr(response))
            print("-" * 60)
            print(response)
            print("=" * 60)

        # Ensure response is properly formatted
        if not response or len(response.strip()) == 0:
            response = "Error: Empty response from LLM"

        # Add provider information
        provider_info = f"[{self.provider_name.upper()}] "

        return f"{provider_info}{response}"
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        providers = []

        if OpenAIProvider().is_available():
            providers.append("openai")

        for env_name in sorted(os.environ):
            if not env_name.endswith("_API_KEY") or env_name == "OPENAI_API_KEY":
                continue
            provider_name = env_name[:-8].lower()
            provider = OtherProvider(provider_name)
            if provider.is_available():
                providers.append(provider_name)

        return providers

# Backward compatibility with original interface
COLREGSInterpreter = MultiLLMCOLREGSInterpreter
