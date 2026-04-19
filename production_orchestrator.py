import logging
from typing import Dict, Any, Optional

from unified_context_engine_master import UnifiedContextEngine
from visual_storyboard_engine import VisualStoryboardEngine
from rhythmic_assembly_engine import RhythmicAssemblyEngine
from style_grading_engine import StyleGradingEngine

logger = logging.getLogger(__name__)


class ProductionOrchestrator:
    """
    Qaivid Production Orchestrator
    """

    def __init__(self, openai_api_key: str):
        self.context_engine = UnifiedContextEngine(openai_api_key)
        self.storyboard_engine = VisualStoryboardEngine()
        self.assembly_engine = RhythmicAssemblyEngine()
        self.style_engine = StyleGradingEngine()

    async def run_full_production(
        self,
        text: str,
        genre: str,
        audio_analytics: Optional[Dict[str, Any]] = None,
        user_image_url: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
        project_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = self._validate_text(text)
        genre = self._validate_genre(genre)
        audio_analytics = self._validate_optional_dict(audio_analytics)
        pre_analysis = self._validate_optional_dict(pre_analysis)
        style_profile = self._validate_optional_dict(style_profile)
        project_meta = self._validate_optional_dict(project_meta)

        logger.info("Starting full Qaivid production for genre=%s", genre)

        if user_image_url:
            self.storyboard_engine.inject_user_reference(user_image_url)

        context_packet = await self.context_engine.generate(
            text=text,
            genre=genre,
            pre_analysis=pre_analysis,
            style_profile=style_profile,
        )
        self._assert_non_empty(context_packet, "Context engine returned empty output.")

        storyboard = self.storyboard_engine.build_storyboard(
            context_packet, style_profile=style_profile
        )
        self._assert_non_empty(storyboard, "Storyboard engine returned empty output.")

        timeline = self.assembly_engine.assemble_timeline(
            storyboard=storyboard,
            audio_data=audio_analytics,
        )
        self._assert_non_empty(timeline, "Rhythmic assembly engine returned empty output.")

        styled_timeline = self.style_engine.apply_style(
            timeline=timeline,
            style_profile=style_profile,
        )
        self._assert_non_empty(styled_timeline, "Style grading engine returned empty output.")

        logger.info("Qaivid production complete.")

        return self._build_production_package(
            text=text,
            genre=genre,
            audio_analytics=audio_analytics,
            pre_analysis=pre_analysis,
            style_profile=style_profile,
            user_image_url=user_image_url,
            project_meta=project_meta,
            context_packet=context_packet,
            storyboard=storyboard,
            timeline=timeline,
            styled_timeline=styled_timeline,
        )

    async def run_context_only(
        self,
        text: str,
        genre: str,
        pre_analysis: Optional[Dict[str, Any]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = self._validate_text(text)
        genre = self._validate_genre(genre)
        pre_analysis = self._validate_optional_dict(pre_analysis)
        style_profile = self._validate_optional_dict(style_profile)

        context_packet = await self.context_engine.generate(
            text=text,
            genre=genre,
            pre_analysis=pre_analysis,
            style_profile=style_profile,
        )
        self._assert_non_empty(context_packet, "Context engine returned empty output.")
        return context_packet

    async def run_to_storyboard(
        self,
        text: str,
        genre: str,
        user_image_url: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = self._validate_text(text)
        genre = self._validate_genre(genre)
        pre_analysis = self._validate_optional_dict(pre_analysis)

        if user_image_url:
            self.storyboard_engine.inject_user_reference(user_image_url)

        context_packet = await self.context_engine.generate(
            text=text,
            genre=genre,
            pre_analysis=pre_analysis,
        )
        self._assert_non_empty(context_packet, "Context engine returned empty output.")

        storyboard = self.storyboard_engine.build_storyboard(context_packet)
        self._assert_non_empty(storyboard, "Storyboard engine returned empty output.")

        return {
            "context_packet": context_packet,
            "storyboard": storyboard,
        }

    async def run_to_timeline(
        self,
        text: str,
        genre: str,
        audio_analytics: Optional[Dict[str, Any]] = None,
        user_image_url: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = self._validate_text(text)
        genre = self._validate_genre(genre)
        audio_analytics = self._validate_optional_dict(audio_analytics)
        pre_analysis = self._validate_optional_dict(pre_analysis)
        style_profile = self._validate_optional_dict(style_profile)

        if user_image_url:
            self.storyboard_engine.inject_user_reference(user_image_url)

        context_packet = await self.context_engine.generate(
            text=text,
            genre=genre,
            pre_analysis=pre_analysis,
            style_profile=style_profile,
        )
        self._assert_non_empty(context_packet, "Context engine returned empty output.")

        storyboard = self.storyboard_engine.build_storyboard(
            context_packet, style_profile=style_profile
        )
        self._assert_non_empty(storyboard, "Storyboard engine returned empty output.")

        timeline = self.assembly_engine.assemble_timeline(
            storyboard=storyboard,
            audio_data=audio_analytics,
        )
        self._assert_non_empty(timeline, "Rhythmic assembly engine returned empty output.")

        return {
            "context_packet": context_packet,
            "storyboard": storyboard,
            "timeline": timeline,
        }

    async def run_to_style(
        self,
        text: str,
        genre: str,
        audio_analytics: Optional[Dict[str, Any]] = None,
        user_image_url: Optional[str] = None,
        pre_analysis: Optional[Dict[str, Any]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.run_full_production(
            text=text,
            genre=genre,
            audio_analytics=audio_analytics,
            user_image_url=user_image_url,
            pre_analysis=pre_analysis,
            style_profile=style_profile,
        )

    def _build_production_package(
        self,
        text: str,
        genre: str,
        audio_analytics: Dict[str, Any],
        pre_analysis: Dict[str, Any],
        style_profile: Dict[str, Any],
        user_image_url: Optional[str],
        project_meta: Dict[str, Any],
        context_packet: Dict[str, Any],
        storyboard: list,
        timeline: list,
        styled_timeline: list,
    ) -> Dict[str, Any]:
        return {
            "meta": {
                "workflow_stage": "full_preproduction",
                "engine_stack": [
                    "UnifiedContextEngine",
                    "VisualStoryboardEngine",
                    "RhythmicAssemblyEngine",
                    "StyleGradingEngine",
                ],
                "project_meta": project_meta,
            },
            "inputs": {
                "text": text,
                "genre": genre,
                "audio_analytics": audio_analytics,
                "pre_analysis": pre_analysis,
                "style_profile": style_profile,
                "user_image_url": user_image_url,
            },
            "context_packet": context_packet,
            "storyboard": storyboard,
            "timeline": timeline,
            "styled_timeline": styled_timeline,
            "summary": {
                "storyboard_shot_count": len(storyboard),
                "timeline_shot_count": len(timeline),
                "styled_timeline_shot_count": len(styled_timeline),
                "total_duration": round(sum(x.get("duration", 0.0) for x in timeline), 3),
            },
        }

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Text must be a string.")
        return text.strip()

    def _validate_genre(self, genre: str) -> str:
        if not isinstance(genre, str) or not genre.strip():
            raise ValueError("Genre must be a non-empty string.")
        return genre.strip()

    def _validate_optional_dict(self, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Optional structured inputs must be dictionaries.")
        return value

    def _assert_non_empty(self, value: Any, message: str) -> None:
        if value is None:
            raise ValueError(message)
        if isinstance(value, (list, dict)) and len(value) == 0:
            raise ValueError(message)
