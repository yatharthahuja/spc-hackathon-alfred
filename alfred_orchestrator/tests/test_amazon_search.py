from __future__ import annotations

from dataclasses import replace

from app.config import Settings
from app.integrations.amazon.mock_client import MockAmazonSearchClient
from app.orchestrator.prompt_registry import PromptRegistry
from app.orchestrator.skill_planner import CatalogSkillPlanner
from app.orchestrator.task_registry import SkillCatalog
from app.skills.amazon_search import AmazonSearchSkill, extract_product_query


def test_extract_product_query_strips_filler():
    assert extract_product_query("Can you order apples on Amazon? Yeah, sure.") == "apples"


def test_amazon_search_skill_reports_top_result():
    skill = AmazonSearchSkill(
        replace(Settings.load(), amazon_use_mock=True),
        client=MockAmazonSearchClient(),
    )
    result = skill.run(query="apples", user_text="order apples on Amazon")

    assert result.status == "success"
    assert "Apples" in result.output["title"]
    assert result.output["answer_text"]
    assert "amazon.com" in result.output["product_url"]


def test_planner_selects_amazon_search_for_order_request():
    settings = replace(Settings.load(), openai_api_key="")
    planner = CatalogSkillPlanner(
        settings,
        SkillCatalog(settings.configs_dir / "skills.yaml"),
        PromptRegistry(settings.configs_dir / "prompts.yaml"),
    )

    choice = planner.choose_skill("Can you order apples on Amazon? Yeah, sure.")

    assert choice.skill_name == "amazon_search"
    assert choice.arguments.get("query") == "apples"
