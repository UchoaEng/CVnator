"""The fact validator is the safety-critical component; these are its guarantees."""

from __future__ import annotations

import pytest

from cvnator.resume.fact_validator import (
    FactValidator,
    ViolationKind,
    content_coverage,
)


def test_blocks_technology_the_profile_does_not_support(profile):
    report = FactValidator(profile).validate_text(
        "Experienced with Python, SQL and Kubernetes in production."
    )
    assert not report.ok
    kinds = {v.kind for v in report.blocking}
    assert ViolationKind.UNSUPPORTED_SKILL in kinds
    assert any("kubernetes" in v.evidence for v in report.blocking)


def test_allows_technology_the_profile_supports(profile):
    assert FactValidator(profile).validate_text("Built dashboards with Grafana and SQL.").ok


def test_alias_on_the_supported_side_is_not_double_reported(profile):
    """A profile listing "kubernetes" covers "k8s" - and one mention is one violation."""
    profile.skills.append(type(profile.skills[0])(name="Kubernetes", category="hard"))
    validator = FactValidator(profile)
    assert validator.validate_text("Ran workloads on k8s.").ok

    report = FactValidator(profile).validate_text("Used Terraform heavily.")
    assert len(report.blocking) == 1


def test_blocks_inflated_duration(profile):
    report = FactValidator(profile).validate_text("I have 9 years of Python experience.")
    assert not report.ok
    assert report.blocking[0].kind is ViolationKind.INFLATED_DURATION


def test_allows_supported_duration(profile):
    assert FactValidator(profile).validate_text("I have 3 years of Python.").ok


def test_blocks_work_authorization_claim_when_profile_has_none(profile):
    profile.work_authorization.countries_authorized = []
    report = FactValidator(profile).validate_text("I am authorized to work in Canada.")
    assert not report.ok
    assert report.blocking[0].kind is ViolationKind.UNSUPPORTED_AUTHORIZATION


def test_permits_the_specs_own_example_rewrite(profile):
    """The expansion the product specification cites as acceptable must pass."""
    original = "Criação de dashboards operacionais."
    rewritten = (
        "Desenvolvimento e acompanhamento de dashboards para análise de indicadores "
        "operacionais e monitoramento de performance."
    )
    assert FactValidator(profile).validate_rewrite(original, rewritten).ok


@pytest.mark.parametrize(
    "rewritten,expected",
    [
        ("Criação de dashboards operacionais usando Tableau.", ViolationKind.UNSUPPORTED_CLAIM),
        ("Desenvolvi uma equipe de 12 engenheiros em três continentes.",
         ViolationKind.UNGROUNDED_REWRITE),
        ("Fiz coisas.", ViolationKind.UNGROUNDED_REWRITE),
    ],
)
def test_blocks_bad_rewrites(profile, rewritten, expected):
    report = FactValidator(profile).validate_rewrite(
        "Criação de dashboards operacionais.", rewritten
    )
    assert not report.ok
    assert expected in {v.kind for v in report.blocking}


def test_a_cross_language_fabrication_is_routed_to_entailment_not_silently_allowed(profile):
    """The lexical layer cannot judge a translation, so it must hand it on.

    This is the one case where a deterministic pass alone is insufficient, and the
    contract is that it says so rather than approving by default.
    """
    validator = FactValidator(profile)
    fabricated = "Led a team of 12 engineers across three continents."
    assert validator.needs_entailment("Criação de dashboards operacionais.", fabricated)


def test_rewrite_with_no_source_is_blocked(profile):
    report = FactValidator(profile).validate_rewrite("", "Anything at all here.")
    assert not report.ok
    assert report.blocking[0].kind is ViolationKind.UNGROUNDED_REWRITE


def test_coverage_rewards_expansion_and_punishes_replacement():
    original = "Criação de dashboards operacionais."
    assert content_coverage(original, original + " Extra context added here.") == 1.0
    assert content_coverage(original, "Something entirely different.") < 0.4


def test_cross_language_rewrite_skips_lexical_coverage(profile):
    """A translation must not be blocked for lexical drift; entailment judges it."""
    validator = FactValidator(profile)
    report = validator.validate_rewrite(
        "Criação de dashboards operacionais.",
        "Built and maintained dashboards for operational indicators.",
    )
    assert report.ok, report.summary()
    assert validator.needs_entailment(
        "Criação de dashboards operacionais.",
        "Built and maintained dashboards for operational indicators.",
    )


def test_cross_language_rewrite_still_blocks_new_technology(profile):
    """Technology names do not translate, so that check survives a language change."""
    report = FactValidator(profile).validate_rewrite(
        "Criação de dashboards operacionais.",
        "Built Tableau dashboards for operational indicators.",
    )
    assert not report.ok
    assert ViolationKind.UNSUPPORTED_CLAIM in {v.kind for v in report.blocking}
