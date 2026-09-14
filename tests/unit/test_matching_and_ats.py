"""Hard filters, skill scoring, ATS analysis and scoring - all the deterministic parts."""

from __future__ import annotations

from cvnator.domain.job import Seniority
from cvnator.domain.profile import RemotePolicy
from cvnator.jobs.matching import apply_hard_filters, score_language, score_skills, score_title
from cvnator.jobs.prioritization import prioritize, score_priority
from cvnator.resume.ats import (
    ATSAnalysis,
    ATSRequirement,
    extract_action_verbs,
    extract_keywords_lexically,
    score_resume,
)

# ------------------------------------------------------------------- hard filters


def test_excluded_keyword_rejects_the_job(job, search, profile):
    search.excluded_keywords = ["Power BI", "estágio"]
    job.description += " Experience with Power BI required."
    result = apply_hard_filters(job, search, profile)
    assert not result.passed
    assert "Power BI" in result.reason


def test_blacklisted_company_rejects_the_job(job, search, profile):
    search.companies_blacklist = ["northwind analytics"]
    assert not apply_hard_filters(job, search, profile).passed


def test_whitelist_rejects_everything_not_on_it(job, search, profile):
    search.companies_whitelist = ["Some Other Co"]
    assert not apply_hard_filters(job, search, profile).passed
    search.companies_whitelist = ["Northwind Analytics"]
    assert apply_hard_filters(job, search, profile).passed


def test_missing_required_keyword_rejects_the_job(job, search, profile):
    search.required_keywords = ["Kubernetes"]
    result = apply_hard_filters(job, search, profile)
    assert not result.passed
    assert "Kubernetes" in result.reason


def test_remote_policy_filter(job, search, profile):
    search.remote_policies = [RemotePolicy.ONSITE]
    job.remote_policy = RemotePolicy.REMOTE
    assert not apply_hard_filters(job, search, profile).passed


def test_unstated_remote_policy_gets_the_benefit_of_the_doubt(job, search, profile):
    search.remote_policies = [RemotePolicy.REMOTE]
    job.remote_policy = RemotePolicy.ANY
    assert apply_hard_filters(job, search, profile).passed


def test_salary_ceiling_below_the_floor_rejects(job, search, profile):
    search.salary_minimum = 10000
    job.salary.maximum = 6000
    assert not apply_hard_filters(job, search, profile).passed


def test_sponsorship_mismatch_rejects(job, search, profile):
    profile.work_authorization.requires_sponsorship = True
    job.visa_sponsorship = False
    assert not apply_hard_filters(job, search, profile).passed


def test_seniority_filter(job, search, profile):
    search.seniorities = [Seniority.SENIOR]
    job.seniority = Seniority.JUNIOR
    assert not apply_hard_filters(job, search, profile).passed


# ----------------------------------------------------------------------- scoring


def test_skill_scoring_rewards_explicit_profile_skills(profile):
    analysis = ATSAnalysis(technologies=["Python", "SQL", "Kubernetes"])
    overall, scores = score_skills(profile, analysis)
    assert scores["python"] == 1.0
    assert scores["sql"] == 1.0
    assert scores["kubernetes"] == 0.0
    assert 0.6 < overall < 0.7


def test_skill_scoring_with_no_required_technologies_is_neutral(profile):
    overall, scores = score_skills(profile, ATSAnalysis())
    assert overall == 1.0
    assert scores == {}


def test_title_scoring(job, search):
    assert score_title(job, search) > 0.9
    job.role = "Chef de Cuisine"
    assert score_title(job, search) < 0.5


def test_language_scoring_reflects_proficiency(job, profile):
    assert score_language(job, profile, ATSAnalysis(language="pt")) == 1.0
    assert score_language(job, profile, ATSAnalysis(language="en")) == 0.9
    assert score_language(job, profile, ATSAnalysis(language="de")) == 0.8
    assert score_language(job, profile, ATSAnalysis(language="es")) == 0.3


# --------------------------------------------------------------------- ATS layer


def test_lexical_keyword_extraction_finds_technologies():
    keywords = extract_keywords_lexically(
        "We need SQL and Python. SQL is essential. Python is essential. Grafana preferred."
    )
    assert "sql" in keywords
    assert "python" in keywords
    assert "grafana" in keywords


def test_action_verbs_detected_in_both_languages():
    assert "build" in extract_action_verbs("You will build and maintain dashboards")
    assert "criar" in extract_action_verbs("Você vai criar e manter dashboards")


def test_ats_score_rises_when_the_resume_covers_more():
    analysis = ATSAnalysis(
        technologies=["SQL", "Python", "Grafana", "dbt"],
        keywords=["dashboard", "reporting"],
        requirements=[
            ATSRequirement(text="SQL", is_required=True),
            ATSRequirement(text="Python", is_required=True),
            ATSRequirement(text="Kubernetes", is_required=True),
            ATSRequirement(text="dbt", is_required=False),
        ],
    )
    poor = score_resume("Experienced professional. Worked with spreadsheets.", analysis)
    good = score_resume(
        "Built dashboards and reporting using SQL, Python and Grafana.", analysis
    )
    assert good.match_score > poor.match_score
    # What the resume genuinely lacks is reported, not glossed over.
    assert "Kubernetes" in good.missing_requirements
    assert "dbt" in good.missing_technologies
    assert good.match_score < 1.0


def test_ats_score_always_carries_its_disclaimer():
    payload = score_resume("anything", ATSAnalysis(technologies=["SQL"])).as_dict()
    assert "Not a prediction" in payload["disclaimer"]


def test_ats_score_is_bounded():
    analysis = ATSAnalysis(technologies=["SQL"], keywords=["sql"])
    assert 0.0 <= score_resume("", analysis).match_score <= 1.0
    assert 0.0 <= score_resume("SQL " * 500, analysis).match_score <= 1.0


# ------------------------------------------------------------------ prioritization


def test_higher_match_sorts_first(job, search):
    low = score_priority(job, 1, 0.55, search)
    high = score_priority(job, 2, 0.95, search)
    assert prioritize([low, high])[0].job_id == 2


def test_easy_apply_beats_an_external_application_at_equal_match(job, search):
    from cvnator.domain.job import ApplyMethod

    easy = job.model_copy(deep=True)
    easy.apply_method = ApplyMethod.EASY_APPLY
    external = job.model_copy(deep=True)
    external.apply_method = ApplyMethod.EXTERNAL

    ranked = prioritize(
        [score_priority(external, 1, 0.8, search), score_priority(easy, 2, 0.8, search)]
    )
    assert ranked[0].job_id == 2
