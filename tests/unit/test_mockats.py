"""The mock ATS is test infrastructure, so its own correctness matters.

A broken confirmation page here does not look like a harness bug — it looks like the
agent failing to confirm a submission, and it silently invalidates every end-to-end
result. These tests walk the full wizard the way the agent does.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from cvnator.mockats.server import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        test_client.post("/_debug/reset")
        yield test_client


def _start(client) -> str:
    response = client.post("/jobs/MJ-1001/apply", follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"].split("/")[2]


def test_the_confirmation_route_is_reachable(client):
    """Regression: "/apply/{token}/{step}" with step:int shadowed ".../done".

    FastAPI matches routes in definition order, so "done" was coerced to int and
    returned 422. The application had actually been submitted; the agent simply could
    not see the confirmation, and correctly refused to report success.
    """
    token = _start(client)
    assert client.get(f"/apply/{token}/done").status_code == 200
    assert client.get(f"/apply/{token}/1").status_code == 200


def test_an_unknown_token_is_not_found_rather_than_unprocessable(client):
    assert client.get("/apply/nope/done").status_code == 404
    assert client.get("/apply/nope/1").status_code == 404


def test_the_full_wizard_reaches_a_reference_number(client):
    token = _start(client)

    step1 = client.post(
        f"/apply/{token}/1",
        data={
            "full_name": "Ana Souza",
            "email": "ana@example.com",
            "phone": "+55 11 90000-0000",
            "city": "São Paulo",
            "linkedin": "",
        },
        follow_redirects=False,
    )
    assert step1.status_code == 303

    step2 = client.post(
        f"/apply/{token}/2",
        data={
            "action": "next",
            "current_title": "Data Analyst",
            "current_company": "Acme Log",
            "years_experience": "6",
            "seniority": "mid",
        },
        follow_redirects=False,
    )
    assert step2.status_code == 303

    step3 = client.post(
        f"/apply/{token}/3",
        data={"action": "next"},
        files={"resume": ("cv.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        follow_redirects=False,
    )
    assert step3.status_code == 303

    step4 = client.post(
        f"/apply/{token}/4",
        data={
            "action": "next",
            "sql_years": "5",
            "english_level": "advanced",
            "work_auth": "yes",
            "tools": ["Grafana"],
            "motivation": "I have built operational dashboards and modelled data in SQL "
            "for several years, which matches this role closely.",
            "consent": "yes",
        },
        follow_redirects=False,
    )
    assert step4.status_code == 303

    submitted = client.post(
        f"/apply/{token}/5", data={"action": "submit"}, follow_redirects=False
    )
    assert submitted.status_code == 303

    confirmation = client.get(f"/apply/{token}/done")
    assert confirmation.status_code == 200
    assert "Application submitted" in confirmation.text

    received = client.get("/_debug/applications").json()[token]
    assert received["submitted"] is True
    assert received["reference"].startswith("MA-")
    assert received["resume_filename"] == "cv.pdf"


def test_each_step_rejects_incomplete_input(client):
    token = _start(client)

    empty = client.post(f"/apply/{token}/1", data={"full_name": "", "email": "not-an-email"})
    assert "is required" in empty.text
    assert "not valid" in empty.text

    client.post(
        f"/apply/{token}/1",
        data={
            "full_name": "Ana",
            "email": "a@b.com",
            "phone": "1",
            "city": "SP",
            "linkedin": "",
        },
    )
    no_resume = client.post(f"/apply/{token}/3", data={"action": "next"})
    assert "resume file is required" in no_resume.text


def test_the_upload_rejects_a_disallowed_file_type(client):
    token = _start(client)
    client.post(
        f"/apply/{token}/1",
        data={
            "full_name": "Ana",
            "email": "a@b.com",
            "phone": "1",
            "city": "SP",
            "linkedin": "",
        },
    )
    response = client.post(
        f"/apply/{token}/3",
        data={"action": "next"},
        files={"resume": ("cv.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert "PDF or DOCX" in response.text


def test_back_navigation_does_not_lose_answers(client):
    token = _start(client)
    client.post(
        f"/apply/{token}/1",
        data={
            "full_name": "Ana Souza",
            "email": "a@b.com",
            "phone": "1",
            "city": "SP",
            "linkedin": "",
        },
    )
    client.post(
        f"/apply/{token}/2",
        data={
            "action": "back",
            "current_title": "Data Analyst",
            "current_company": "Acme Log",
            "years_experience": "6",
            "seniority": "mid",
        },
    )
    assert "Ana Souza" in client.get(f"/apply/{token}/1").text
