import pytest
from fastapi.testclient import TestClient

from wstv.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("endpoint", ("ready", "live"))
def test_ready_live(client: TestClient, endpoint: str):
    res = client.get(f"/.well-known/{endpoint}")
    assert res.status_code == 204


def test_meta(client: TestClient):
    res = client.get("/meta")
    assert res.status_code == 200
    assert res.json()


@pytest.mark.parametrize(
    "sentence",
    (
        "Python is easy to learn.",
        "AI can enhance decision-making processes.",
        "SpaceX is revolutionizing space travel.",
        "Python excels in data analysis.",
        "AI algorithms learn from data.",
        "Mars rovers explore new terrains.",
        "Python supports multiple programming paradigms.",
        "AI improves user experience on websites.",
        "The International Space Station orbits Earth.",
        "Python's syntax is very readable.",
        "AI in healthcare can predict outcomes.",
        "Astronauts conduct experiments in space.",
        "Python is widely used in web development.",
        "Machine learning is a subset of AI.",
        "NASA aims to return humans to the Moon.",
        "Python libraries simplify complex tasks.",
        "Autonomous vehicles rely on AI technologies.",
        "Voyager 1 has left our solar system.",
        "Python is open-source and community-driven.",
        "Voice assistants use AI to understand speech.",
        "Telescopes help in observing distant galaxies.",
        "Python's popularity grows each year.",
        "AI can identify patterns in big data.",
        "Satellites provide crucial weather data.",
        "Python can run on many operating systems.",
        "Neural networks mimic human brain functions.",
        "Space debris is a growing concern in orbit.",
        "Python scripts automate repetitive tasks.",
        "AI ethics is a growing field of study.",
        "The Hubble Space Telescope has changed our view of the cosmos.",
    ),
)
def test_vectorize_sentence(client: TestClient, sentence: str):
    res = client.post("/vectors/", json={"text": sentence})
    assert res.status_code == 200
    assert len(res.json()["vector"]) > 100


@pytest.mark.parametrize("task_type", ("passage", "query"))
def test_vectorize_with_task_type(client: TestClient, task_type: str):
    res = client.post(
        "/vectors/",
        json={
            "text": "The London Eye is a ferris wheel at the River Thames.",
            "config": {"task_type": task_type},
        },
    )
    assert res.status_code == 200
    assert len(res.json()["vector"]) > 100


@pytest.mark.parametrize("dimensions", (128, 256))
def test_vectorize_with_dimensions(client: TestClient, dimensions: int):
    res = client.post(
        "/vectors/",
        json={
            "text": "The London Eye is a ferris wheel at the River Thames.",
            "config": {"dimensions": dimensions},
        },
    )
    assert res.status_code == 200
    assert len(res.json()["vector"]) == dimensions
