from app.config import Settings
from app.model_registry import resolve_model_route


def test_large_is_alias_to_nano_backend() -> None:
    settings = Settings()
    route = resolve_model_route("voyage-4-large", settings)

    assert route.logical_model == "voyage-4-large"
    assert route.backend_model == "voyageai/voyage-4-nano"
    assert route.lane == "large-shim"
    assert route.is_alias is True


def test_nano_is_not_alias() -> None:
    settings = Settings()
    route = resolve_model_route("voyage-4-nano", settings)

    assert route.logical_model == "voyage-4-nano"
    assert route.backend_model == "voyageai/voyage-4-nano"
    assert route.lane == "nano"
    assert route.is_alias is False
