from utils import make_changelog_entry


def test_make_changelog_entry():
    entry = make_changelog_entry("test", {"x": 1})
    assert "time" in entry and entry["action"] == "test" and entry["details"]["x"] == 1
