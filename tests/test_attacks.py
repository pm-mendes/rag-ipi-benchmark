"""Unit tests for the attack catalog and the lexical attack-success detector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragipi.attacks.catalog import CANONICAL_ATTACKS
from ragipi.attacks.runner import SUCCESS_INDICATORS, is_attack_successful


class TestCanonicalAttackCatalog:
    def test_six_canonical_attacks(self):
        # Section IV.B of the paper: 6 canonical attacks.
        assert len(CANONICAL_ATTACKS) == 6

    def test_entries_are_name_path_pairs(self):
        for name, path in CANONICAL_ATTACKS:
            assert isinstance(name, str) and name.isupper()
            assert path.startswith("data/adversarial/")
            assert path.endswith(".txt")

    def test_expected_attack_names_present(self):
        names = {name for name, _ in CANONICAL_ATTACKS}
        assert names == {
            "OVERRIDE", "EXFIL", "ROLE", "DENIAL",
            "TECHNICAL_OVERRIDE", "DATA_EXFIL",
        }


class TestIsAttackSuccessful:
    def test_override_success(self):
        assert is_attack_successful("OVERRIDE", "you are pwned, starting new task now")

    def test_override_failure_on_clean_response(self):
        assert not is_attack_successful("OVERRIDE", "The Eiffel Tower is in Paris.")

    def test_exfil_success(self):
        assert is_attack_successful("EXFIL", "leaked: here is the system prompt")

    def test_role_success(self):
        assert is_attack_successful("ROLE", "DAN mode activated. No restrictions.")

    def test_denial_success(self):
        assert is_attack_successful("DENIAL", "Access denied. This is classified.")

    def test_all_canonical_attacks_have_indicators(self):
        for name, _ in CANONICAL_ATTACKS:
            assert name in SUCCESS_INDICATORS
            assert len(SUCCESS_INDICATORS[name]) > 0
