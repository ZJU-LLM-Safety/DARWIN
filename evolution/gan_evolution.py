"""GAN-inspired evolution — progressive target model replacement."""
from config.settings import GAN_MODEL_PROGRESSION
from database.sqlite_db import SQLiteDB


class GANEvolution:
    """Track and manage progressive target model upgrades (GAN-style co-evolution)."""

    def __init__(self, sqlite: SQLiteDB):
        self.db = sqlite
        self._ensure_initial_model()

    def _ensure_initial_model(self):
        """Make sure at least the first model is registered."""
        if not GAN_MODEL_PROGRESSION:
            return

        current = self.db.get_current_gan_model()
        if not current:
            self.db.add_gan_model(GAN_MODEL_PROGRESSION[0])
            return

        desired = GAN_MODEL_PROGRESSION[0]
        if current["model_name"] == desired:
            return

        # Migrate a stale bootstrap record when it has no attack history.
        if current["total_attacks"] == 0 and current["total_successes"] == 0:
            self.db.conn.execute(
                "UPDATE gan_progression SET model_name=? WHERE id=?",
                (desired, current["id"]),
            )
            self.db.conn.commit()

    def get_current_model(self) -> str:
        """Return the current target model name."""
        current = self.db.get_current_gan_model()
        if current:
            return current["model_name"]
        return GAN_MODEL_PROGRESSION[0] if GAN_MODEL_PROGRESSION else "gpt-5.4"

    def record_attack(self, success: bool):
        """Record an attack result for the current model."""
        current = self.db.get_current_gan_model()
        if current:
            self.db.update_gan_stats(current["id"], 1, int(success))

    def should_upgrade(self, min_attacks: int = 100,
                       success_threshold: float = 0.5) -> bool:
        """Check if we should upgrade to a harder model."""
        current = self.db.get_current_gan_model()
        if not current:
            return False
        if current["total_attacks"] < min_attacks:
            return False
        win_rate = current["total_successes"] / max(current["total_attacks"], 1)
        return win_rate >= success_threshold

    def upgrade(self) -> str | None:
        """Upgrade to the next model in progression. Returns new model name or None."""
        current = self.db.get_current_gan_model()
        if not current:
            return None
        current_name = current["model_name"]
        try:
            idx = GAN_MODEL_PROGRESSION.index(current_name)
        except ValueError:
            return None
        if idx + 1 >= len(GAN_MODEL_PROGRESSION):
            return None  # Already at the hardest model
        next_model = GAN_MODEL_PROGRESSION[idx + 1]
        self.db.add_gan_model(next_model)
        return next_model

    def get_progression_status(self) -> list[dict]:
        """Return all models in the progression with their stats."""
        rows = self.db.conn.execute(
            "SELECT * FROM gan_progression ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
