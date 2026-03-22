import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.core.config import settings
from app.db.MySQL import Base, engine
from app.routers import attendance_router, image_router, person_router, video_router

logger = logging.getLogger(__name__)


def wait_for_database() -> None:
    """Retry database connection before creating tables."""
    attempts = settings.db_connect_retries
    while attempts > 0:
        try:
            with engine.connect():
                logger.info("Database connection established")
                return
        except Exception as exc:  # noqa: BLE001
            attempts -= 1
            if attempts == 0:
                raise
            logger.warning("DB not ready (%s). Retrying in %.1fs...", exc, settings.db_connect_interval)
            time.sleep(settings.db_connect_interval)


def ensure_schema_up_to_date() -> None:
    """Apply minimal migrations for existing databases."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "appearances" in tables:
        columns = {col["name"] for col in inspector.get_columns("appearances")}
        with engine.begin() as conn:
            if "name" not in columns:
                logger.info("Adding missing 'name' column to appearances table")
                conn.execute(
                    text("ALTER TABLE appearances ADD COLUMN name VARCHAR(255) NOT NULL DEFAULT 'unknown'")
                )
                conn.execute(
                    text(
                        "UPDATE appearances a "
                        "LEFT JOIN persons p ON a.person_id = p.id "
                        "SET a.name = COALESCE(p.name, 'unknown') "
                        "WHERE a.name = 'unknown' OR a.name IS NULL"
                    )
                )
            if "frame_time" in columns:
                logger.info("Dropping deprecated 'frame_time' column from appearances table")
                conn.execute(text("ALTER TABLE appearances DROP COLUMN frame_time"))


# Create DB tables on startup. In production replace with Alembic migrations.
wait_for_database()
Base.metadata.create_all(bind=engine)
ensure_schema_up_to_date()

app = FastAPI(title=settings.app_name)


app.include_router(image_router.router)
app.include_router(video_router.router)
app.include_router(person_router.router)
app.include_router(attendance_router.router)


@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok"}


@app.head("/", tags=["health"])
def health_head():
    return {"status": "ok"}
