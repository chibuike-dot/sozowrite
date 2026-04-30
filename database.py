import sqlite3
from typing import List, Dict, Any, Optional
import aiosqlite
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from .models import (
    UsageLog,
    ModelPriority,
    ProviderConfig
)

logger = logging.getLogger(__name__)

DATABASE_URL = "ai_aggregator.db"

async def init_db():
    """Initialize the database with required tables"""
    async with aiosqlite.connect(DATABASE_URL) as db:
        # Enable WAL mode for better concurrency
        await db.execute("PRAGMA journal_mode=WAL")

        # Create tables if they don't exist
        await db.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            tokens_used INTEGER NOT NULL,
            cost REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            status TEXT DEFAULT 'completed'
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS model_priorities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            priority TEXT NOT NULL,
            cost_per_token REAL NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            UNIQUE(provider, model_name)
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS fallback_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            priority INTEGER NOT NULL,
            UNIQUE(provider, model_name)
        )
        """)

        await db.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            monthly_quota INTEGER DEFAULT 10000
        )
        """)

        # Create indexes for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage_logs(api_key)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_logs(provider)")

        await db.commit()

        # Initialize with default priorities if empty
        await db.execute("""
        INSERT OR IGNORE INTO model_priorities (provider, model_name, priority, cost_per_token)
        VALUES
            ('mistral', 'mistral-tiny', 'free', 0.000002),
            ('mistral', 'mistral-small', 'free', 0.0000025),
            ('gemini', 'gemini-1.5-flash', 'paid', 0.0000025),
            ('gemini', 'gemini-1.5-pro', 'paid', 0.000005),
            ('deepseek', 'deepseek-chat', 'free', 0.0000015),
            ('deepseek', 'deepseek-coder', 'free', 0.0000018)
        """)

        await db.execute("""
        INSERT OR IGNORE INTO fallback_chain (provider, model_name, priority)
        VALUES
            ('mistral', 'mistral-small', 1),
            ('gemini', 'gemini-1.5-flash', 2),
            ('deepseek', 'deepseek-chat', 3)
        """)

        await db.commit()

@asynccontextmanager
async def get_db_connection():
    """Get a database connection with context management"""
    conn = await aiosqlite.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

async def log_usage(
    api_key: str,
    provider: str,
    model_name: str,
    tokens_used: int,
    cost: float,
    timestamp: datetime
):
    """Log API usage to the database"""
    async with get_db_connection() as db:
        await db.execute(
            """
            INSERT INTO usage_logs
            (api_key, provider, model_name, tokens_used, cost, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (api_key, provider, model_name, tokens_used, cost, timestamp.isoformat())
        )
        await db.commit()

async def get_api_key_usage(
    api_key: str,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """Get usage statistics for an API key"""
    async with get_db_connection() as db:
        # Get overall usage
        if provider:
            query = """
            SELECT
                provider,
                model_name,
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                COUNT(*) as request_count
            FROM usage_logs
            WHERE api_key = ? AND provider = ?
            GROUP BY provider, model_name
            """
            params = (api_key, provider)
        else:
            query = """
            SELECT
                provider,
                model_name,
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                COUNT(*) as request_count
            FROM usage_logs
            WHERE api_key = ?
            GROUP BY provider, model_name
            """
            params = (api_key,)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        # Get monthly usage
        now = datetime.utcnow()
        first_day_of_month = now.replace(day=1)

        if provider:
            monthly_query = """
            SELECT
                provider,
                SUM(tokens_used) as monthly_tokens,
                SUM(cost) as monthly_cost
            FROM usage_logs
            WHERE api_key = ? AND provider = ? AND timestamp >= ?
            GROUP BY provider
            """
            monthly_params = (api_key, provider, first_day_of_month.isoformat())
        else:
            monthly_query = """
            SELECT
                provider,
                SUM(tokens_used) as monthly_tokens,
                SUM(cost) as monthly_cost
            FROM usage_logs
            WHERE api_key = ? AND timestamp >= ?
            GROUP BY provider
            """
            monthly_params = (api_key, first_day_of_month.isoformat())

        monthly_cursor = await db.execute(monthly_query, monthly_params)
        monthly_rows = await monthly_cursor.fetchall()

        # Format results
        usage_by_model = {}
        for row in rows:
            provider_name = row[0]
            if provider_name not in usage_by_model:
                usage_by_model[provider_name] = {
                    "models": {},
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "request_count": 0
                }

            usage_by_model[provider_name]["models"][row[1]] = {
                "tokens_used": row[2],
                "cost": row[3],
                "request_count": row[4]
            }
            usage_by_model[provider_name]["total_tokens"] += row[2]
            usage_by_model[provider_name]["total_cost"] += row[3]
            usage_by_model[provider_name]["request_count"] += row[4]

        # Add monthly data
        monthly_usage = {row[0]: {"monthly_tokens": row[1], "monthly_cost": row[2]} for row in monthly_rows}

        for provider_name in usage_by_model:
            if provider_name in monthly_usage:
                usage_by_model[provider_name].update(monthly_usage[provider_name])

        return {
            "api_key": api_key,
            "period": {
                "start": first_day_of_month.isoformat(),
                "end": now.isoformat()
            },
            "usage_by_provider": usage_by_model,
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_model_priority() -> List[Dict[str, Any]]:
    """Get models ordered by priority"""
    async with get_db_connection() as db:
        cursor = await db.execute("""
        SELECT provider, model_name, priority, cost_per_token
        FROM model_priorities
        WHERE is_active = TRUE
        ORDER BY
            CASE priority
                WHEN 'free' THEN 1
                WHEN 'paid' THEN 2
                WHEN 'premium' THEN 3
                ELSE 4
            END,
            cost_per_token ASC
        """)
        rows = await cursor.fetchall()

        return [
            {
                "provider": row[0],
                "model_name": row[1],
                "priority": row[2],
                "cost_per_token": row[3]
            }
            for row in rows
        ]

async def get_fallback_chain() -> List[Dict[str, Any]]:
    """Get the fallback chain ordered by priority"""
    async with get_db_connection() as db:
        cursor = await db.execute("""
        SELECT provider, model_name, priority
        FROM fallback_chain
        ORDER BY priority ASC
        """)
        rows = await cursor.fetchall()

        return [
            {
                "provider": row[0],
                "model_name": row[1],
                "priority": row[2]
            }
            for row in rows
        ]

async def update_model_metrics(
    provider: str,
    model_name: str,
    response_time: float,
    success: bool
):
    """Update model performance metrics"""
    async with get_db_connection() as db:
        if success:
            await db.execute("""
            UPDATE model_priorities
            SET
                response_time_avg = (
                    SELECT AVG(
                        CASE
                            WHEN response_time_avg IS NULL THEN ?
                            ELSE (response_time_avg + ?) / 2
                        END
                    )
                    FROM model_priorities
                    WHERE provider = ? AND model_name = ?
                ),
                current_load = current_load + 1
            WHERE provider = ? AND model_name = ?
            """, (response_time, response_time, provider, model_name, provider, model_name))
        else:
            await db.execute("""
            UPDATE model_priorities
            SET current_load = current_load + 5  # Penalize failures more
            WHERE provider = ? AND model_name = ?
            """, (provider, model_name))

        await db.commit()
