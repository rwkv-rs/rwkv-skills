#!/usr/bin/env python3
"""Migration script to add session management fields to the task table.

This script adds the following fields to the task table:
- session_id: VARCHAR(255) - Groups tasks from the same dispatch run
- session_git_hash: VARCHAR(255) - Git hash when the session was created
- session_status: VARCHAR(20) - Status of the task within the session

Usage:
    python scripts/migrate_add_session_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from src.db.orm import init_orm, get_session
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


def migrate() -> None:
    """Add session management fields to the task table."""
    print("🔧 Starting migration: add session fields to task table")

    init_orm(DEFAULT_DB_CONFIG)

    with get_session() as conn:
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'task'
            AND column_name IN ('session_id', 'session_git_hash', 'session_status')
        """))
        existing_columns = {row[0] for row in result}

        if existing_columns:
            print(f"⚠️  Some session columns already exist: {existing_columns}")
            print("   Skipping migration (columns may have been added already)")
            return

        print("📝 Adding session_id column...")
        conn.execute(text("""
            ALTER TABLE task
            ADD COLUMN session_id VARCHAR(255)
        """))

        print("📝 Adding session_git_hash column...")
        conn.execute(text("""
            ALTER TABLE task
            ADD COLUMN session_git_hash VARCHAR(255)
        """))

        print("📝 Adding session_status column...")
        conn.execute(text("""
            ALTER TABLE task
            ADD COLUMN session_status VARCHAR(20) DEFAULT 'pending'
        """))

        print("📝 Creating indexes...")
        conn.execute(text("""
            CREATE INDEX idx_task_session ON task(session_id)
        """))

        conn.execute(text("""
            CREATE INDEX idx_task_session_git_hash ON task(session_git_hash)
        """))

        # get_session() will auto-commit on exit

        print("✅ Migration completed successfully!")
        print()
        print("Session management fields added:")
        print("  - session_id: Groups tasks from the same dispatch run")
        print("  - session_git_hash: Git hash when the session was created")
        print("  - session_status: Status of the task within the session")
        print()
        print("You can now use:")
        print("  - rwkv-skills-scheduler dispatch  # Creates a session automatically")
        print("  - rwkv-skills-scheduler resume    # Resume incomplete sessions")
        print("  - rwkv-skills-scheduler status --session-id <id>  # View session status")


if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
