import os
import unittest
from unittest.mock import MagicMock, patch

from src.infra.database import DatabaseManager
from src.eval.scheduler.config import DBConfig

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton
        DatabaseManager._instance = None
        self.db = DatabaseManager.instance()
        self.config = DBConfig(
            host="localhost",
            port=5432,
            user="test_user",
            password="test_password",
            dbname="test_db",
            enabled=True
        )

    @patch("psycopg.ConnectionPool")
    def test_initialize(self, mock_pool_cls):
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        
        self.db.initialize(self.config)
        
        # Verify pool creation
        mock_pool_cls.assert_called_once()
        conn_str = mock_pool_cls.call_args[0][0]
        self.assertIn("host=localhost", conn_str)
        self.assertIn("dbname=test_db", conn_str)
        
        # Verify schema init
        mock_conn.execute.assert_called()
        executed_sql = [call_args[0][0] for call_args in mock_conn.execute.call_args_list]
        self.assertTrue(
            any("CREATE TABLE IF NOT EXISTS eval_subject" in sql for sql in executed_sql)
        )

    @patch("psycopg.ConnectionPool")
    def test_transaction_commit(self, mock_pool_cls):
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        
        self.db.initialize(self.config)
        
        with self.db.get_connection() as conn:
            conn.execute("INSERT INTO test VALUES (1)")
            
        # Verify commit was called
        mock_conn.commit.assert_called_once()

    @patch("psycopg.ConnectionPool")
    def test_transaction_rollback(self, mock_pool_cls):
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        
        self.db.initialize(self.config)
        
        with self.assertRaises(ValueError):
            with self.db.get_connection() as conn:
                conn.execute("INSERT INTO test VALUES (1)")
                raise ValueError("Boom")
            
        # Verify rollback was called
        mock_conn.rollback.assert_called_once()
        # Commit should not be called
        mock_conn.commit.assert_not_called()

if __name__ == "__main__":
    unittest.main()
