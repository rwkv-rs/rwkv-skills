from __future__ import annotations

"""Database manager - 兼容层，实际使用 SQLAlchemy ORM。"""

from src.eval.scheduler.config import DBConfig
from .orm import init_orm


class DatabaseManager:
    """兼容旧代码的数据库管理器，实际初始化委托给 orm.init_orm。"""

    _instance: DatabaseManager | None = None
    _config: DBConfig | None = None

    def __init__(self) -> None:
        raise RuntimeError("Use DatabaseManager.instance() instead")

    @classmethod
    def instance(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def initialize(self, config: DBConfig) -> None:
        if self._config is not None:
            return
        self._config = config
        init_orm(config)

    @property
    def config(self) -> DBConfig | None:
        return self._config


__all__ = ["DatabaseManager"]
