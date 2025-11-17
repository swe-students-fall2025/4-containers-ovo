"""
Database connection module for MongoDB.
"""

from pymongo import MongoClient
from config import Config

_client = None
_db = None


def get_client():
    """
    Get or create MongoDB client.

    Returns:
        MongoClient: MongoDB client instance
    """
    global _client

    if _client is None:
        # Set shorter timeout for faster failure when MongoDB is not available
        _client = MongoClient(
            Config.MONGO_URI,
            serverSelectionTimeoutMS=2000,  # 2 seconds timeout
            connectTimeoutMS=2000,
        )

    return _client


def get_database():
    """
    Get or create database connection.

    Returns:
        Database: MongoDB database instance
    """
    global _client, _db

    if _db is None:
        _client = get_client()
        _db = _client[Config.MONGO_DB_NAME]

    return _db


def close_database():
    """Close database connection."""
    global _client, _db

    if _client is not None:
        _client.close()
        _client = None
        _db = None
