#!/usr/bin/env python3
"""
Test script for Lattice integration.

Run: python -m dacribagents.cli.test_lattice
"""

from __future__ import annotations

import os
import sys


def test_settings():
    """Test that settings load correctly with new Kafka fields."""
    print("\n=== Testing Settings ===")
    
    from dacribagents.infrastructure.settings import get_settings
    
    settings = get_settings()
    
    print(f"✓ App: {settings.app_name} v{settings.app_version}")
    print(f"✓ Environment: {settings.environment}")
    print(f"✓ Lattice Tenant: {settings.lattice_tenant_id}")
    print(f"✓ Lattice Milvus: {settings.lattice_milvus_uri}")
    print(f"✓ Lattice Postgres: {settings.lattice_postgres_host}:{settings.lattice_postgres_port}/{settings.lattice_postgres_db}")
    print(f"✓ Kafka Brokers: {settings.kafka_brokers or '(not configured)'}")
    print(f"✓ Kafka Enabled: {settings.kafka_enabled}")
    print(f"✓ Kafka Topic: {settings.kafka_topic_raw}")
    print(f"✓ RAG Top-K: {settings.rag_top_k}")
    print(f"✓ Email Poll Minutes: {settings.email_poll_minutes}")
    
    return True


def test_kafka_connection():
    """Test Kafka connectivity."""
    print("\n=== Testing Kafka Connection ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    if not settings.kafka_enabled:
        print("⚠ Kafka not configured - skipping")
        print("  Set KAFKA_BROKERS, KAFKA_SASL_USERNAME, KAFKA_SASL_PASSWORD")
        return False
    
    try:
        from confluent_kafka import Producer
        from confluent_kafka.admin import AdminClient
        
        config = settings.get_kafka_config()
        
        # Test admin connection
        admin = AdminClient(config)
        metadata = admin.list_topics(timeout=10)
        
        print(f"✓ Connected to Kafka cluster")
        print(f"✓ Broker count: {len(metadata.brokers)}")
        
        # Check if our topic exists
        topic = settings.kafka_topic_raw
        if topic in metadata.topics:
            topic_meta = metadata.topics[topic]
            print(f"✓ Topic '{topic}' exists with {len(topic_meta.partitions)} partitions")
        else:
            print(f"⚠ Topic '{topic}' does not exist (will be auto-created on first publish)")
        
        return True
        
    except Exception as e:
        print(f"✗ Kafka connection failed: {e}")
        return False


def test_publisher():
    """Test the Lattice Kafka publisher."""
    print("\n=== Testing Publisher ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    if not settings.kafka_enabled:
        print("⚠ Kafka not configured - skipping")
        return False
    
    try:
        from dacribagents.infrastructure.lattice.publisher import get_lattice_publisher
        
        publisher = get_lattice_publisher()
        print(f"✓ Publisher created for topic: {publisher.topic}")
        print(f"✓ Tenant ID: {publisher.tenant_id}")
        
        # Don't actually publish - just verify it initializes
        return True
        
    except Exception as e:
        print(f"✗ Publisher init failed: {e}")
        return False


def test_lattice_postgres():
    """Test Lattice Postgres connectivity."""
    print("\n=== Testing Lattice Postgres ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(settings.lattice_postgres_dsn)
        cur = conn.cursor()
        
        # Check if email table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'email'
            )
        """)
        email_exists = cur.fetchone()[0]
        
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'email_chunk'
            )
        """)
        chunk_exists = cur.fetchone()[0]
        
        if email_exists:
            cur.execute("SELECT COUNT(*) FROM email")
            email_count = cur.fetchone()[0]
            print(f"✓ email table: {email_count} rows")
        else:
            print("⚠ email table does not exist")
        
        if chunk_exists:
            cur.execute("SELECT COUNT(*) FROM email_chunk")
            chunk_count = cur.fetchone()[0]
            print(f"✓ email_chunk table: {chunk_count} rows")
        else:
            print("⚠ email_chunk table does not exist")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Lattice Postgres connection failed: {e}")
        print(f"  DSN: {settings.lattice_postgres_dsn}")
        return False


def test_lattice_milvus():
    """Test Lattice Milvus connectivity."""
    print("\n=== Testing Lattice Milvus ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    try:
        from pymilvus import MilvusClient
        
        client = MilvusClient(uri=settings.lattice_milvus_uri)
        
        # List collections
        collections = client.list_collections()
        print(f"✓ Connected to Milvus at {settings.lattice_milvus_uri}")
        print(f"✓ Collections: {collections}")
        
        # Check for email_chunks collection
        target = settings.lattice_milvus_collection
        if target in collections:
            stats = client.get_collection_stats(target)
            print(f"✓ Collection '{target}': {stats.get('row_count', 'unknown')} vectors")
        else:
            print(f"⚠ Collection '{target}' does not exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Lattice Milvus connection failed: {e}")
        return False


def test_retriever():
    """Test the Lattice email retriever."""
    print("\n=== Testing Retriever ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    try:
        from dacribagents.infrastructure.lattice.retriever import LatticeEmailRetriever
        
        retriever = LatticeEmailRetriever(
            milvus_uri=settings.lattice_milvus_uri,
            postgres_dsn=settings.lattice_postgres_dsn,
            tenant_id=settings.lattice_tenant_id,
            top_k=3,
        )
        
        print(f"✓ Retriever initialized")
        print(f"  Tenant: {retriever.tenant_id}")
        print(f"  Collection: {retriever.collection}")
        print(f"  Top-K: {retriever.top_k}")
        
        # Try a test query
        print("\n  Running test query: 'solar panel installation'...")
        docs = retriever.get_relevant_documents("solar panel installation")
        
        if docs:
            print(f"✓ Found {len(docs)} results")
            for i, doc in enumerate(docs[:3], 1):
                print(f"  {i}. {doc.metadata.get('subject', 'No subject')[:50]}... (score: {doc.metadata.get('score', 0):.3f})")
        else:
            print("⚠ No results found (collection may be empty)")
        
        return True
        
    except Exception as e:
        print(f"✗ Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_publish_sample():
    """Test publishing a sample email to Kafka."""
    print("\n=== Testing Sample Publish ===")
    
    from dacribagents.infrastructure.settings import get_settings
    settings = get_settings()
    
    if not settings.kafka_enabled:
        print("⚠ Kafka not configured - skipping")
        return False
    
    # Confirm before publishing
    response = input("Publish a test message to Kafka? (y/N): ")
    if response.lower() != 'y':
        print("  Skipped")
        return True
    
    try:
        from dacribagents.infrastructure.lattice.publisher import get_lattice_publisher
        
        # Create a minimal RFC822 email
        test_email = b"""From: test@example.com
To: agents@woodcreek.me
Subject: Test Email from Lattice Integration
Date: Mon, 30 Dec 2024 10:00:00 -0600
Message-ID: <test-integration-001@woodcreek.me>

This is a test email to verify the Lattice integration is working.

- Woodcreek Agents Test
"""
        
        publisher = get_lattice_publisher()
        
        message_id = publisher.publish(
            rfc822_bytes=test_email,
            account_id="test-integration",
            alias="agents",
            provider_message_id="test-001",
            folder="INBOX",
        )
        
        # Flush to ensure delivery
        remaining = publisher.flush(timeout=10)
        
        if remaining == 0:
            print(f"✓ Published test message: {message_id}")
            print(f"  Topic: {publisher.topic}")
            
            errors = publisher.get_errors()
            if errors:
                print(f"⚠ Delivery errors: {errors}")
                return False
            
            return True
        else:
            print(f"✗ {remaining} messages still pending after flush")
            return False
        
    except Exception as e:
        print(f"✗ Publish failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("  Lattice Integration Tests")
    print("=" * 60)
    
    results = {}
    
    # Always run settings test
    results["settings"] = test_settings()
    
    # Kafka tests
    results["kafka_connection"] = test_kafka_connection()
    results["publisher"] = test_publisher()
    
    # Lattice storage tests
    results["lattice_postgres"] = test_lattice_postgres()
    results["lattice_milvus"] = test_lattice_milvus()
    results["retriever"] = test_retriever()
    
    # Optional: publish test
    if "--publish" in sys.argv:
        results["publish"] = test_publish_sample()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())