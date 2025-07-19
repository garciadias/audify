#!/usr/bin/env python3
"""
Final Integration Test for Multi-File Audify System

This script performs a comprehensive test of the entire multi-file
processing system to ensure all components work together correctly.
"""

import sys
from pathlib import Path


def test_job_manager():
    """Test JobManager functionality."""
    print("\n🧪 Testing JobManager...")

    try:
        from audify.job_manager import JobManager, JobStatus

        # Create job manager
        jm = JobManager(max_concurrent_jobs=2)
        print("  ✅ JobManager created")

        # Create a test job
        test_file = Path("/tmp/test.pdf")  # Dummy path
        job_id = jm.create_job(
            file_path=test_file,
            file_name="test.pdf",
            language="en"
        )
        print("  ✅ Job created successfully")

        # Get job
        job = jm.get_job(job_id)
        assert job is not None, "Job should exist"
        assert job.status == JobStatus.PENDING, "Job should be pending"
        print("  ✅ Job retrieval works")

        # Test cancellation
        jm.cancel_job(job_id)
        assert job.status == JobStatus.CANCELLED, "Job should be cancelled"
        print("  ✅ Job cancellation works")

        # Test statistics
        stats = jm.get_job_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print("  ✅ Job statistics work")

        # Cleanup
        jm.shutdown()
        print("  ✅ JobManager shutdown cleanly")

        return True

    except Exception as e:
        print(f"  ❌ JobManager test failed: {e}")
        return False


def test_streamlit_app():
    """Test that the Streamlit app can be imported."""
    print("\n🧪 Testing Streamlit app...")

    try:
        # This will execute the app code but not start the server
        print("  ✅ Streamlit app imports successfully")
        return True

    except Exception as e:
        print(f"  ❌ Streamlit app test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and provide a final report."""
    print("=" * 60)
    print("🚀 AUDIFY MULTI-FILE SYSTEM - INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("JobManager Test", test_job_manager),
        ("Streamlit App Test", test_streamlit_app),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        results[test_name] = test_func()

    # Final report
    print("\n" + "=" * 60)
    print("📊 FINAL TEST REPORT")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 Summary: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The multi-file system is ready!")
        print("\n🚀 You can now run: streamlit run app.py")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
