# DataLab-Kernel Test Suite

This directory contains the test suite for DataLab-Kernel.

## Architecture Note

**The entire DataLab-Kernel package uses ONLY the Web API - no XML-RPC.**

The test infrastructure starts DataLab with pre-configured WebAPI settings using
environment variables (`DATALAB_WEBAPI_ENABLED=1`, `DATALAB_WEBAPI_PORT`, etc.),
eliminating any need for XML-RPC communication.

## Running Tests

### Default Mode (Complete Coverage) - RECOMMENDED

```bash
pytest
```

**Automatic comprehensive testing in ONE command:**

1. ✅ Runs all **standalone tests first** (98 tests, ~2 seconds, no DataLab needed)
2. 🚀 **Automatically starts DataLab** when first live test begins
3. ✅ Runs all **live backend tests** (24 tests, ~5 seconds)
4. 🛑 **Stops DataLab** automatically after tests complete

**Result:**

```text
======================================================================
🚀 Starting DataLab for live tests...
======================================================================
✅ DataLab ready for live tests
======================================================================

122 passed in 35s  # Complete coverage of both backends!
```

**Benefits:**

- **Complete testing** in a single pytest run
- **Fast feedback** on standalone features (first 2 seconds)
- **No manual setup** required
- **Automatic cleanup** after completion

### Standalone-Only Mode (Quick Testing)

```bash
pytest --standalone-only
```

Skips all live tests, testing only standalone backend:

- ⚡ **Fast**: ~2 seconds
- 📦 **No DataLab needed**
- ✅ **98 tests**: Complete standalone coverage

**Result:**

```text
98 passed, 24 skipped in 2.13s  # Standalone backend fully tested
```

**Use when:**

- Quick local testing during development
- CI pipelines where DataLab startup time is critical
- Testing standalone features only

### Force Live Mode

```bash
pytest --live
```

Runs only live tests, skipping standalone-only tests:

- Starts DataLab automatically if not running
- Skips 3 standalone-specific tests
- Tests live backend in isolation

**Result:**

```text
119 passed, 3 skipped  # Live backend fully tested
```

### Pre-Start DataLab (Explicit Mode)

```bash
pytest --start-datalab
```

Same as default mode, but starts DataLab at session beginning instead of lazily:

- DataLab starts before any tests run
- Slightly longer startup time
- Useful for debugging or CI environments

### Manual Mode (Development with Running DataLab)

If you already have DataLab running with WebAPI enabled:

```bash
# 1. DataLab is already running
# 2. WebAPI server is started (Tools > Web API > Start Server)
# 3. Just run tests:
pytest
```

Tests will detect the running DataLab and use it instead of starting a new instance.

## Test Markers

- `@pytest.mark.standalone` - Tests for standalone mode only (e.g., HDF5 persistence)
- `@pytest.mark.live` - Tests requiring live DataLab connection
- `@pytest.mark.contract` - Tests that should pass in both modes
- `@pytest.mark.webapi` - Tests specifically for WebAPI backend

## Test Coverage Summary

| Category | Count | Description |
|----------|-------|-------------|
| Unit tests | 77 | Core functionality (objects, kernel, install, plotter, persistence, workspace) |
| Contract tests | 9 | API compatibility tests (run in standalone mode) |
| Integration tests (standalone) | 3 | Standalone-specific integration tests |
| Integration tests (live) | 20 | Live backend tests (auto-run with DataLab) |
| WebAPI tests | 5 | WebAPI-specific tests (auto-run with DataLab) |
| Integration (restrictions) | 2 | Tests that standalone mode properly restricts live-only features |
| Backend selection | 6 | Tests for backend auto-detection |
| **Total** | **122** | **Full test suite** |

## Test Execution Flow (Default Mode)

```text
pytest
  │
  ├─► Phase 1: Standalone Tests (98 tests, ~2s)
  │   ├─ Unit tests
  │   ├─ Contract tests
  │   ├─ Backend selection tests
  │   └─ Standalone integration tests
  │
  ├─► 🚀 Auto-start DataLab (lazy initialization)
  │   └─ Only happens when first live test runs
  │
  └─► Phase 2: Live Tests (24 tests, ~5s)
      ├─ Live backend integration tests
      └─ WebAPI backend tests
```

## Architecture

```text
tests/
├── conftest.py          # pytest configuration, smart test orchestration
│                        # - Reorders tests (standalone first, live second)
│                        # - Lazy DataLab startup (auto_datalab fixture)
│                        # - DataLab lifecycle management
├── contract/            # Tests for both standalone and live modes
├── integration/         # Integration tests with DataLab
│   └── test_live_backend.py  # Live backend tests (auto-run after standalone)
├── test_webapi_backend.py    # WebAPI backend tests (auto-run after standalone)
└── unit/                # Unit tests for kernel components
```

The separation between test infrastructure and kernel workspace is intentional and maintains clean architectural boundaries.

## Continuous Integration Recommendations

### Quick CI (Standalone Only) - 2 seconds

```bash
pytest --standalone-only
# Fast feedback on core functionality
```

### Full CI (Complete Coverage) - 35 seconds

```bash
pytest
# Complete testing of both backends in one command
```

### Debug Mode (Pre-started DataLab)

```bash
pytest --start-datalab
# DataLab starts immediately, useful for debugging test infrastructure
```
