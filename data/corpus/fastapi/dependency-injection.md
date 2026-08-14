---
title: Dependency Injection - FastAPI
source_url: https://fastapi.tiangolo.com/tutorial/dependencies/
---

# Dependency Injection

FastAPI includes an extremely easy-to-use but powerful Dependency
Injection system. Dependencies can be shared between path operations,
and the same dependency can be reused across the application.

## What is a Dependency?

A "dependency" is just a function. FastAPI calls the function before
your path operation, and injects the return value into your function as
a parameter. This is useful for:

- Sharing logic (database connections, authentication) across endpoints.
- Enforcing permissions and roles.
- Reducing code duplication.

## Basic Dependency

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

When the endpoint is called, FastAPI:

1. Calls `common_parameters` with the query parameters `q`, `skip`,
   `limit` parsed from the request.
2. Takes the returned dict and injects it as `commons`.
3. Calls `read_items` with `commons`.

## Classes as Dependencies

A dependency can be a class. FastAPI will instantiate it using the
type-hinted constructor parameters:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends()):
    return {"q": commons.q, "skip": commons.skip, "limit": commons.limit}
```

Writing `= Depends()` (with no argument) is shorthand for
`= Depends(CommonQueryParams)` — FastAPI infers the dependency from the
type hint.

## Sub-Dependencies

Dependencies can themselves depend on other functions. FastAPI resolves
the full dependency tree automatically:

```python
def query_extractor(q: str | None = None):
    return q

def query_or_default_extractor(q: str = Depends(query_extractor)):
    if q is None:
        return "default-query"
    return q

@app.get("/items/")
async def read_extractor(extracted: str = Depends(query_or_default_extractor)):
    return {"extracted": extracted}
```

## Dependencies with `yield`

For dependencies that need cleanup (database sessions, file handles),
use `yield` instead of `return`. The code after the `yield` runs as a
finally block once the request finishes:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Global Dependencies

You can attach dependencies to the whole app (or a router) so they run
on every request:

```python
app = FastAPI(dependencies=[Depends(verify_token)])
```

This is commonly used for authentication or rate-limiting middleware
that should apply to every endpoint.
