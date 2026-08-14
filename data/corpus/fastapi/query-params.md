---
title: Query Parameters - FastAPI
source_url: https://fastapi.tiangolo.com/tutorial/query-params/
---

# Query Parameters

You can declare query parameters in FastAPI by adding them to your path
operation function as default parameters. FastAPI automatically parses
them from the URL query string.

## Basic Usage

When you declare parameters that are not part of the path, FastAPI
interprets them as query parameters:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

The query is the set of key-value pairs that go after the `?` in the URL,
separated by `&` characters. For example, in the URL:

    http://127.0.0.1:8000/items/?skip=0&limit=10

the query parameters are `skip=0` and `limit=10`.

## Optional Parameters

You can declare optional query parameters by giving them a default value
of `None`:

```python
from typing import Union

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```

In Python 3.10+ you can use `str | None` instead of `Union[str, None]`.

## Boolean Type Conversion

When you declare a parameter with a `bool` type, FastAPI converts
common truthy/falsy string values:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if not short:
        item.update({"description": "This is an amazing item"})
    return item
```

The values `True`, `true`, `1`, `yes`, `on` are converted to `True`.
The values `False`, `false`, `0`, `no`, `off` are converted to `False`.

## Multiple Path and Query Parameters

You can mix path and query parameters in the same function. FastAPI
knows which is which by matching the function parameters to the path
template:

```python
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if not short:
        item.update({"description": "This is an amazing item"})
    if q:
        item.update({"q": q})
    return item
```

## Required Query Parameters

When a parameter is declared without a default value, it becomes
required. If the client does not provide it, FastAPI returns a 422
Unprocessable Entity error with a clear validation message.

```python
@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    return {"item_id": item_id, "needy": needy}
```

You can also make some parameters required and others optional in the
same function. The order does not matter — FastAPI determines
required vs optional by the presence of a default value, not by
parameter ordering.
