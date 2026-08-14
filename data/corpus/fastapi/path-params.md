---
title: Path Parameters - FastAPI
source_url: https://fastapi.tiangolo.com/tutorial/path-params/
---

# Path Parameters

You can declare path parameters in FastAPI using Python format strings
(the same syntax used by `str.format`). FastAPI validates and converts
the path parameter to the declared type.

## Basic Usage

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
```

The value of `item_id` in the path `/items/{item_id}` is passed to the
function as the `item_id` argument. So if you run the server and go to
`http://127.0.0.1:8000/items/foo`, you will see:

```json
{"item_id": "foo"}
```

## Path Parameters with Types

You can declare the type of a path parameter using Python type hints.
FastAPI uses the type to parse and validate the value:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Now `item_id` is validated as an integer. A request to `/items/3` returns
`{"item_id": 3}` (a JSON number, not a string). A request to
`/items/abc` returns a 422 error because `abc` cannot be converted to
`int`.

## Path Parameter Order Matters

When you define two paths that could overlap, the order of route
declaration matters. For example:

```python
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

The `/users/me` route must be declared **before** `/users/{user_id}`,
otherwise `/users/me` would match the parameterized path and `user_id`
would be `"me"`.

## Predefined Values with Enum

If you have a path parameter that should only accept a fixed set of
values, use a standard `enum.Enum`:

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW"}
    return {"model_name": model_name}
```

FastAPI validates that the incoming path value is one of the enum
members and returns a 422 otherwise. Inside the function you can
compare against enum members directly.

## Path Parameter Validation

For finer control over path parameters (minimum/maximum length, regex
patterns), use `Path` from `fastapi`:

```python
from fastapi import FastAPI, Path

@app.get("/items/{item_id}")
async def read_item(item_id: int = Path(ge=1, le=1000)):
    return {"item_id": item_id}
```

This restricts `item_id` to integers between 1 and 1000 inclusive.
Path parameters always have a value (they come from the URL), so you
cannot declare them optional — but you can constrain them with `Path`.
