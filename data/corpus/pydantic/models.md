---
title: Models - Pydantic v2
source_url: https://docs.pydantic.dev/2/concepts/models/
---

# Models in Pydantic v2

Pydantic models are the core of the library. A model is a class that
inherits from `BaseModel` and declares fields as annotated attributes.
Pydantic validates the input data at instantiation time and converts it
to the declared types.

## BaseModel

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None = None

user = User(id=123)
print(user.model_dump())
# {'id': 123, 'name': 'John Doe', 'signup_ts': None}
```

Fields without a default value are **required**. Fields with a default
value are optional. Pydantic raises `ValidationError` if required fields
are missing or if a value cannot be coerced to the declared type.

## BaseModel vs BaseSettings

`BaseModel` is for data validation and serialization. `BaseSettings`
(available via `pydantic-settings`) is for configuration management —
it reads values from environment variables, `.env` files, and secrets,
with the same validation logic as `BaseModel`.

| Class           | Purpose                         | Source of Data              |
| --------------- | ------------------------------- | --------------------------- |
| `BaseModel`     | Validate and serialize payloads | Constructor arguments       |
| `BaseSettings`  | App configuration              | Env vars, `.env`, secrets   |

Use `BaseModel` when the data comes from an API request body or a
database row. Use `BaseSettings` when the data comes from the
environment (API keys, database URLs, feature flags).

## Field Validation

Use `Field` to add constraints and metadata:

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="The price must be positive")
    tags: list[str] = Field(default_factory=list)
```

For custom validation logic, use `@field_validator`:

```python
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    username: str

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("must be alphanumeric")
        return v
```

## model_config

Model behavior is configured via the `model_config` attribute, which
accepts a `ConfigDict`:

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",        # reject unknown fields
        str_strip_whitespace=True,
        frozen=True,           # immutable instances
    )
```

Common options: `extra` (`"allow"`, `"ignore"`, `"forbid"`), `frozen`,
`str_strip_whitespace`, `populate_by_name`, `use_enum_values`.

## Serialization

Pydantic v2 uses `model_dump()` (replaces v1's `.dict()`) and
`model_dump_json()` (replaces v1's `.json()`):

```python
user = User(id=1, name="Alice")
user.model_dump()           # → dict
user.model_dump_json()      # → JSON string
user.model_dump(exclude={"name"})  # → dict without 'name'
```

## Nested Models

Models can be nested — a field's type can be another `BaseModel`:

```python
class Address(BaseModel):
    city: str
    zip_code: str

class UserWithAddress(BaseModel):
    id: int
    address: Address

user = UserWithAddress(id=1, address={"city": "NYC", "zip_code": "10001"})
print(user.address.city)  # "NYC"
```

Pydantic recursively validates nested models, so passing a dict for
`address` automatically constructs the `Address` instance.
