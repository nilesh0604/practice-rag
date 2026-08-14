---
title: Types - Pydantic v2
source_url: https://docs.pydantic.dev/2/concepts/types/
---

# Types in Pydantic v2

Pydantic supports a wide range of types out of the box. When a value is
assigned to a field, Pydantic attempts to coerce it to the declared
type. If coercion fails, a `ValidationError` is raised.

## Standard Library Types

Pydantic validates common Python types directly:

```python
from pydantic import BaseModel

class MyModel(BaseModel):
    a: int          # coerces "1" → 1, 1.0 → 1
    b: float        # coerces "1.5" → 1.5, 1 → 1.0
    c: str          # coerces b"hello" → "hello"
    d: bool         # coerces 1 → True, "true" → True, "yes" → True
    e: bytes        # coerces "hello" → b"hello"
    f: list[int]    # coerces ["1", "2"] → [1, 2]
    g: dict[str, int]  # coerces {"a": "1"} → {"a": 1}
```

## Optional and Union

Use `Optional` or `Union` for fields that accept multiple types or
`None`:

```python
from typing import Optional, Union

class Item(BaseModel):
    id: int
    desc: Optional[str] = None      # str or None, default None
    value: Union[int, float]        # int or float, required
```

In Python 3.10+ you can use the `|` syntax:

```python
class Item(BaseModel):
    id: int
    desc: str | None = None
    value: int | float
```

## Constrained Types

Use `Annotated` with `Field` constraints for fine-grained control:

```python
from typing import Annotated
from pydantic import BaseModel, Field

PositiveInt = Annotated[int, Field(gt=0)]
ShortStr = Annotated[str, Field(max_length=10)]

class Product(BaseModel):
    quantity: PositiveInt
    code: ShortStr
```

## Strict Mode

By default Pydantic coerces types (e.g. `"1"` → `1` for an `int`
field). In strict mode, only exact type matches are accepted:

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)
    x: int

StrictModel(x=1)      # OK
StrictModel(x="1")    # ValidationError in strict mode
```

You can also enable strict mode per-field using `StrictInt`, `StrictStr`,
etc. from `pydantic.types`.

## URL and Email Types

Pydantic provides specialized string types for common formats:

```python
from pydantic import BaseModel, HttpUrl, EmailStr

class Profile(BaseModel):
    website: HttpUrl
    email: EmailStr

p = Profile(website="https://example.com", email="user@example.com")
print(p.website.scheme)   # "https"
print(p.website.host)     # "example.com"
```

`HttpUrl` parses the URL into a structured object with `.scheme`,
`.host`, `.path`, etc. `EmailStr` requires the `email-validator`
package.

## Enums

Pydantic validates `enum.Enum` and `enum.IntEnum` values:

```python
from enum import Enum
from pydantic import BaseModel

class Color(str, Enum):
    red = "red"
    green = "green"
    blue = "blue"

class Item(BaseModel):
    color: Color

Item(color="red")   # OK → Color.red
Item(color="pink")  # ValidationError
```

## Datetime Types

Pydantic handles `datetime`, `date`, `time`, and `timedelta`. It parses
ISO 8601 strings and Unix timestamps:

```python
from datetime import datetime
from pydantic import BaseModel

class Event(BaseModel):
    ts: datetime

Event(ts="2026-01-15T00:00:00Z")  # parsed to aware UTC datetime
Event(ts=1736899200)              # parsed from Unix timestamp
```

Naive datetimes are left naive; aware datetimes keep their timezone. To
normalize everything to UTC, use a `@field_validator`.
