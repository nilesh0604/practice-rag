---
title: Introduction to SQLModel
source_url: https://sqlmodel.tiangolo.com/
---

# SQLModel

SQLModel is a library for interacting with SQL databases from Python
code, with Python objects. It is designed to be simple, intuitive, and
easy to use, while also being powerful and flexible.

## What is SQLModel?

SQLModel is built on top of **Pydantic** and **SQLAlchemy**. It combines
the validation and serialization of Pydantic with the ORM power of
SQLAlchemy, so you define a single class that serves as both a Pydantic
model and a SQLAlchemy table:

```python
from typing import Optional
from sqlmodel import Field, SQLModel

class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None
```

The `table=True` argument tells SQLModel that this class is a database
table. Without `table=True`, it behaves like a plain Pydantic model
(useful for read/write schemas that exclude database-only fields).

## SQLModel vs SQLAlchemy

SQLAlchemy is a full-featured SQL toolkit and ORM. SQLModel is a thin
wrapper that:

- Uses Pydantic for validation and serialization (so API models and
  database models can share the same class).
- Uses SQLAlchemy's engine and session under the hood.
- Simplifies the most common CRUD patterns.

| Feature              | SQLAlchemy       | SQLModel                  |
| -------------------- | ---------------- | ------------------------- |
| Validation           | Manual / events  | Pydantic (automatic)      |
| Serialization        | Manual           | `model_dump()` / JSON     |
| Learning curve       | Steep            | Gentle                    |
| Flexibility          | Full             | Common patterns covered   |
| Underlying engine    | Its own          | SQLAlchemy                |

If you need advanced SQLAlchemy features (complex joins, polymorphic
inheritance, custom types), you can drop down to SQLAlchemy directly —
SQLModel does not hide it.

## Creating the Database

```python
from sqlmodel import SQLModel, create_engine, Session

engine = create_engine("sqlite:///database.db")

SQLModel.metadata.create_all(engine)
```

This creates all tables defined by your `SQLModel` subclasses. For
production, use Alembic migrations instead of `create_all`.

## Reading and Writing

```python
from sqlmodel import Session, select

with Session(engine) as session:
    hero = Hero(name="Deadpond", secret_name="Dive Wilson")
    session.add(hero)
    session.commit()
    session.refresh(hero)
    print(hero.id)  # auto-generated primary key

    statement = select(Hero).where(Hero.name == "Deadpond")
    results = session.exec(statement)
    db_hero = results.first()
```

## Read vs Write Models

A common pattern is to define a base model without `table=True` for API
schemas, and a table model that inherits from it:

```python
class HeroBase(SQLModel):
    name: str
    secret_name: str
    age: int | None = None

class Hero(HeroBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

class HeroCreate(HeroBase):
    pass

class HeroRead(HeroBase):
    id: int
```

This keeps API input/output schemas separate from the persistence model
while avoiding field duplication.
