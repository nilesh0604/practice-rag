---
title: Relationships - SQLModel
source_url: https://sqlmodel.tiangolo.com/tutorial/relationship/
---

# Relationships in SQLModel

SQLModel supports one-to-many and many-to-many relationships using
`Relationship` and `ForeignKey`. The syntax is simpler than raw
SQLAlchemy while using the same underlying engine.

## One-to-Many

A one-to-many relationship is defined with `Relationship` on the parent
and `ForeignKey` on the child:

```python
from typing import Optional, List
from sqlmodel import Field, Relationship, SQLModel

class Team(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    headquarters: str
    heroes: List["Hero"] = Relationship(back_populates="team")

class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None
    team_id: Optional[int] = Field(default=None, foreign_key="team.id")
    team: Optional["Team"] = Relationship(back_populates="heroes")
```

- `foreign_key="team.id"` creates the SQL foreign key column on the
  `hero` table.
- `Relationship(back_populates=...)` links the two Python attributes so
  accessing `team.heroes` loads the related heroes and `hero.team`
  loads the parent team.

## Loading Related Objects

Related objects are loaded lazily by default — they are fetched from the
database only when you access the attribute, and only within an active
session:

```python
with Session(engine) as session:
    team = session.get(Team, 1)
    print(team.heroes)  # triggers a SELECT for heroes where team_id=1
```

If you access `team.heroes` after the session is closed, SQLAlchemy
raises `DetachedInstanceError`. To avoid this, eagerly load
relationships with `selectinload`:

```python
from sqlalchemy.orm import selectinload

statement = select(Team).options(selectinload(Team.heroes))
teams = session.exec(statement).all()
# team.heroes is already loaded; safe to access after session close
```

## Many-to-Many

Many-to-many requires a link table. Define it as a `SQLModel` with
`table=True` and two foreign keys:

```python
class HeroTeamLink(SQLModel, table=True):
    team_id: Optional[int] = Field(default=None, foreign_key="team.id", primary_key=True)
    hero_id: Optional[int] = Field(default=None, foreign_key="hero.id", primary_key=True)

class Team(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    heroes: List["Hero"] = Relationship(back_populates="teams", link_model=HeroTeamLink)

class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    teams: List["Team"] = Relationship(back_populates="heroes", link_model=HeroTeamLink)
```

The `link_model` argument tells SQLModel to use the link table for the
association. The link table's primary key is the composite of both
foreign keys.

## Cascade Deletes

To delete a parent and its children automatically, set `ondelete` on
the foreign key and configure the relationship cascade:

```python
class Team(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    heroes: List["Hero"] = Relationship(
        back_populates="team",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

class Hero(SQLModel, table=True):
    team_id: Optional[int] = Field(default=None, foreign_key="team.id", ondelete="CASCADE")
    team: Optional["Team"] = Relationship(back_populates="heroes")
```

Deleting the team via `session.delete(team)` will now also delete all
its heroes in the same transaction.
