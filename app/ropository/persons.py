from typing import Optional

from sqlalchemy.orm import Session

from app.models.entities import Person


def upsert_person(db: Session, name: str, embedding_text: Optional[str]) -> Person:
    person = db.query(Person).filter(Person.name == name).first()
    if not person:
        person = Person(name=name, embedding=embedding_text)
        db.add(person)
    else:
        person.embedding = embedding_text
    db.commit()
    db.refresh(person)
    return person
