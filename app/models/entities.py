from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.db.MySQL import Base


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    # Optional serialized embedding for later fine-tuning or re-indexing.
    embedding = Column(Text, nullable=True)
    appearances = relationship("Appearance", back_populates="person", cascade="all, delete-orphan")


class Appearance(Base):
    __tablename__ = "appearances"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    name = Column(String(255), nullable=False, index=True)  # cached label from ArcFace
    appeared_at = Column(DateTime, default=datetime.utcnow, index=True)

    person = relationship("Person", back_populates="appearances")


class AttendanceStat(Base):
    __tablename__ = "attendance_stats"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False, unique=True)
    on_time_days = Column(Integer, default=0)
    late_days = Column(Integer, default=0)
    early_leave_days = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person")
