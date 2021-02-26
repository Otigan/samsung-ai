from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists
BASE_PATH = 'sqlite:///database.db'
Base = declarative_base()


class Calories(Base):
    __tablename__ = 'Calories'
    id = Column(Integer, primary_key=True)
    calories = Column(Float)

def init_base():
    if not database_exists(BASE_PATH):
        create_database(BASE_PATH)
    engine = create_engine(BASE_PATH)
    Base.metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


