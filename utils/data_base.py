from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy_utils import create_database, database_exists
BASE_PATH = 'sqlite:///database.db'
Base = declarative_base()


class Recipe(Base):
    __tablename__ = 'Recipe'
    id = Column(Integer, primary_key=True)
    calories = Column(Float)
    total_time = Column(String)
    keywords = Column(Text)
    ingredients = Column(Text)
    ingredients_quantity = Column(String)
    fat_content = Column(Float)
    saturated_fat_content = Column(Float)
    cholesterol_content = Column(Float)
    sodium_content = Column(Float)
    carbohydrate_content = Column(Float)
    fiber_content = Column(Float)
    sugar_content = Column(Float)
    protein_content = Column(Float)
    instructions = Column(Text)
    name = Column(String)
    images = relationship('FoodImage')



class FoodImage(Base):
    __tablename__ = 'FoodImage'
    id = Column(Integer, primary_key=True)
    recipe_id = Column(Integer, ForeignKey('Recipe.id'))


def init_base():
    if not database_exists(BASE_PATH):
        create_database(BASE_PATH)
    engine = create_engine(BASE_PATH)
    Base.metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session