from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from typing import List, Optional


Base = declarative_base()


class User(Base):
    __tablename__ = "age_gender_labeled"
    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer, nullable=True)
    ethnicity = Column(Integer, nullable=True)
    gender = Column(Boolean, nullable=True)
    img_name = Column(String, nullable=True)
    pixels = Column(String, nullable=True)


class DBManager:
    def __init__(self, db_url: str) -> None:
        """Initialize the database connection.
        
        Args:
            db_url (str): The database connection URL.
        """
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def create_table_if_not_exists(self) -> None:
        """Create the users table if it does not exist."""
        Base.metadata.create_all(self.engine)
    
    def insert_record(
            self, age: Optional[int] = None,
            ethnicity: Optional[int] = None,
            gender: Optional[bool] = None,
            img_name: Optional[str] = None,
            pixels: Optional[List[int]] = None
        ) -> None:
        """Insert a new record into the 'age_gender_labeled_sample' table.

        Args:
            age (Optional[int]): The age of the user. Defaults to None.
            ethnicity (Optional[int]): The ethnicity of the user (represented as an integer). Defaults to None.
            gender (Optional[bool]): The gender of the user (True for male, False for female). Defaults to None.
            img_name (Optional[str]): The image name associated with the user. Defaults to None.
            pixels (Optional[List[int]]): A list of pixel values representing the user's image. Defaults to None.

        This method creates a new User instance and inserts it into the database.
        """
        session = self.Session()
        new_user = User(age=age, ethnicity=ethnicity, gender=gender, img_name=img_name, pixels=pixels)
        session.add(new_user)
        session.commit()
        session.close()

    def update_user(
        self,
        user_id: int,
        age: Optional[int] = None,
        ethnicity: Optional[int] = None,
        gender: Optional[bool] = None,
        img_name: Optional[str] = None,
        pixels: Optional[List[int]] = None
    ) -> None:
        """Update an existing user's record in the 'age_gender_labeled_sample' table.

        Args:
            user_id (int): The ID of the user to update.
            age (Optional[int]): The new age for the user. Defaults to None.
            ethnicity (Optional[int]): The new ethnicity for the user. Defaults to None.
            gender (Optional[bool]): The new gender for the user (True for male, False for female). Defaults to None.
            img_name (Optional[str]): The new image name for the user. Defaults to None.
            pixels (Optional[List[int]]): The new list of pixel values for the user. Defaults to None.

        This method updates the specified fields for the user with the given user ID,
        only updating those fields for which values are provided.
        """
        session = self.Session()
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            if age is not None:
                user.age = age
            if ethnicity is not None:
                user.ethnicity = ethnicity
            if gender is not None:
                user.gender = gender
            if img_name is not None:
                user.img_name = img_name
            if pixels is not None:
                user.pixels = pixels
            session.commit()
        session.close()
    
    def delete_user(self, user_id: int) -> None:
        """Delete a user from the 'age_gender_labeled_sample' table by their ID.

        Args:
            user_id (int): The ID of the user to delete.

        This method removes a user from the database based on their ID.
        """
        session = self.Session()
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            session.delete(user)
            session.commit()
        session.close()

    def truncate_table(self, table_name: str) -> None:
        """Truncate the specified table.
        
        Args:
            table_name (str): The name of the table to truncate.

        This method removes all rows from the table and resets any auto-incrementing primary key.
        """
        session = self.Session()
        try:
            session.execute(text(f"DELETE FROM {table_name}"))
            session.commit()
        except Exception as e:
            print(f"Error truncating table {table_name}:, {e}")
        finally:
            session.close()
