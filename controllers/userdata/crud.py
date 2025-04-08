from dataclasses import dataclass
from http import HTTPStatus
import json
import os
import threading
from typing import Any, Callable, Generic, Optional, TypeVar, cast

from fastapi.encoders import jsonable_encoder
import pydantic

from models.userdata import UserDataResource, UserDataSchema
from modules.api.wrapper import ApiError
from modules.baseclass import Singleton
from modules.logger.provisioner import ProvisionedLogger


T = TypeVar("T")
JSONStorageControllerValidator = Callable[[Any], UserDataResource[T]]

logger = ProvisionedLogger().provision("JSONStorageController")

        
class _JSONStorageLockManager(metaclass=Singleton):
  __locks: dict[str, threading.Lock]
  def __init__(self):
    self.__locks = {}

  def provision(self, storage_path: str):
    lock = self.__locks.get(storage_path, None)
    if lock is None:
      lock = threading.Lock()
      self.__locks[storage_path] = lock
    return lock


# CRUD shouldn't occur too open, so it should be safe reading to and from the storage controller. Besides, we got caching in FE.
# Plus, using JSON is better since we already have pydantic as a validator.
# The data doesn't have much relationship with each other and is mostly hierarchical in nature. Plus we got validation in check to ensure that table filters are valid.
# Plus, imagine trying to encode the hierarchical table filter structure in SQL. Hell to the no. We're using JSON, no SQL.

@dataclass
class JSONStorageController(Generic[T]):
  path: str
  validator: JSONStorageControllerValidator

  @property
  def lock(self):
    return _JSONStorageLockManager().provision(self.path)

  def __internal_validate(self, item: Any)->Optional[UserDataResource[T]]:
    try:
      return self.validator(item)
    except pydantic.ValidationError:
      logger.error(f"Discarding invalid entry in \"{self.path}\" with the following shape: {item}")
      return None
    
  def __assert_uniqueness_constraint(self, items: list[UserDataResource[T]], candidate: UserDataResource[T])->Optional[UserDataResource]:
    unique_ids = set(map(lambda x: x.id, items))
    unique_names = set(map(lambda x: x.name, items))
    if candidate.id in unique_ids:
      raise ApiError(f"ID \"{candidate.id}\" already exists.", HTTPStatus.UNPROCESSABLE_ENTITY)
    if candidate.name in unique_names:
      raise ApiError(f"The name \"{candidate.name}\" already exists. Please choose another name.", HTTPStatus.UNPROCESSABLE_ENTITY)

  def read_file(self)->list[UserDataResource[T]]:
    if not os.path.exists(self.path):
      logger.info(f"{self.path} doesn't exist yet.")
      return []
    with open(self.path, "r") as f:
      try:
        contents = json.load(f)
      except Exception:
        logger.error(f"Failed to load {self.path}. The file may be corrupted. Clearing all data.")
        return []

    if not isinstance(contents, list):
      logger.warning(f"The contents of \"{self.path}\" ({contents}) is not a valid list. We will consider it as corrupt and thus throw away the contents.")
      return []
  
    return cast(list[UserDataResource[T]], list(filter(lambda x: x is not None, map(self.__internal_validate, contents))))
  
  def write_file(self, contents: list[UserDataResource[T]]):
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    with open(self.path, "w") as f:
      json.dump(jsonable_encoder(contents), f)

  def all(self)->list[UserDataResource[T]]:
    logger.info(f"{self.path} - GET ALL")
    with self.lock:
      current_state = self.read_file()
      return current_state
    
  def get(self, id: str)->Optional[UserDataResource[T]]:
    logger.info(f"{self.path} - GET {id}")
    with self.lock:
      current_state = self.read_file()
      for item in current_state:
        if item.id == id:
          return item
      return None

  def create(self, item: UserDataSchema[T]):
    logger.info(f"{self.path} - CREATE {item}")
    import uuid
    addition = UserDataResource[T].from_schema(item, uuid.uuid4().hex)
    # Ensure that Generic is properly resolved. Pydantic cannot resolve the nested T generic up there.
    addition = self.validator(addition)
    with self.lock:
      current_state = self.read_file()
      self.__assert_uniqueness_constraint(current_state, addition)
      current_state.append(addition)
      self.write_file(current_state)

  def update(self, id: str, item: UserDataSchema[T]):
    logger.info(f"{self.path} - UPDATE {id} WITH {item}")
    with self.lock:
      current_state = self.read_file()
      __has_update = False
      for index in range(len(current_state)):
        candidate = current_state[index]
        if candidate.id == id:
          new_data = UserDataResource[T].from_schema(item, id)
          new_data = self.validator(new_data)
          current_state.pop(index)
          self.__assert_uniqueness_constraint(current_state, new_data)
          current_state.insert(index, new_data)
          __has_update = True
          break

      # If no update just treat this as a create
      if not __has_update:
        raise ApiError(f"There are no entries with ID \"{id}\" in \"{self.path}\"", HTTPStatus.NOT_FOUND)

      self.write_file(current_state)
    
  def delete(self, id: str):
    logger.info(f"{self.path} - DELETE {id}")
    with self.lock:
      current_state = self.read_file()
      __has_delete = False
      for index in range(len(current_state)):
        candidate = current_state[index]
        if candidate.id == id:
          current_state.pop(index)
          __has_delete = True
          break

      if not __has_delete:
        raise ApiError(f"There are no entries with ID \"{id}\" in \"{self.path}\"", HTTPStatus.NOT_FOUND)

      self.write_file(current_state)
  
__all__ = [
  "JSONStorageController",
]