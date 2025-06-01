from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, TypeVar
from modules.baseclass import Singleton
from modules.regression.exceptions import MissingStoredRegressionModelException
from modules.storage.cache import CacheClient, CacheItem
if TYPE_CHECKING:
  from statsmodels.regression.linear_model import RegressionResultsWrapper
  from statsmodels.discrete.discrete_model import BinaryResultsWrapper, MultinomialResultsWrapper
  from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

T = TypeVar("T")


@dataclass
class RegressionModelCacheWrapper(Generic[T]):
  model: T
  levels: Optional[list[str]]

@dataclass
class BaseRegressionModelCacheAdapter(Generic[T]):
  cache: CacheClient[RegressionModelCacheWrapper[T]]
  def save(self, model: RegressionModelCacheWrapper[T]):
    import uuid
    id = uuid.uuid4().hex
    self.cache.set(CacheItem(
      key=id,
      value=model,
    ))
    return id
  
  def load(self, id: str)->RegressionModelCacheWrapper[T]:
    model = self.cache.get(id)
    if model is None:
      raise MissingStoredRegressionModelException(id=id)
    return model
  

class RegressionModelCacheManager(metaclass=Singleton):
  linear: BaseRegressionModelCacheAdapter["RegressionResultsWrapper"]
  logistic: BaseRegressionModelCacheAdapter["BinaryResultsWrapper"]
  multinomial_logistic: BaseRegressionModelCacheAdapter["MultinomialResultsWrapper"]
  ordinal: BaseRegressionModelCacheAdapter["OrderedResultsWrapper"]

  def __init__(self):
    self.linear = BaseRegressionModelCacheAdapter(
      cache=CacheClient(name="Linear Regression Model", maxsize=20, ttl=None)
    )
    self.logistic = BaseRegressionModelCacheAdapter(
      cache=CacheClient(name="Logistic Regression Model", maxsize=20, ttl=None)
    )
    self.multinomial_logistic = BaseRegressionModelCacheAdapter(
      cache=CacheClient(name="Multinomial Logistic Regression Model", maxsize=20, ttl=None)
    )
    self.ordinal = BaseRegressionModelCacheAdapter(
      cache=CacheClient(name="Ordinal Regression Model", maxsize=20, ttl=None)
    )
