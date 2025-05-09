from contextlib import contextmanager
import os
import threading
from modules.baseclass import Singleton
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.project.lock import ProjectThreadLockManager

from watchdog.observers import Observer
from watchdog.events import DirDeletedEvent, DirModifiedEvent, DirMovedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent, FileSystemEventHandler, FileSystemEvent

from modules.project.paths import DATA_DIRECTORY, ProjectPaths
import pathlib

logger = ProvisionedLogger().provision("ProjectCacheManager")
class ProjectCacheInvalidatorEventHandler(FileSystemEventHandler):
  def __init__(self, *, projects: dict[str, ProjectCache]) -> None:
    super().__init__()
    self.projects = projects

  def invalidate_cache_from_event(self, event: FileSystemEvent):
    path: str
    if isinstance(event.src_path, str):
      path = event.src_path
    elif isinstance(event.src_path, bytes):
      path = event.src_path.decode(encoding='utf-8')
    else:
      return
    
    data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
    relative_path = pathlib.Path(path).relative_to(data_directory)
    if len(relative_path.parts) == 0:
      return
    
    project_id = relative_path.parts[0]
    logger.debug(f"Project cache invalidator will be checking the following path {relative_path.parts} for which cache to invalidate.")

    project_lock = ProjectThreadLockManager().get(project_id)
    with project_lock:
      project_cache = self.projects.get(project_id, None)
      if project_cache is None:
        return
      
      path_parts = list(relative_path.parts[1:])
      checked_path = path_parts[0]
      # Test workspace
      if checked_path == ProjectPaths.Workspace:
        project_cache.workspaces.invalidate()
        return
      # Test config
      if checked_path == ProjectPaths.Config:
        project_cache.config_cache.invalidate()
        return
      # Test column
      # Parts: topic modeling, column, file
      is_topic_modeling_folder = (checked_path == ProjectPaths.TopicModelingFolderName and len(path_parts) >= 3)
      if not is_topic_modeling_folder:
        return
      
      # Format is topic-modeling/{column}/{file}
      checked_path = path_parts[2]
      if checked_path == ProjectPaths.BERTopicFolder:
        project_cache.bertopic_models.invalidate()
        return
      if checked_path == ProjectPaths.VisualizationEmbeddingsFileName:
        project_cache.visualization_vectors.invalidate()
        return
      if checked_path == ProjectPaths.TopicEvaluationFileName:
        project_cache.topic_evaluations.invalidate()
        return
      if checked_path == ProjectPaths.TopicModelExperimentsFileName:
        project_cache.bertopic_experiments.invalidate()
        return
      if checked_path == ProjectPaths.TopicsFileName:
        project_cache.topics.invalidate()
        return
    return
  
  def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
    super().on_modified(event)
    logger.debug(f"Observed event {event.event_type} on {event.src_path}")
    self.invalidate_cache_from_event(event)
    
  def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
    super().on_deleted(event)
    logger.debug(f"Observed event {event.event_type} on {event.src_path}")
    self.invalidate_cache_from_event(event)
  
  def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
    super().on_moved(event)
    logger.debug(f"Observed event {event.event_type} on {event.src_path} (moved to {event.dest_path})")
    self.invalidate_cache_from_event(event)



class ProjectCacheManager(metaclass=Singleton):
  projects: dict[str, ProjectCache]
  lock: threading.RLock
  def __init__(self):
    super().__init__()
    self.projects = dict()
    self.lock = threading.RLock()

  def get(self, project_id: str)->ProjectCache:
    with self.lock:
      cache = self.projects.get(project_id, None)
      if cache is None:
        cache = ProjectCache(
          project_id=project_id,
          lock=ProjectThreadLockManager().get(project_id),
        )
    self.projects[project_id] = cache
    return cache

  def invalidate(self, project_id: str):
    project_cache = self.projects.get(project_id)
    if project_cache is not None:
      project_cache.invalidate()

  @contextmanager
  def run(self):
    observer = Observer()
    event_handler = ProjectCacheInvalidatorEventHandler(
      projects=self.projects,
    )
    observed_path = os.path.join(os.getcwd(), DATA_DIRECTORY)
    observer.schedule(event_handler, path=observed_path, event_filter=[
      DirModifiedEvent,
      DirDeletedEvent,
      FileDeletedEvent,
      FileModifiedEvent,
      FileMovedEvent,
      DirMovedEvent
    ], recursive=True)
    logger.info(f"ProjectCacheManager set to observing {observed_path} for file changes.")
    
    observer.start()
    try:
      yield
    except (Exception, KeyboardInterrupt):
      pass
    logger.info(f"Shutting down Watchdog observer...")
    observer.stop()
    observer.join()

__all__ = [
  "ProjectCacheManager"
]