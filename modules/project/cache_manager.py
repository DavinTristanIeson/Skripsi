from contextlib import contextmanager
import os
import threading
from modules.baseclass import Singleton
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.project.lock import ProjectThreadLockManager

from watchdog.observers import Observer
from watchdog.events import DirDeletedEvent, DirModifiedEvent, DirMovedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent, FileSystemEventHandler

from modules.project.paths import DATA_DIRECTORY, ProjectPaths
import pathlib

logger = ProvisionedLogger().provision("ProjectCacheManager")
class ProjectCacheInvalidatorEventHandler(FileSystemEventHandler):
  def __init__(self, *, projects: dict[str, ProjectCache]) -> None:
    super().__init__()
    self.projects = projects

  def invalidate_cache_from_event(self, event_path: str | bytes, event_type: str):
    if isinstance(event_path, str):
      path = event_path
    elif isinstance(event_path, bytes):
      path = event_path.decode(encoding='utf-8')
    else:
      return
    
    if path.endswith(".log") or path.endswith(".lock"):
      # Irrelevant. Don't need to preserve.
      return
    
    data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
    relative_path = pathlib.Path(path).relative_to(data_directory)
    if len(relative_path.parts) < 2:
      return
    
    project_id = relative_path.parts[0]
    logger.debug(f"Project cache invalidator will be checking the following path {relative_path.parts} (type: {event_type}) for which cache to invalidate.")

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
      
      # region Test column
      if checked_path != ProjectPaths.TopicModelingFolderName:
        return
      

      # Parts: topic modeling
      if len(path_parts) <= 1:
        project_cache.invalidate_topic_modeling(column=None)
        return

      # Parts: topic modeling, column
      column = path_parts[1]

      if len(path_parts) <= 2:
        # Parts topic modeling
        # Invalidated the folder, but not nested folders.
        project_cache.invalidate_topic_modeling(column=column)
        return
              
      # Format is topic-modeling/{column}/{file}
      checked_path = path_parts[2]
      if checked_path == ProjectPaths.BERTopicFolder:
        project_cache.bertopic_models.invalidate(key=column)
        return
      if checked_path == ProjectPaths.DocumentEmbeddingsFileName:
        project_cache.document_vectors.invalidate(key=column)
        return
      if checked_path == ProjectPaths.UMAPEmbeddingsFileName:
        project_cache.umap_vectors.invalidate(key=column)
        return
      if checked_path == ProjectPaths.VisualizationEmbeddingsFileName:
        project_cache.visualization_vectors.invalidate(key=column)
        return
      if checked_path == ProjectPaths.TopicEvaluationFileName:
        project_cache.topic_evaluations.invalidate(key=column)
        return
      if checked_path == ProjectPaths.TopicModelExperimentsFileName:
        project_cache.bertopic_experiments.invalidate(key=column)
        return
      if checked_path == ProjectPaths.TopicsFileName:
        project_cache.topics.invalidate(key=column)
        return
    return
  
  def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
    super().on_modified(event)
    if event.is_directory:
      return
    self.invalidate_cache_from_event(event_path=event.src_path, event_type=event.event_type)
    
  def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
    super().on_deleted(event)
    self.invalidate_cache_from_event(event_path=event.src_path, event_type=event.event_type)
  
  def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
    super().on_moved(event)
    self.invalidate_cache_from_event(event_path=event.dest_path, event_type=event.event_type)

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