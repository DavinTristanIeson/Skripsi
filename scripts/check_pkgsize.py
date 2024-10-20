# Code from https://stackoverflow.com/questions/34266159/how-to-see-sizes-of-installed-pip-packages

import os
import pkg_resources

def calc_container(path):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      total_size += os.path.getsize(fp)
  return total_size

dists = [d for d in pkg_resources.working_set]
MB = 1_000_000
for dist in dists:
  if dist.location is None:
    continue
  try:
    path = os.path.join(dist.location, dist.project_name)
    size = calc_container(path)
    if size / MB > 1.0:
      print(f"{dist}: {size / MB} MB")
  except OSError:
    '{} no longer exists'.format(dist.project_name)