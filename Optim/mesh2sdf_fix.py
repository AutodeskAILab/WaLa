"""
Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
import trimesh
import skimage.measure

import mesh2sdf.core

def compute(vertices: np.ndarray, faces: np.ndarray, size: int = 128,
            fix: bool = False, level: float = 0.015, return_mesh: bool = False, new_fix = False, fix_centering = False):
  r''' Converts a input mesh to signed distance field (SDF).
  Args:
    vertices (np.ndarray): The vertices of the input mesh, the vertices MUST be
        in range [-1, 1].
    faces (np.ndarray): The faces of the input mesh.
    size (int): The resolution of the resulting SDF.
    fix (bool): If the input mesh is not watertight, set :attr:`fix` as True.
    level (float): The value used to extract level sets when :attr:`fix` is True,
        with a default value of 0.015 (as a reference 2/128 = 0.015625). And the
        recommended default value is 2/size.
    return_mesh (bool): If True, also return the fixed mesh.
  '''

  # compute sdf
  sdf = mesh2sdf.core.compute(vertices, faces, size)
  if not fix:
    return (sdf, trimesh.Trimesh(vertices, faces)) if return_mesh else sdf

  # NOTE: the negative value is not reliable if the mesh is not watertight
  sdf = np.abs(sdf)
  vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

  # keep the max component of the extracted mesh
  mesh = trimesh.Trimesh(vertices, faces)

  components = mesh.split(only_watertight=False)

  if new_fix:
    # Autodesk code begins
    # Author: edward1997104 <ka.hei.hui@autodesk.com>
    keep_flags = []

    ## check whether each component contained by others
    for c in components: # if one of them not in others
      bounds = c.bounds
      keep_flag = True
      for c_other in components:
        if c_other != c: #
          inside_flag = trimesh.bounds.contains(c_other.bounds, bounds)
          if all(inside_flag):
            keep_flag = False

      keep_flags.append(keep_flag)

    #process the mesh
    components = [c for (c, f) in zip(components, keep_flags) if f is True]
    print(f"Remaining component len : {len(components)}")
    mesh = trimesh.util.concatenate(components)
    # Autodesk code ends
  else:
    bbox = []
    for c in components:
      bbmin = c.vertices.min(0)
      bbmax = c.vertices.max(0)
      bbox.append((bbmax - bbmin).max())
    max_component = np.argmax(bbox)
    mesh = components[max_component]

  if fix_centering: # fix centering
    mesh.vertices = ((mesh.vertices) * (2.0 / (size - 1)) - 1.0)  # normalize it to [-1, 1]
  else:
    mesh.vertices = ((mesh.vertices) * (2.0 / size) - 1.0)  # normalize it to [-1, 1]

  # re-compute sdf
  sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
  return (sdf, mesh) if return_mesh else sdf