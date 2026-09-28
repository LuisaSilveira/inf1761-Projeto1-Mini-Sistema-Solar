from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import wgpu
import math
from shape import Shape

if TYPE_CHECKING:
  from state import State

class Disk (Shape):
  """Circle/disk centered at origin with radius 1, tessellated into `nslices`
  triangles. Coord buffer at slot 0 (vec2), texcoord buffer at slot 1 (vec2).
  UV mapping: (s,t) = (0.5 + 0.5*cos(theta), 0.5 - 0.5*sin(theta)) so the
  texture center maps to the disk center."""

  def __init__ (self, device: wgpu.GPUDevice, nslices: int = 64) -> None:
    coords: list[list[float]] = [[0.0, 0.0]]      # center vertex
    texcoords: list[list[float]] = [[0.5, 0.5]]   # center UV

    for i in range(nslices):
      angle = 2 * math.pi * i / nslices
      x = math.cos(angle)
      y = math.sin(angle)
      coords.append([x, y])
      texcoords.append([0.5 + 0.5 * x, 0.5 - 0.5 * y])

    indices: list[int] = []
    for i in range(nslices):
      next_i = (i + 1) % nslices + 1   # wraps: após o último vértice da borda, volta ao primeiro
      indices += [0, i + 1, next_i]

    bcoords = np.array(coords, dtype='float32')
    btexcoords = np.array(texcoords, dtype='float32')
    bindex = np.array(indices, dtype='uint32')

    self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(
      data=bcoords, usage=wgpu.BufferUsage.VERTEX)
    self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(
      data=btexcoords, usage=wgpu.BufferUsage.VERTEX)
    self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(
      data=bindex, usage=wgpu.BufferUsage.INDEX)
    self.nind = len(indices)

  def draw (self, st: State) -> None:
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.coord_vbo)
    st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
    st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
    st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)
