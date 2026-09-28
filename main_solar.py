from __future__ import annotations

from typing import Any
import time
import os

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import Camera2D
from transform import Transform
from node import Node
from shader import Shader
from pipeline import Pipeline
from scene import Scene
from renderer import Renderer
from engine import Engine
from texture import Texture
from textureset import TextureSet
from sampler import Sampler
from square import Square
from disk import Disk

HERE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(HERE, "images")
SHD  = os.path.join(HERE, "shaders", "2d")

# Tamanhos (raio)
R_SOL      = 1.4
R_MERCURIO = 0.35
R_TERRA    = 0.55
R_LUA      = 0.22

# Distâncias orbitais
D_MERCURIO = 3.0
D_TERRA    = 6.0
D_LUA      = 1.4

# Velocidades angulares (graus/segundo)
W_SOL_ROT     =  8.0
W_MERC_TRANS  = 47.0
W_TERRA_TRANS = 29.0
W_TERRA_ROT   = 120.0
W_LUA_TRANS   = 90.0

canvas:   RenderCanvas
device:   wgpu.GPUDevice
context:  Any
renderer: Renderer
camera:   Camera2D
scene:    Scene
background_transform: Transform
last_t:   float = 0.0


class Rotate(Engine):
  """Rotaciona um Transform acumulando ângulo ao longo do tempo."""
  def __init__(self, trf: Transform, speed_deg_s: float) -> None:
    self.trf   = trf
    self.speed = speed_deg_s
    self.angle = 0.0

  def update(self, dt: float) -> None:
    self.angle += self.speed * dt
    self.trf.load_identity()
    self.trf.rotate(self.angle, 0, 0, 1)


class TranslateOrbit(Engine):
  """Orbita circular: gira pivot_trf ao redor da origem, com orbit_trf
  transladando o astro para a distância orbital no eixo X.
  Como a Lua não deve herdar a rotação axial da Terra, ela é filha de
  terra_orbit_node (só translação) e não de terra_body (rotação + translação).
  """
  def __init__(self, pivot_trf: Transform, orbit_trf: Transform,
               dist: float, speed_deg_s: float) -> None:
    self.pivot = pivot_trf
    self.orbit = orbit_trf
    self.speed = speed_deg_s
    self.angle = 0.0
    self.orbit.load_identity()
    self.orbit.translate(dist, 0, 0)

  def update(self, dt: float) -> None:
    self.angle += self.speed * dt
    self.pivot.load_identity()
    self.pivot.rotate(self.angle, 0, 0, 1)


def make_body(dev: wgpu.GPUDevice, shader: Shader,
              radius: float, img: str) -> tuple[Node, Transform]:
  """Cria textura + sampler + TextureSet + nó com Disk escalado.
  Devolve o nó e o Transform de escala (usado pelo engine Rotate da Terra)."""
  tex  = Texture(dev, "tex", img)
  smp  = Sampler(dev, "tex_sampler")
  tset = TextureSet([smp, tex])
  shader.add_texture_set(tset)
  trf = Transform()
  trf.scale(radius, radius, 1)
  return Node(trf=trf, apps=[tset], shps=[Disk(dev)]), trf


def initialize(dev: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene, background_transform

  camera = Camera2D(-10, 10, -10, 10)

  shader_tex = Shader(dev, os.path.join(SHD, "shader_tex.wgsl"))
  shader_tex.set_vertex_buffers([
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ])
  pipeline_tex = Pipeline(shader_tex, target_format, depth_stencil=None)

  # Fundo
  tex_bg  = Texture(dev, "tex", os.path.join(IMG, "espaco.png"))
  smp_bg = Sampler(dev, "tex_sampler", address_mode_u="repeat", address_mode_v="repeat")
  tset_bg = TextureSet([smp_bg, tex_bg])
  shader_tex.add_texture_set(tset_bg)
  trf_bg = Transform()
  trf_bg.scale(12, 12, 1)
  background_transform = trf_bg
  bg_node = Node(trf=trf_bg, apps=[tset_bg], shps=[Square(dev)])

   # Sol
  trf_sol_rot = Transform()
  sol_body, _ = make_body(dev, shader_tex, R_SOL, os.path.join(IMG, "sol.png"))
  sol_node = Node(trf=trf_sol_rot, nodes=[sol_body])
  eng_sol = Rotate(trf_sol_rot, W_SOL_ROT)

  # Mercúrio
  trf_merc_pivot = Transform()
  trf_merc_orbit = Transform()
  merc_body, _ = make_body(dev, shader_tex, R_MERCURIO, os.path.join(IMG, "mercurio.png"))
  merc_node = Node(trf=trf_merc_pivot,
                   nodes=[Node(trf=trf_merc_orbit, nodes=[merc_body])])
  eng_merc = TranslateOrbit(trf_merc_pivot, trf_merc_orbit, D_MERCURIO, W_MERC_TRANS)

  # Terra
  trf_terra_pivot    = Transform()
  trf_terra_orbit    = Transform()
  trf_terra_self_rot = Transform()
  terra_body, trf_terra_scale = make_body(dev, shader_tex, R_TERRA, os.path.join(IMG, "terra.png"))
  # terra_body gira; o disco escalado é filho do nó de rotação
  terra_rot_node   = Node(trf=trf_terra_self_rot, nodes=[terra_body])
  terra_orbit_node = Node(trf=trf_terra_orbit,    nodes=[terra_rot_node])
  terra_node       = Node(trf=trf_terra_pivot,    nodes=[terra_orbit_node])
  eng_terra_trans  = TranslateOrbit(trf_terra_pivot, trf_terra_orbit, D_TERRA, W_TERRA_TRANS)
  eng_terra_rot    = Rotate(trf_terra_self_rot, W_TERRA_ROT)

  # Lua
  trf_lua_pivot = Transform()
  trf_lua_orbit = Transform()
  lua_body, _ = make_body(dev, shader_tex, R_LUA, os.path.join(IMG, "lua.png"))
  lua_node = Node(trf=trf_lua_pivot,
                  nodes=[Node(trf=trf_lua_orbit, nodes=[lua_body])])
  eng_lua = TranslateOrbit(trf_lua_pivot, trf_lua_orbit, D_LUA, W_LUA_TRANS)
  terra_orbit_node.add_node(lua_node)

  # Grafo de cena
  #   root (pipeline)
  #   ├── bg_node
  #   ├── sol_node
  #   │   └── sol_body  (gira)
  #   ├── merc_node
  #   │   └── merc_orbit → merc_body
  #   └── terra_node
  #       └── terra_orbit
  #           ├── terra_rot_node (gira)
  #           │   └── terra_body
  #           └── lua_node
  #               └── lua_orbit → lua_body
  root = Node(pipeline_tex, nodes=[bg_node, sol_node, merc_node, terra_node])
  global scene
  scene = Scene(root)
  scene.add_engine(eng_sol)
  scene.add_engine(eng_merc)
  scene.add_engine(eng_terra_trans)
  scene.add_engine(eng_terra_rot)
  scene.add_engine(eng_lua)


def update(dt: float) -> None:
  scene.update(dt)


def draw() -> None:
  global last_t
  t = time.perf_counter()
  update(t - last_t)
  last_t = t
  target_texture = context.get_current_texture()
  width, height = target_texture.width, target_texture.height
  aspect = width / height
  half_width = 10 * max(1.0, aspect)
  half_height = 10 * max(1.0, 1.0 / aspect)
  background_transform.load_identity()
  background_transform.scale(half_width, half_height, 1)
  renderer.render(target_texture, scene, camera)


def on_key(event: Any) -> None:
  if event["key"] == "q":
    canvas.close()


def main() -> None:
  global canvas, device, context, renderer, last_t

  canvas = RenderCanvas(size=(800, 800),
                        title="Mini Sistema Solar 2D – INF1761",
                        update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device  = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)
  renderer = Renderer(device, clear_value=(0.0, 0.0, 0.05, 1.0))

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  last_t = time.perf_counter()
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()