import tkinter as tk
from tkinter import ttk
import taichi as ti
import numpy as np
import utils
from engine.mpm_solver import MPMSolver
import threading

class MPMInterface:
    def __init__(self):

        ti.init(arch=ti.gpu, device_memory_GB=10.0)
        self.mpm = MPMSolver(res=(64, 64, 64), size=10, max_num_particles=2**20, use_ggui=True)
        self.mpm.add_cube(lower_corner=[2, 1, 3], cube_size=[1, 1, 3], material=self.mpm.material_snow)
        self.mpm.set_gravity((0, 0, 0))
        
        self.res = (1200, 800)
        self.window = ti.ui.Window("SIMULATION", self.res, vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.camera.position(8.0, 4.5, -1.5)
        self.camera.lookat(4.25, 1.89, 3.7)
        self.camera.fov(55)
        
        self.material_type_colors = np.array([
            [0.1, 0.1, 1.0, 0.8], [236/255,84/255,59/255,1.0], 
            [1.0,1.0,1.0,1.0], [1.0,1.0,0.0,1.0]
        ], dtype=np.float32)
        
        self.particles_radius = 0.02
        self.simulation_running = False
        
        # Параметры
        self.params = {
            'material': self.mpm.material_snow, 'E': 500, 'nu': 0.2, 'rho': 1000,
            'dt_scale': 1.0, 'gravity_y': -200.0, 'num_particles': 8000,
            'plasticity_min_strain': -0.025, 'plasticity_max_strain': 0.0045
        }
        self.material_names = ['Water', 'Elastic', 'Snow', 'Sand']
        
        self.setup_taichi_ui()
    
    def setup_taichi_ui(self):
        @ti.kernel
        def set_color(ti_color: ti.template(), material_color: ti.types.ndarray(), ti_material: ti.template()):
            for I in ti.grouped(ti_material):
                material_id = ti_material[I]
                color_4d = ti.Vector([0.0, 0.0, 0.0, 1.0])
                for d in ti.static(range(3)):
                    color_4d[d] = material_color[material_id, d]
                ti_color[I] = color_4d
        
        self.set_color = set_color
    
    def render(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        
        self.set_color(self.mpm.color_with_alpha, self.material_type_colors, self.mpm.material)
        self.scene.particles(self.mpm.x, per_vertex_color=self.mpm.color_with_alpha, radius=self.particles_radius)
        
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(2.5, 8.0, 2.5), color=(1.0, 1.0, 0.9))
        self.scene.point_light(pos=(6.0, 8.0, 6.0), color=(0.9, 0.95, 1.0))
        
        self.canvas.scene(self.scene)
    
    def render_ui(self):

        self.window.GUI.begin("MPM", 0.02, 0.02, 0.32, 0.75)
        
        self.window.GUI.text("Elastisity:")
        self.params['E'] = self.window.GUI.slider_float("E (Young)", self.params['E'], 1, 1e3)
        self.params['nu'] = self.window.GUI.slider_float("v (Poisson)", self.params['nu'], 0.0, 0.49)
        self.params['rho'] = self.window.GUI.slider_float("ρ (Density)", self.params['rho'], 100, 10000)
        
        self.window.GUI.text("Plastisity:")
        min_strain = self.window.GUI.slider_float("Min ε (Compression)", self.params['plasticity_min_strain'], -0.2, 0.0)
        max_strain = self.window.GUI.slider_float("Max ε (Tension)", self.params['plasticity_max_strain'], 0.0, 0.2)
        
        self.window.GUI.text("Time/Gravity:")
        self.params['dt_scale'] = self.window.GUI.slider_float("dt (Time step)", self.params['dt_scale'], 0.1, 2.0)
        self.params['gravity_y'] = self.window.GUI.slider_float("g (Gravity)", self.params['gravity_y'], -1000, 0)


        self.window.GUI.text("Particle settings:")
        self.params['num_particles'] = self.window.GUI.slider_int("Particles", self.params['num_particles'], 1000, 1000000)

        self.mpm.n_particles[None] = int(self.params['num_particles'])

        real_n = self.mpm.n_particles[None]
        max_n = self.mpm.max_num_particles
        self.window.GUI.text(f"Set: {real_n}/{max_n}")
        if self.window.GUI.button("Rebuild Particles"):
            self.rebuild_particles(self.params['num_particles'])

        self.particles_radius = self.window.GUI.slider_float("Radius", self.particles_radius, 0.005, 0.08)

        if self.window.GUI.button("START"):
            self.simulation_running = True
        
        if self.window.GUI.button("STOP"):
            self.simulation_running = False
        
        self.window.GUI.end()


    def rebuild_particles(self, target_n):
        self.simulation_running = False
        self.mpm.n_particles[None]  = 0
        vol = 1 * 1 * 3
        dx = self.mpm.dx
        density = max(1.0, target_n * (dx ** 3) / vol)
        
        self.mpm.add_cube(lower_corner=[2, 1, 3], cube_size=[1, 1, 3], material=self.mpm.material_snow, sample_density=density)
        print(f"Rebuilt: target={target_n}, actual={self.mpm.n_particles[None]}")
        
    
    def run(self):
        while self.window.running:
            self.canvas.set_background_color((0.52, 0.54, 0.6))
            
            if self.simulation_running:
                dt = 4e-3 * self.params['dt_scale']
                self.mpm.set_gravity((0, self.params['gravity_y'], 0))
                self.mpm.step(dt)
            
            self.render()
            self.render_ui()
            self.window.show()

if __name__ == "__main__":
    interface = MPMInterface()
    interface.run()
