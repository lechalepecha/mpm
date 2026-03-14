"""
viewer.py  —  Окно просмотра частиц + MPM симуляция (moderngl + pygame)
Запускается как отдельный процесс из ui.py

Два режима работы:
  1. Статический просмотр вокселей (без MPM)
  2. MPM симуляция: взрыв, разрушение, жидкость

Команды от UI (cmd_queue):
  ("color",   [r,g,b])
  ("radius",  float)
  ("quit",)
"""

import math
import sys
import os
import time

import moderngl
import numpy as np
import pygame

# ─── GLSL шейдеры ───────────────────────────────────────────────────────────────

VERT = """
#version 330
uniform mat4  u_mvp;
uniform float u_point_size;
in  vec3  in_pos;
in  float in_dmg;
out vec3  v_pos;
out float v_dmg;
void main() {
    v_pos        = in_pos;
    v_dmg        = in_dmg;
    gl_Position  = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_point_size;
}
"""

FRAG = """
#version 330
in  vec3  v_pos;
in  float v_dmg;
out vec4  frag_color;
uniform vec3  u_cam_pos;
uniform vec3  u_color;
uniform float u_use_damage;
void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float nz = sqrt(1.0 - r2);
    vec3  n  = normalize(vec3(uv.x, -uv.y, nz));
    vec3  L  = normalize(vec3(0.5, 1.0, 0.7));
    float d  = max(dot(n, L), 0.0);
    vec3  H  = normalize(L + normalize(u_cam_pos - v_pos));
    float s  = pow(max(dot(n, H), 0.0), 40.0);
    vec3 col;
    if (u_use_damage > 0.5) {
        float t  = clamp(v_dmg, 0.0, 1.0);
        vec3 c0  = u_color;
        vec3 c1  = vec3(0.90, 0.75, 0.10);
        vec3 c2  = vec3(0.85, 0.15, 0.05);
        if (t < 0.5) col = mix(c0, c1, t * 2.0);
        else         col = mix(c1, c2, (t - 0.5) * 2.0);
    } else {
        col = u_color;
    }
    frag_color = vec4(clamp(col*(0.18+0.72*d)+vec3(0.28*s), 0.0, 1.0), 1.0);
}
"""

HUD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HUD_FRAG = """
#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_tex;
void main() { frag_color = texture(u_tex, v_uv); }
"""

# ─── Матрицы ────────────────────────────────────────────────────────────────────

def _persp(fovy, aspect, near, far):
    f  = 1.0 / math.tan(math.radians(fovy) * 0.5)
    nf = 1.0 / (near - far)
    m  = np.zeros(16, np.float32)
    m[0]=f/aspect; m[5]=f; m[10]=(far+near)*nf; m[11]=-1.0; m[14]=2*far*near*nf
    return m

def _lookat(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.zeros(16, np.float32)
    m[0]=r[0]; m[1]=u[0]; m[2]=-f[0]
    m[4]=r[1]; m[5]=u[1]; m[6]=-f[1]
    m[8]=r[2]; m[9]=u[2]; m[10]=-f[2]
    m[12]=-np.dot(r,eye); m[13]=-np.dot(u,eye); m[14]=np.dot(f,eye); m[15]=1.0
    return m

def _mul(a, b):
    return (a.reshape(4,4,order='F') @ b.reshape(4,4,order='F')).flatten(order='F')

# ─── Камера ─────────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, pos, target):
        self.pos    = np.array(pos,    np.float64)
        self.target = np.array(target, np.float64)
        d = self.target - self.pos
        self.yaw   = math.degrees(math.atan2(d[2], d[0]))
        self.pitch = math.degrees(math.asin(
            np.clip(d[1] / max(np.linalg.norm(d), 1e-6), -1, 1)))
        self._upd()

    def _upd(self):
        yr = math.radians(self.yaw); pr = math.radians(self.pitch)
        self.front = np.array([math.cos(pr)*math.cos(yr),
                               math.sin(pr),
                               math.cos(pr)*math.sin(yr)])
        self.front /= np.linalg.norm(self.front)
        wu = np.array([0., 1., 0.])
        self.right = np.cross(self.front, wu)
        nm = np.linalg.norm(self.right)
        self.right = self.right / nm if nm > 1e-6 else np.array([1.,0.,0.])
        self.up = np.cross(self.right, self.front)

    def rotate(self, dx, dy, s=0.18):
        self.yaw  += dx * s
        self.pitch = max(-89., min(89., self.pitch - dy * s))
        self._upd()

    def move(self, fwd, rt, up, spd):
        self.pos += self.front*fwd*spd + self.right*rt*spd + self.up*up*spd

    def view(self):
        return _lookat(self.pos, self.pos + self.front, np.array([0.,1.,0.]))

# ─── HUD ────────────────────────────────────────────────────────────────────────

def draw_panel(surf, font, font_b, state: dict):
    lines = []
    lines.append(("MPM Симуляция" if state.get("mpm_active") else "Просмотр", True))
    lines.append((f"Частиц: {state['n']:,}", False))
    lines.append((f"FPS:    {state['fps']:.0f}", False))
    if state.get("mpm_active"):
        lines.append((f"Время:  {state['sim_t']:.4f} с", False))
        lines.append(("", False))
        lines.append(("[D] — режим повреждений", False))
        lines.append(("[ПРОБЕЛ] — пауза", False))

    pad_x, pad_y = 10, 10
    bg_w, bg_h   = 210, len(lines) * 18 + 14
    bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
    bg.fill((18, 18, 22, 185))
    surf.blit(bg, (pad_x, pad_y))

    cy = pad_y + 8
    for text, bold in lines:
        if not text:
            cy += 5; continue
        f   = font_b if bold else font
        col = (255, 200, 80) if bold else (185, 188, 200)
        surf.blit(f.render(text, True, col), (pad_x + 8, cy))
        cy += 18

def draw_damage_legend(surf, font, font_b, W, H):
    lx = W - 140; ly = H - 130
    surf.blit(font_b.render("Повреждение", True, (200,200,210)), (lx - 8, ly - 22))
    items = [
        ((185,187,200), "Целый"),
        ((230,192, 25), "Трещины"),
        ((218, 38, 12), "Разрушен"),
    ]
    for i, (col, lbl) in enumerate(items):
        y = ly + i * 34
        pygame.draw.rect(surf, col,          (lx, y, 22, 22))
        pygame.draw.rect(surf, (120,120,130),(lx, y, 22, 22), 1)
        surf.blit(font.render(lbl, True, (200,200,210)), (lx+28, y+3))

# ─── Viewer ─────────────────────────────────────────────────────────────────────

class Viewer:
    W, H = 950, 720

    def __init__(self, particles, color, radius, cmd_queue, mpm_queue=None):
        self.particles  = particles.astype(np.float32)
        self.damage     = np.zeros(len(particles), np.float32)
        self.color      = list(color)
        self.radius     = radius
        self.cmd_q      = cmd_queue
        self.mpm_q      = mpm_queue

        self.mpm_active = mpm_queue is not None
        self.mpm_paused = False
        self.sim_time   = 0.0
        self.use_damage = self.mpm_active
        self.fps        = 0.0
        self._n         = len(particles)

        mn, mx      = particles.min(0), particles.max(0)
        self.center = ((mn + mx) * 0.5).astype(np.float64)
        self.span   = max(float(np.linalg.norm(mx - mn)), 0.01)

        self._mouse_down = False
        self._last_mouse = (0, 0)

        self._init_pygame()
        self._init_gl()
        self._upload(self.particles, self.damage)
        pygame.font.init()
        self.font   = pygame.font.SysFont("segoeui", 14)
        self.font_b = pygame.font.SysFont("segoeui", 15, bold=True)

    def _init_pygame(self):
        pygame.init()
        for a, v in [(pygame.GL_CONTEXT_MAJOR_VERSION, 3),
                     (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                     (pygame.GL_CONTEXT_PROFILE_MASK,
                      pygame.GL_CONTEXT_PROFILE_CORE),
                     (pygame.GL_DEPTH_SIZE, 24)]:
            pygame.display.gl_set_attribute(a, v)
        self.screen = pygame.display.set_mode(
            (self.W, self.H),
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        pygame.display.set_caption("MPM Viewer")

    def _init_gl(self):
        self.ctx  = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        sp = self.span
        self.proj = _persp(60.0, self.W/self.H, 0.001*sp, 20.0*sp)
        cam_pos   = self.center + np.array([0., sp*0.3, sp*1.2])
        self.cam  = Camera(cam_pos, self.center)

        # HUD quad
        self.hud_prog = self.ctx.program(vertex_shader=HUD_VERT,
                                         fragment_shader=HUD_FRAG)
        verts = np.array([
            -1,-1, 0,1,   1,-1, 1,1,   1,1, 1,0,
            -1,-1, 0,1,   1, 1, 1,0,  -1,1, 0,0], np.float32)
        self.hud_vbo = self.ctx.buffer(verts.tobytes())
        self.hud_vao = self.ctx.vertex_array(
            self.hud_prog, [(self.hud_vbo, "2f 2f", "in_pos", "in_uv")])
        self.hud_tex = self.ctx.texture((self.W, self.H), 4)
        self.hud_tex.use(0)
        self.hud_prog["u_tex"].value = 0

    def _upload(self, pts, dmg):
        if hasattr(self, "vbo") and self.vbo:
            try: self.vbo.release(); self.vbo_dmg.release(); self.vao.release()
            except: pass
        self.vbo     = self.ctx.buffer(pts.astype(np.float32).flatten().tobytes())
        self.vbo_dmg = self.ctx.buffer(dmg.astype(np.float32).tobytes())
        self.vao     = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_pos"),
                        (self.vbo_dmg, "1f", "in_dmg")])
        self._n = len(pts)

    def _rebuild_proj(self):
        self.proj = _persp(60.0, self.W/self.H,
                           0.001*self.span, 20.0*self.span)
        self.hud_tex.release()
        self.hud_tex = self.ctx.texture((self.W, self.H), 4)
        self.hud_tex.use(0)

    def run(self):
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(0) / 1000.0
            self.fps = 1.0 / dt if dt > 1e-6 else 999.0

            # Команды от UI
            while not self.cmd_q.empty():
                try:
                    cmd = self.cmd_q.get_nowait()
                    if   cmd[0] == "quit":       pygame.quit(); return
                    elif cmd[0] == "color":      self.color  = list(cmd[1])
                    elif cmd[0] == "radius":     self.radius = float(cmd[1])
                    elif cmd[0] == "use_damage": self.use_damage = bool(cmd[1])
                except: pass

            # Кадры от MPM
            if self.mpm_q is not None:
                while not self.mpm_q.empty():
                    try:
                        msg = self.mpm_q.get_nowait()
                        if msg[0] == "frame":
                            _, pb, sh, db = msg
                            pts = np.frombuffer(pb, np.float32).reshape(sh)
                            dmg = np.frombuffer(db, np.float32)
                            self._upload(pts, dmg)
                        elif msg[0] == "time":
                            self.sim_time = float(msg[1])
                    except: pass

            # События
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit(); return
                    if ev.key == pygame.K_d:
                        self.use_damage = not self.use_damage
                    if ev.key == pygame.K_SPACE and self.mpm_q:
                        self.mpm_paused = not self.mpm_paused
                        try:
                            self.mpm_q.put_nowait(
                                ("pause",) if self.mpm_paused else ("resume",))
                        except: pass
                if ev.type == pygame.VIDEORESIZE:
                    self.W, self.H = ev.w, ev.h
                    self.ctx.viewport = (0, 0, self.W, self.H)
                    self._rebuild_proj()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._mouse_down = True; self._last_mouse = ev.pos
                if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self._mouse_down = False
                if ev.type == pygame.MOUSEMOTION and self._mouse_down:
                    dx = ev.pos[0] - self._last_mouse[0]
                    dy = ev.pos[1] - self._last_mouse[1]
                    self._last_mouse = ev.pos
                    self.cam.rotate(dx, dy)
                if ev.type == pygame.MOUSEWHEEL:
                    self.cam.move(ev.y * self.span * 0.08, 0, 0, 1.0)

            keys = pygame.key.get_pressed()
            sh   = 3.0 if keys[pygame.K_LSHIFT] else 1.0
            spd  = 0.008 * sh * self.span * max(dt*60, 1.0)
            fwd = right = up = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:    fwd   += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  fwd   -= 1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: right += 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:  right -= 1.0
            if keys[pygame.K_e] or keys[pygame.K_SPACE]: up    += 1.0
            if keys[pygame.K_q] or keys[pygame.K_LCTRL]: up    -= 1.0
            self.cam.move(fwd, right, up, spd)

            # Рендер частиц
            self.ctx.clear(0.18, 0.19, 0.22, 1.0)
            d  = max(float(np.linalg.norm(self.cam.pos - self.center)), 0.01)
            px = max(self.H * self.radius /
                     (d * math.tan(math.radians(30.0))), 1.5)
            mvp = _mul(self.proj, self.cam.view())
            self.prog["u_mvp"].write(mvp.tobytes())
            self.prog["u_cam_pos"].write(self.cam.pos.astype(np.float32).tobytes())
            self.prog["u_color"].write(np.array(self.color, np.float32).tobytes())
            self.prog["u_point_size"].value = px
            self.prog["u_use_damage"].value = 1.0 if self.use_damage else 0.0
            self.vao.render(moderngl.POINTS)

            # HUD overlay
            hud = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            hud.fill((0, 0, 0, 0))
            state = dict(mpm_active=self.mpm_active, n=self._n,
                         fps=self.fps, sim_t=self.sim_time,
                         paused=self.mpm_paused)
            draw_panel(hud, self.font, self.font_b, state)
            if self.use_damage:
                draw_damage_legend(hud, self.font, self.font_b, self.W, self.H)

            raw = pygame.image.tostring(hud, "RGBA", False)
            self.hud_tex.write(raw)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.hud_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST)

            caption = (f"MPM Viewer  —  {self._n:,} частиц  |  "
                       f"{self.fps:.0f} FPS")
            if self.mpm_paused: caption += "  [пауза]"
            pygame.display.set_caption(caption)
            pygame.display.flip()


# ─── Точки входа ────────────────────────────────────────────────────────────────

def run_viewer(particles_bytes, shape, color, radius, cmd_queue,
               mpm_queue=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    pts = np.frombuffer(particles_bytes, dtype=np.float32).reshape(shape)
    v   = Viewer(pts, color, radius, cmd_queue, mpm_queue)
    v.run()
