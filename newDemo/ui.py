"""
ui.py  —  Главный UI (tkinter)
Вокселизация в отдельном процессе, просмотрщик открывается отдельным окном.
"""

import multiprocessing as mp
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

# =============================================================================
#  Вокселизация в отдельном процессе
# =============================================================================

def _vox_worker(mesh_path: str, params: dict, q: mp.Queue):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import taichi as ti
        ti.init(arch=ti.gpu, device_memory_GB=4,
                offline_cache=True, debug=False)
        import voxelizer as VOX
        import numpy as np, time

        q.put(("progress", 0.02, "Загрузка меша..."))
        verts, tris = VOX.load_mesh(mesh_path)
        q.put(("progress", 0.06,
               f"Меш: {len(verts):,} вершин, {len(tris):,} треугольников"))

        res        = params["res"]
        n_cams     = params["cameras"]
        voxel_size = params["grid_size"] / res
        origin_v   = ti.math.vec3(*[-params["grid_size"] / 2.0] * 3)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        cameras = VOX.fibonacci_sphere(n_cams, params["cam_radius"])

        grid = ti.field(dtype=ti.i32, shape=(res,) * 3)
        VOX.kernel_init_solid(grid)
        ti.sync()
        q.put(("progress", 0.09, "Грид инициализирован"))

        for ci, cam_np in enumerate(cameras):
            cam_v = ti.math.vec3(*cam_np.tolist())
            t0 = time.perf_counter()
            VOX.kernel_carve_one_camera(
                grid, res, voxel_size, origin_v, cam_v, v0, v1, v2)
            ti.sync()
            VOX.throttle_sleep(time.perf_counter() - t0, params["load"])
            prog = 0.09 + 0.84 * (ci + 1) / n_cams
            q.put(("progress", prog, f"Камера {ci+1}/{n_cams}"))

        q.put(("progress", 0.94, "Сбор частиц..."))
        n_solid = VOX.kernel_count(grid)
        ti.sync()

        if n_solid == 0:
            q.put(("error", "Нет SOLID вокселей — увеличь cam_radius"))
            return

        counter = ti.field(dtype=ti.i32, shape=())
        counter[None] = 0
        out_buf = ti.ndarray(dtype=ti.math.vec3, shape=n_solid)
        VOX.kernel_collect(grid, res, voxel_size, origin_v, out_buf, counter)
        ti.sync()

        pts = out_buf.to_numpy()[:int(counter[None])].astype(np.float32)
        q.put(("progress", 1.0, f"Готово: {len(pts):,} частиц"))
        q.put(("result", pts.tobytes(), pts.shape))

    except Exception:
        import traceback
        q.put(("error", traceback.format_exc()))


def _viewer_proc(particles_bytes, shape, color, radius, cmd_queue,
                 mpm_queue=None):
    """Запускает viewer в дочернем процессе."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from viewer import run_viewer
    run_viewer(particles_bytes, shape, color, radius, cmd_queue, mpm_queue)


def _mpm_worker(particles_bytes, shape, mat_params, exp_params,
                result_queue, cmd_queue):
    """Запускает MPM симулятор в дочернем процессе."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpm import run_mpm
    run_mpm(particles_bytes, shape, mat_params, exp_params,
            result_queue, cmd_queue)


# =============================================================================
#  UI
# =============================================================================

MAT_PRESETS = {
    "Бетон":  dict(mat_type=0, density=2300, E=3.0e10, nu=0.20, friction=0.60,
                   cohesion=2.0e6, hardening=0.0, bulk=0.0, gamma=0.0,
                   yield_stress=14.5e6, color="#b8b8b2"),
    "Камень": dict(mat_type=0, density=2600, E=5.0e10, nu=0.25, friction=0.70,
                   cohesion=5.0e6, hardening=0.0, bulk=0.0, gamma=0.0,
                   yield_stress=25.0e6, color="#8c8480"),
    "Дерево": dict(mat_type=1, density=600,  E=1.0e10, nu=0.30, friction=0.50,
                   cohesion=8.0e5, hardening=0.1, bulk=0.0, gamma=0.0,
                   yield_stress=5.0e6, color="#a67240"),
    "Металл": dict(mat_type=1, density=7800, E=2.0e11, nu=0.30, friction=0.40,
                   cohesion=0.0,   hardening=0.5, bulk=0.0, gamma=0.0,
                   yield_stress=250e6, color="#c0c2cc"),
    "Кирпич": dict(mat_type=0, density=1800, E=1.5e10, nu=0.22, friction=0.65,
                   cohesion=1.5e6, hardening=0.0, bulk=0.0, gamma=0.0,
                   yield_stress=10.0e6, color="#b85f3f"),
    "Нефть":  dict(mat_type=2, density=850,  E=0.0,    nu=0.0,  friction=0.0,
                   cohesion=0.0,   hardening=0.0, bulk=1.0e6, gamma=7.0,
                   yield_stress=0.0, color="#2a1a05"),
}


def hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]

def rgb01_to_hex(c):
    return "#{:02x}{:02x}{:02x}".format(
        int(c[0]*255), int(c[1]*255), int(c[2]*255))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MPM Voxelizer")
        self.resizable(False, False)
        self.configure(bg="#2b2b2b")

        # Состояние
        self._mesh_path   = tk.StringVar()
        self._vp_res      = tk.IntVar(value=64)
        self._vp_cameras  = tk.IntVar(value=64)
        self._vp_radius   = tk.DoubleVar(value=3.5)
        self._vp_grid     = tk.DoubleVar(value=2.4)
        self._vp_load     = tk.DoubleVar(value=0.8)

        self._mat_color   = "#b8b8b2"
        self._mat_radius  = tk.DoubleVar(value=0.012)
        self._mat_density = tk.DoubleVar(value=2000.0)
        self._mat_E       = tk.DoubleVar(value=3.0e10)
        self._mat_nu      = tk.DoubleVar(value=0.2)
        self._mat_frict   = tk.DoubleVar(value=0.6)
        self._mat_cohes   = tk.DoubleVar(value=2.0e6)
        self._mat_hard    = tk.DoubleVar(value=0.0)

        self._job_proc    = None   # mp.Process вокселизации
        self._job_q       = None   # mp.Queue вокселизации
        self._viewer_proc = None   # mp.Process просмотрщика
        self._viewer_q    = None   # mp.Queue команд просмотрщику
        self._particles   = None   # последний результат (bytes, shape)

        self._mpm_proc    = None   # mp.Process MPM симулятора
        self._mpm_q_cmd   = None   # команды -> MPM
        self._mpm_q_res   = None   # кадры MPM -> viewer

        # MPM параметры взрыва
        self._exp_pressure = tk.DoubleVar(value=2.0e9)
        self._exp_radius   = tk.DoubleVar(value=0.35)
        self._exp_delay    = tk.DoubleVar(value=0.02)
        self._exp_duration = tk.DoubleVar(value=0.03)
        self._exp_cx = tk.DoubleVar(value=0.0)
        self._exp_cy = tk.DoubleVar(value=0.0)
        self._exp_cz = tk.DoubleVar(value=0.0)

        # Тип материала MPM
        self._mat_type  = tk.StringVar(value="Бетон")
        self._mat_bulk  = tk.DoubleVar(value=1.0e6)
        self._mat_gamma = tk.DoubleVar(value=7.0)

        self._build_ui()
        self._poll()   # запускаем периодический опрос очереди

    # ── построение UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",        background="#2b2b2b", foreground="#e0e0e0",
                                font=("Segoe UI", 10))
        s.configure("TLabel",   background="#2b2b2b", foreground="#e0e0e0")
        s.configure("TFrame",   background="#2b2b2b")
        s.configure("TLabelframe", background="#2b2b2b", foreground="#aaaaaa")
        s.configure("TLabelframe.Label", background="#2b2b2b",
                                         foreground="#aaaaaa")
        s.configure("TButton",  background="#3c3f41", foreground="#e0e0e0",
                                padding=4)
        s.map("TButton", background=[("active", "#4c5052")])
        s.configure("TEntry",   fieldbackground="#3c3f41",
                                foreground="#e0e0e0", insertcolor="#e0e0e0")
        s.configure("TScale",   background="#2b2b2b", troughcolor="#3c3f41")
        s.configure("Accent.TButton", background="#2d6db5",
                                      foreground="white", padding=6)
        s.map("Accent.TButton", background=[("active", "#3a80cc")])
        s.configure("TProgressbar", troughcolor="#3c3f41",
                                    background="#2d6db5")

        pad = dict(padx=8, pady=4)

        # ── Файл ──────────────────────────────────────────────────────────────
        f_file = ttk.LabelFrame(self, text="Файл модели", padding=8)
        f_file.pack(fill="x", padx=10, pady=(10, 4))

        row = ttk.Frame(f_file); row.pack(fill="x")
        ttk.Entry(row, textvariable=self._mesh_path, width=36).pack(
            side="left", padx=(0, 4))
        ttk.Button(row, text="Обзор...",
                   command=self._browse).pack(side="left")

        # ── Вокселизация ──────────────────────────────────────────────────────
        f_vox = ttk.LabelFrame(self, text="Вокселизация", padding=8)
        f_vox.pack(fill="x", padx=10, pady=4)

        self._slider(f_vox, "Разрешение",   self._vp_res,
                     32, 256, integer=True)
        self._slider(f_vox, "Камеры",       self._vp_cameras,
                     8, 256, integer=True)
        self._slider(f_vox, "Радиус камер", self._vp_radius, 1.5, 6.0)
        self._slider(f_vox, "Размер грида", self._vp_grid,   1.5, 4.0)
        self._slider(f_vox, "Нагрузка GPU", self._vp_load,   0.1, 1.0)

        self._btn_vox = ttk.Button(
            f_vox, text="Вокселизировать",
            style="Accent.TButton", command=self._start_vox)
        self._btn_vox.pack(fill="x", pady=(6, 0))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            f_vox, variable=self._progress_var,
            maximum=1.0, length=300)
        self._progress_bar.pack(fill="x", pady=(4, 0))

        self._status_var = tk.StringVar(value="")
        ttk.Label(f_vox, textvariable=self._status_var,
                  foreground="#888888").pack(anchor="w")

        # ── Материал ──────────────────────────────────────────────────────────
        f_mat = ttk.LabelFrame(self, text="Материал (MPM)", padding=8)
        f_mat.pack(fill="x", padx=10, pady=4)

        # Пресеты
        row_pre = ttk.Frame(f_mat); row_pre.pack(fill="x", pady=(0, 6))
        ttk.Label(row_pre, text="Пресет:").pack(side="left")
        for name in MAT_PRESETS:
            ttk.Button(row_pre, text=name, width=7,
                       command=lambda n=name: self._apply_preset(n)
                       ).pack(side="left", padx=2)

        # Цвет
        row_col = ttk.Frame(f_mat); row_col.pack(fill="x", pady=2)
        ttk.Label(row_col, text="Цвет:").pack(side="left")
        self._color_btn = tk.Button(
            row_col, bg=self._mat_color, width=4, height=1,
            relief="flat", cursor="hand2",
            command=self._pick_color)
        self._color_btn.pack(side="left", padx=6)
        self._color_hex = ttk.Label(row_col, text=self._mat_color,
                                    foreground="#888888")
        self._color_hex.pack(side="left")

        # Радиус частиц — с живым обновлением
        self._slider(f_mat, "Радиус частиц", self._mat_radius,
                     0.001, 0.10, fmt=".4f",
                     command=self._send_radius)

        ttk.Separator(f_mat, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(f_mat, text="Физика MPM",
                  foreground="#aaaaaa").pack(anchor="w")

        self._slider(f_mat, "Плотность (кг/м³)", self._mat_density,
                     100, 10000, fmt=".0f")
        self._slider(f_mat, "Модуль Юнга (ГПа)", self._mat_E,
                     1e8, 3e11, fmt=".2e")
        self._slider(f_mat, "Пуассон ν",          self._mat_nu,
                     0.0, 0.499, fmt=".3f")

        ttk.Separator(f_mat, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(f_mat, text="Разрушение (Drucker–Prager)",
                  foreground="#aaaaaa").pack(anchor="w")

        self._slider(f_mat, "Трение tan(φ)", self._mat_frict,
                     0.0, 1.2, fmt=".3f")
        self._slider(f_mat, "Когезия (МПа)", self._mat_cohes,
                     0, 2e7, fmt=".2e")
        self._slider(f_mat, "Упрочнение",    self._mat_hard,
                     -1.0, 1.0, fmt=".3f")

        # Кнопки
        row_bot = ttk.Frame(f_mat); row_bot.pack(fill="x", pady=(6, 0))
        ttk.Button(row_bot, text="Сбросить",
                   command=self._reset_mat).pack(side="left", padx=(0, 4))
        ttk.Button(row_bot, text="В консоль",
                   command=self._print_mat).pack(side="left")

        # ── MPM Симуляция ─────────────────────────────────────────────────────
        f_mpm = ttk.LabelFrame(self, text="MPM Симуляция", padding=8)
        f_mpm.pack(fill="x", padx=10, pady=4)

        # Тип материала
        row_mt = ttk.Frame(f_mpm); row_mt.pack(fill="x", pady=(0,4))
        ttk.Label(row_mt, text="Материал:").pack(side="left")
        for nm in MAT_PRESETS:
            ttk.Button(row_mt, text=nm, width=7,
                       command=lambda n=nm: self._apply_preset(n)
                       ).pack(side="left", padx=2)

        ttk.Separator(f_mpm, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(f_mpm, text="Параметры взрыва",
                  foreground="#aaaaaa").pack(anchor="w")

        self._slider(f_mpm, "Давление (Па)",   self._exp_pressure,
                     1e7, 1e11, fmt=".2e")
        self._slider(f_mpm, "Радиус взрыва",   self._exp_radius,
                     0.01, 1.0, fmt=".3f")
        self._slider(f_mpm, "Задержка (с)",    self._exp_delay,
                     0.0, 0.5, fmt=".3f")
        self._slider(f_mpm, "Длительность (с)",self._exp_duration,
                     0.001, 0.1, fmt=".3f")

        ttk.Separator(f_mpm, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(f_mpm, text="Центр взрыва (x, y, z)",
                  foreground="#aaaaaa").pack(anchor="w")

        row_exp_c = ttk.Frame(f_mpm); row_exp_c.pack(fill="x", pady=2)
        for lbl, var in [("X", self._exp_cx), ("Y", self._exp_cy),
                         ("Z", self._exp_cz)]:
            ttk.Label(row_exp_c, text=lbl, width=2).pack(side="left")
            ttk.Entry(row_exp_c, textvariable=var, width=7).pack(
                side="left", padx=(0, 8))

        row_mpm_btns = ttk.Frame(f_mpm); row_mpm_btns.pack(fill="x", pady=(6,0))
        self._btn_mpm = ttk.Button(
            row_mpm_btns, text="Запустить симуляцию",
            style="Accent.TButton", command=self._start_mpm)
        self._btn_mpm.pack(side="left", fill="x", expand=True, padx=(0,4))
        ttk.Button(row_mpm_btns, text="Стоп",
                   command=self._stop_mpm).pack(side="left")

        self._mpm_status = ttk.Label(f_mpm, text="", foreground="#888888")
        self._mpm_status.pack(anchor="w", pady=(4,0))

        # ── Нижняя строка ─────────────────────────────────────────────────────
        self._footer = ttk.Label(self, text="", foreground="#555555")
        self._footer.pack(anchor="w", padx=10, pady=(0, 8))

    # ── хелперы UI ────────────────────────────────────────────────────────────

    def _slider(self, parent, label, var, lo, hi,
                integer=False, fmt=".2f", command=None):
        row = ttk.Frame(parent); row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")
        val_lbl = ttk.Label(row, width=10, anchor="e")
        val_lbl.pack(side="right")

        def _update(v=None):
            val = var.get()
            val_lbl.config(text=format(val, fmt))
            if command:
                command(val)

        sc = ttk.Scale(row, from_=lo, to=hi, variable=var,
                       orient="horizontal", command=_update)
        sc.pack(side="left", fill="x", expand=True, padx=4)
        _update()

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Выберите 3D модель",
            filetypes=[("3D модели", "*.glb *.obj *.ply *.stl"),
                       ("Все файлы", "*.*")])
        if path:
            self._mesh_path.set(path)

    def _pick_color(self):
        result = colorchooser.askcolor(
            color=self._mat_color, title="Цвет материала")
        if result and result[1]:
            self._mat_color = result[1]
            self._color_btn.config(bg=self._mat_color)
            self._color_hex.config(text=self._mat_color)
            self._send_color()

    def _apply_preset(self, name):
        p = MAT_PRESETS[name]
        self._mat_type.set(name)
        self._mat_density.set(p["density"])
        self._mat_E.set(p["E"])
        self._mat_nu.set(p["nu"])
        self._mat_frict.set(p["friction"])
        self._mat_cohes.set(p["cohesion"])
        self._mat_hard.set(p["hardening"])
        self._mat_bulk.set(p.get("bulk", 0.0))
        self._mat_gamma.set(p.get("gamma", 0.0))
        self._mat_color = p["color"]
        self._color_btn.config(bg=self._mat_color)
        self._color_hex.config(text=self._mat_color)
        self._send_color()

    def _reset_mat(self):
        self._apply_preset("Бетон")
        self._mat_radius.set(0.012)

    def _print_mat(self):
        print("\n[MPM Material]")
        print(f"  color     = {self._mat_color}")
        print(f"  radius    = {self._mat_radius.get():.4f}")
        print(f"  density   = {self._mat_density.get():.0f}")
        print(f"  E         = {self._mat_E.get():.3e}")
        print(f"  nu        = {self._mat_nu.get():.3f}")
        print(f"  friction  = {self._mat_frict.get():.3f}")
        print(f"  cohesion  = {self._mat_cohes.get():.3e}")
        print(f"  hardening = {self._mat_hard.get():.3f}")

    # ── команды просмотрщику ──────────────────────────────────────────────────

    def _send_color(self, _=None):
        if self._viewer_q:
            self._viewer_q.put(("color", hex_to_rgb01(self._mat_color)))

    def _send_radius(self, val=None):
        if self._viewer_q:
            self._viewer_q.put(("radius", self._mat_radius.get()))

    # ── вокселизация ──────────────────────────────────────────────────────────

    def _start_vox(self):
        path = self._mesh_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", "Файл не найден:\n" + path)
            return

        # Убиваем предыдущие процессы
        if self._job_proc and self._job_proc.is_alive():
            self._job_proc.terminate()
        if self._viewer_proc and self._viewer_proc.is_alive():
            self._viewer_proc.terminate()
            self._viewer_proc = None
            self._viewer_q    = None

        params = dict(
            res       = self._vp_res.get(),
            cameras   = self._vp_cameras.get(),
            cam_radius= self._vp_radius.get(),
            grid_size = self._vp_grid.get(),
            load      = self._vp_load.get(),
        )
        self._job_q    = mp.Queue()
        self._job_proc = mp.Process(
            target=_vox_worker,
            args=(path, params, self._job_q),
            daemon=True)
        self._job_proc.start()

        self._btn_vox.state(["disabled"])
        self._progress_var.set(0.0)
        self._status_var.set("Запуск...")
        self._footer.config(text="")

    def _open_viewer(self, particles_bytes, shape, mpm_queue=None):
        if self._viewer_proc and self._viewer_proc.is_alive():
            self._viewer_proc.terminate()

        self._viewer_q = mp.Queue()
        color  = hex_to_rgb01(self._mat_color)
        radius = self._mat_radius.get()

        self._viewer_proc = mp.Process(
            target=_viewer_proc,
            args=(particles_bytes, shape, color, radius,
                  self._viewer_q, mpm_queue),
            daemon=True)
        self._viewer_proc.start()

    def _build_mat_params(self) -> dict:
        """Собирает словарь параметров материала для MPM."""
        name = self._mat_type.get()
        p    = MAT_PRESETS.get(name, MAT_PRESETS["Бетон"])
        return dict(
            mat_type     = int(p["mat_type"]),
            density      = self._mat_density.get(),
            E            = self._mat_E.get(),
            nu           = self._mat_nu.get(),
            yield_stress = float(p.get("yield_stress", 14.5e6)),
            friction     = self._mat_frict.get(),
            cohesion     = self._mat_cohes.get(),
            hardening    = self._mat_hard.get(),
            bulk         = self._mat_bulk.get(),
            gamma_eos    = self._mat_gamma.get(),
            gravity      = [0.0, -9.81, 0.0],
        )

    def _build_exp_params(self) -> dict:
        """Собирает словарь параметров взрыва для MPM."""
        return dict(
            center   = [self._exp_cx.get(),
                        self._exp_cy.get(),
                        self._exp_cz.get()],
            radius   = self._exp_radius.get(),
            pressure = self._exp_pressure.get(),
            duration = self._exp_duration.get(),
            delay    = self._exp_delay.get(),
        )

    def _start_mpm(self):
        if self._particles is None:
            messagebox.showinfo("MPM", "Сначала выполните вокселизацию модели")
            return
        # Останавливаем предыдущий MPM процесс
        self._stop_mpm(silent=True)

        pb, shape = self._particles
        mat = self._build_mat_params()
        exp = self._build_exp_params()

        # mpm_q_res: MPM -> viewer (кадры позиций)
        # mpm_q_cmd: UI  -> MPM  (пауза/стоп)
        self._mpm_q_cmd = mp.Queue()
        self._mpm_q_res = mp.Queue(maxsize=3)

        self._mpm_proc = mp.Process(
            target=_mpm_worker,
            args=(pb, shape, mat, exp,
                  self._mpm_q_res, self._mpm_q_cmd),
            daemon=True)
        self._mpm_proc.start()

        # Переоткрываем viewer — передаём mpm_q_res как источник кадров
        # viewer читает кадры из mpm_queue, UI-команды из cmd_queue
        self._open_viewer(pb, shape, mpm_queue=self._mpm_q_res)
        name = self._mat_type.get()
        self._mpm_status.config(
            text=f"Симуляция запущена: {name}",
            foreground="#66bb66")

    def _stop_mpm(self, silent=False):
        if self._mpm_proc and self._mpm_proc.is_alive():
            try:
                self._mpm_q_cmd.put_nowait(("quit",))
                time.sleep(0.1)
            except: pass
            self._mpm_proc.terminate()
        self._mpm_proc  = None
        self._mpm_q_cmd = None
        self._mpm_q_res = None
        if not silent:
            self._mpm_status.config(text="Симуляция остановлена",
                                    foreground="#888888")

    # ── опрос очереди каждые 100 мс ──────────────────────────────────────────

    def _poll(self):
        if self._job_q is not None:
            while not self._job_q.empty():
                try:
                    msg = self._job_q.get_nowait()
                except Exception:
                    break

                if msg[0] == "progress":
                    _, prog, status = msg
                    self._progress_var.set(prog)
                    self._status_var.set(status)

                elif msg[0] == "result":
                    _, pbytes, shape = msg
                    n = shape[0]
                    self._particles = (pbytes, shape)
                    self._progress_var.set(1.0)
                    self._status_var.set(f"Готово: {n:,} частиц")
                    self._footer.config(
                        text=f"Частиц: {n:,}  |  "
                             f"res={self._vp_res.get()}  "
                             f"cams={self._vp_cameras.get()}")
                    self._btn_vox.state(["!disabled"])
                    self._open_viewer(pbytes, shape)
                    self._job_q = None

                elif msg[0] == "error":
                    self._status_var.set("Ошибка!")
                    self._footer.config(
                        text="Ошибка вокселизации (см. консоль)",
                        foreground="#cc4444")
                    print("[VoxError]\n", msg[1])
                    self._btn_vox.state(["!disabled"])
                    self._job_q = None

        self.after(100, self._poll)

    def on_close(self):
        if self._job_proc and self._job_proc.is_alive():
            self._job_proc.terminate()
        self._stop_mpm(silent=True)
        if self._viewer_proc and self._viewer_proc.is_alive():
            try: self._viewer_q.put_nowait(("quit",))
            except: pass
            time.sleep(0.1)
            self._viewer_proc.terminate()
        self.destroy()


# =============================================================================
if __name__ == "__main__":
    mp.freeze_support()
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
