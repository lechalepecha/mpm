"""
voxelizer.py  —  GPU Voxel Carving на Taichi
============================================================
Алгоритм (классический voxel carving):
  1. Грид инициализируется как SOLID (объект везде)
  2. N камер размещаются на сфере вокруг объекта
  3. Для каждой камеры — параллельный проход по вокселям:
       Луч: камера → центр вокселя
       • Нет пересечения с мешем                → воксель снаружи → EMPTY
       • Пересечение ДАЛЬШЕ вокселя             → воксель снаружи → EMPTY
       • Пересечение БЛИЖЕ вокселя              → воксель за поверхностью, не вырезаем
  4. Воксель остаётся SOLID только если НИ ОДНА камера его не вырезала

Параметр --load (0.0–1.0) задаёт допустимую нагрузку на GPU:
  1.0 = без ограничений (максимальная скорость)
  0.5 = GPU работает 50% времени, 50% пауза
  0.3 = GPU работает 30% времени  (тихий/холодный режим)

Использование:
  python voxelizer.py --mesh ../meshes/roof.glb --res 128 --cameras 64
  python voxelizer.py --mesh ../meshes/roof.glb --res 256 --load 0.5
"""

import argparse
import math
import os
import time

import numpy as np
import open3d as o3d
import taichi as ti

ti.init(arch=ti.gpu, device_memory_GB=4, offline_cache=True, debug=False)

# ── опциональный мониторинг GPU (только NVIDIA, требует pynvml) ──
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:
    _NVML = False


def gpu_temp() -> str:
    if not _NVML:
        return ""
    try:
        t = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        return f"  |  GPU {t}°C"
    except Exception:
        return ""


def throttle_sleep(work_sec: float, load: float) -> None:
    """
    Пауза после GPU-burst так, чтобы средняя загрузка = load.

      duty  = work / (work + sleep)
      sleep = work * (1/duty - 1)

    load=1.0 → пауза 0 (без ограничений)
    load=0.5 → GPU 50% времени
    load=0.3 → GPU 30% времени
    """
    if load >= 1.0 or work_sec <= 0:
        return
    load = max(0.05, min(load, 0.99))
    time.sleep(work_sec * (1.0 / load - 1.0))


# ============================================================
#  Состояния вокселя
# ============================================================
SOLID = 1
EMPTY = 0

# ============================================================
#  Möller–Trumbore ray–triangle intersection
# ============================================================

@ti.func
def ray_tri_intersect(
    ro: ti.math.vec3, rd: ti.math.vec3,
    v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3,
) -> ti.f32:
    # Двустороннее пересечение (both front and back faces)
    EPS   = 1e-7
    e1    = v1 - v0
    e2    = v2 - v0
    h     = rd.cross(e2)
    a     = e1.dot(h)
    t_out = ti.f32(-1.0)
    if ti.abs(a) > EPS:
        f = 1.0 / a
        s = ro - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(e1)
            v = f * rd.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * e2.dot(q)
                if t > EPS:
                    t_out = t
    return t_out


@ti.func
def count_intersections(
    ro: ti.math.vec3, rd: ti.math.vec3,
    tri_v0: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    tri_v1: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    tri_v2: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    t_max: ti.f32,
) -> ti.i32:
    # Считает число пересечений луча с мешем на [0, t_max]
    # Нечётное = точка внутри (parity rule)
    cnt = ti.i32(0)
    for i in range(tri_v0.shape[0]):
        t = ray_tri_intersect(ro, rd, tri_v0[i], tri_v1[i], tri_v2[i])
        if 0.0 < t < t_max:
            cnt += 1
    return cnt


# ============================================================
#  Kernels
# ============================================================

@ti.kernel
def kernel_init_solid(grid: ti.template()):
    for i, j, k in grid:
        grid[i, j, k] = SOLID


@ti.kernel
def kernel_carve_one_camera(
    grid:       ti.template(),
    res:        ti.i32,
    voxel_size: ti.f32,
    origin:     ti.math.vec3,
    cam:        ti.math.vec3,
    tri_v0:     ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    tri_v1:     ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    tri_v2:     ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
):
    for vi in ti.ndrange(res * res * res):
        ix = vi // (res * res)
        iy = (vi // res) % res
        iz = vi % res

        if grid[ix, iy, iz] == EMPTY:
            continue

        vox = ti.math.vec3(
            origin[0] + (ix + 0.5) * voxel_size,
            origin[1] + (iy + 0.5) * voxel_size,
            origin[2] + (iz + 0.5) * voxel_size,
        )

        to_vox = vox - cam
        dist   = to_vox.norm()
        if dist < 1e-6:
            continue
        rd = to_vox / dist   # направление: камера → воксель

        # Считаем число пересечений между камерой и вокселем
        # Чётное (0, 2, 4...) = воксель снаружи меша → вырезаем
        # Нечётное (1, 3, 5..) = воксель внутри меша → оставляем
        n_hits = count_intersections(cam, rd, tri_v0, tri_v1, tri_v2,
                                     dist - voxel_size * 0.5)
        if n_hits % 2 == 0:
            grid[ix, iy, iz] = EMPTY


@ti.kernel
def kernel_count(grid: ti.template()) -> ti.i32:
    cnt = ti.i32(0)
    for i, j, k in grid:
        if grid[i, j, k] == SOLID:
            cnt += 1
    return cnt


@ti.kernel
def kernel_collect(
    grid:       ti.template(),
    res:        ti.i32,
    voxel_size: ti.f32,
    origin:     ti.math.vec3,
    out:        ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    counter:    ti.template(),
):
    for ix, iy, iz in ti.ndrange(res, res, res):
        if grid[ix, iy, iz] == SOLID:
            idx = ti.atomic_add(counter[None], 1)
            out[idx] = ti.math.vec3(
                origin[0] + (ix + 0.5) * voxel_size,
                origin[1] + (iy + 0.5) * voxel_size,
                origin[2] + (iz + 0.5) * voxel_size,
            )


# ============================================================
#  Fibonacci sphere
# ============================================================

def fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for i in range(n):
        y     = 1.0 - (i / max(n - 1, 1)) * 2.0
        r     = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        pts.append([r * math.cos(theta) * radius,
                    y * radius,
                    r * math.sin(theta) * radius])
    return np.array(pts, dtype=np.float32)


# ============================================================
#  Загрузка меша
# ============================================================

def load_mesh(path: str):
    import shutil, tempfile
    print(f"[voxelizer] Zagрuzka {path} ...")

    # open3d не поддерживает не-ASCII пути на Windows —
    # копируем файл во временную папку с безопасным именем
    try:
        path.encode("ascii")
        safe_path = path
    except UnicodeEncodeError:
        ext       = os.path.splitext(path)[1]
        tmp_dir   = tempfile.mkdtemp()
        safe_path = os.path.join(tmp_dir, "mesh" + ext)
        shutil.copy2(path, safe_path)
        print(f"[voxelizer] non-ASCII path, tmp copy: {safe_path}")

    mesh = o3d.io.read_triangle_mesh(safe_path)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not load mesh: {path}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    mn, mx = verts.min(0), verts.max(0)
    center = (mn + mx) / 2.0
    scale  = np.linalg.norm(mx - mn) / 2.0
    verts  = (verts - center) / scale

    tris = np.asarray(mesh.triangles, dtype=np.int32)
    print(f"[voxelizer] {len(verts):,} вершин  {len(tris):,} треугольников")
    return verts.astype(np.float32), tris


# ============================================================
#  Главная функция
# ============================================================

def voxelize(
    mesh_path:  str,
    cache_path: str,
    resolution: int   = 128,
    n_cameras:  int   = 64,
    cam_radius: float = 2.5,
    grid_size:  float = 2.2,
    load:       float = 1.0,   # 0.0–1.0, доля времени GPU под нагрузкой
):
    verts, tris = load_mesh(mesh_path)

    voxel_size = grid_size / resolution
    origin_v   = ti.math.vec3(*[-grid_size / 2.0] * 3)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cameras = fibonacci_sphere(n_cameras, cam_radius)

    load_pct = int(load * 100)
    print(f"[voxelizer] Грид {resolution}³ = {resolution**3:,} вокселей")
    print(f"[voxelizer] {n_cameras} камер  радиус={cam_radius}")
    print(f"[voxelizer] Нагрузка GPU: {load_pct}%"
          + ("  (без ограничений)" if load >= 1.0 else
             f"  (пауза {(1/load - 1)*100:.0f}% от времени kernel)"))

    grid = ti.field(dtype=ti.i32, shape=(resolution, resolution, resolution))

    t0 = time.perf_counter()

    kernel_init_solid(grid)
    ti.sync()

    # ── Carving с throttle ──────────────────────────────────
    for ci, cam_np in enumerate(cameras):
        cam_v = ti.math.vec3(*cam_np.tolist())

        t_work = time.perf_counter()
        kernel_carve_one_camera(
            grid, resolution, voxel_size, origin_v,
            cam_v, v0, v1, v2,
        )
        ti.sync()
        work_sec = time.perf_counter() - t_work

        # Пауза пропорционально времени работы kernel
        throttle_sleep(work_sec, load)

        if (ci + 1) % 8 == 0 or ci == n_cameras - 1:
            elapsed = time.perf_counter() - t0
            eta_sec = elapsed / (ci + 1) * (n_cameras - ci - 1)
            print(f"  камера {ci+1:>4}/{n_cameras}"
                  f"  прошло {elapsed:.1f}s  осталось ~{eta_sec:.0f}s"
                  f"  kernel {work_sec*1000:.0f}ms"
                  + gpu_temp())

    # ── Сбор ───────────────────────────────────────────────
    n_solid = kernel_count(grid)
    ti.sync()
    total = time.perf_counter() - t0
    print(f"[voxelizer] SOLID вокселей: {n_solid:,}  (полное время: {total:.2f}s)")

    if n_solid == 0:
        raise RuntimeError(
            "Ни одного вокселя не осталось — "
            "уменьши --cameras или увеличь --cam-radius"
        )

    counter = ti.field(dtype=ti.i32, shape=())
    counter[None] = 0
    out_buf = ti.ndarray(dtype=ti.math.vec3, shape=n_solid)

    kernel_collect(grid, resolution, voxel_size, origin_v, out_buf, counter)
    ti.sync()

    particles = out_buf.to_numpy()[:int(counter[None])].astype(np.float32)
    print(f"[voxelizer] Частиц: {len(particles):,}")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, particles)
    print(f"[voxelizer] Сохранено → {cache_path}")
    return particles


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU Voxel Carving на Taichi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh",       default="../meshes/roof.glb")
    parser.add_argument("--cache",      default="../cache/roof_particles.npy")
    parser.add_argument("--res",        type=int,   default=128,
                        help="Разрешение грида N (N³ вокселей)")
    parser.add_argument("--cameras",    type=int,   default=64,
                        help="Количество виртуальных камер")
    parser.add_argument("--cam-radius", type=float, default=2.5,
                        help="Радиус сферы камер (> размера объекта)")
    parser.add_argument("--grid-size",  type=float, default=2.2,
                        help="Размер вокселного грида в мировых единицах")
    parser.add_argument("--load",       type=float, default=1.0,
                        help="Нагрузка GPU 0.0–1.0 "
                             "(0.5 = 50%% времени работает, 50%% пауза; "
                             "1.0 = без ограничений)")
    args = parser.parse_args()

    if not 0.0 < args.load <= 1.0:
        parser.error("--load должен быть в диапазоне (0.0, 1.0]")

    voxelize(
        mesh_path  = args.mesh,
        cache_path = args.cache,
        resolution = args.res,
        n_cameras  = args.cameras,
        cam_radius = args.cam_radius,
        grid_size  = args.grid_size,
        load       = args.load,
    )