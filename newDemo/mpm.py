"""
mpm.py — MLS-MPM симулятор
Все @ti.kernel БЕЗ аргументов. Параметры передаются через ti.field(shape=()).
"""
import math, time
import numpy as np
import taichi as ti
import taichi.math as tm

MAT_CONCRETE = 0
MAT_ELASTIC  = 1
MAT_FLUID    = 2

# ─── поля (создаются в init_sim) ──────────────────────────────────────────────
_x = _v = _C = _F = _Jp = _damage = _eps_p = None
_gv = _gm = None
# скалярные параметры
_mu = _lam = _Rb = _Rbt = _eb1 = _eb0 = _eb2 = _Kb = _E = None
_fric = _coh = _hard = _Kbulk = _gam = _mtype = _sim_time = None
_grav = _exp_c = _exp_r = _exp_p = _exp_dur = _exp_del = None
# константы (Python float/int, не поля)
_NP = _NG = _inv_dx = _dx = _p_mass = _p_vol = _DT = None
# те же константы как ti.field для доступа из @ti.kernel
_f_pmass = _f_pvol = _f_dt = _f_inv_dx = _f_NG = None


def init_sim(n_particles, grid_res, dx_val, mat_params, exp_params):
    global _x,_v,_C,_F,_Jp,_damage,_eps_p
    global _gv,_gm
    global _mu,_lam,_Rb,_Rbt,_eb1,_eb0,_eb2,_Kb,_E
    global _fric,_coh,_hard,_Kbulk,_gam,_mtype,_sim_time
    global _grav,_exp_c,_exp_r,_exp_p,_exp_dur,_exp_del
    global _NP,_NG,_inv_dx,_dx,_p_mass,_p_vol,_DT

    NP = n_particles
    NG = grid_res

    # --- материал ---
    rho0  = float(mat_params.get("density",      2300.0))
    E     = float(mat_params.get("E",            3.0e10))
    nu    = float(mat_params.get("nu",           0.20))
    frict = float(mat_params.get("friction",     0.60))
    cohes = float(mat_params.get("cohesion",     2.0e6))
    hard  = float(mat_params.get("hardening",    0.0))
    Kbulk = float(mat_params.get("bulk",         1.0e6))
    gam   = float(mat_params.get("gamma_eos",    7.0))
    yld   = float(mat_params.get("yield_stress", 14.5e6))
    mt    = int  (mat_params.get("mat_type",     MAT_CONCRETE))
    grav  = mat_params.get("gravity", [0.0, -9.81, 0.0])

    mu_v  = E/(2*(1+nu))         if E>0 else 0.0
    lam_v = E*nu/((1+nu)*(1-2*nu)) if E>0 else 0.0
    Eb1_v = -(yld/max(E,1))*0.6
    denom = E*(-0.002) - 0.6*(-yld)
    Kb_v  = (0.4*(-yld)*E)/denom if abs(denom)>0 else 0.0

    p_vol_v  = (dx_val*0.5)**3
    p_mass_v = rho0 * p_vol_v
    cwave    = math.sqrt((lam_v+2*mu_v)/max(rho0,1.0))
    DT_v     = max(min(0.5*dx_val/max(cwave,1.0), 2e-4), 1e-5)

    # --- поля частиц ---
    _x      = ti.Vector.field(3, ti.f32, NP)
    _v      = ti.Vector.field(3, ti.f32, NP)
    _C      = ti.Matrix.field(3,3, ti.f32, NP)
    _F      = ti.Matrix.field(3,3, ti.f32, NP)
    _Jp     = ti.field(ti.f32, NP)
    _damage = ti.field(ti.f32, NP)
    _eps_p  = ti.field(ti.f32, NP)

    # --- поля грида ---
    _gv = ti.Vector.field(3, ti.f32, (NG,NG,NG))
    _gm = ti.field(ti.f32,          (NG,NG,NG))

    # --- скалярные параметры как поля shape=() ---
    def sf(v): f=ti.field(ti.f32,()); f[None]=float(v); return f
    def si(v): f=ti.field(ti.i32,()); f[None]=int(v);   return f

    _mu       = sf(mu_v);   _lam  = sf(lam_v)
    _Rb       = sf(yld);    _Rbt  = sf(max(cohes*0.1,1e4))
    _eb1      = sf(Eb1_v);  _eb0  = sf(-0.002); _eb2=sf(-0.0035)
    _Kb       = sf(Kb_v);   _E    = sf(E)
    _fric     = sf(frict);  _coh  = sf(cohes); _hard=sf(hard)
    _Kbulk    = sf(Kbulk);  _gam  = sf(gam)
    _mtype    = si(mt)
    _sim_time = sf(0.0)

    _grav  = ti.Vector.field(3, ti.f32, ())
    _grav[None] = ti.Vector([grav[0], grav[1], grav[2]])

    ec = exp_params.get("center", [0.5,0.5,0.5])
    _exp_c = ti.Vector.field(3, ti.f32, ())
    _exp_c[None] = ti.Vector([float(ec[0]),float(ec[1]),float(ec[2])])
    _exp_r   = sf(exp_params.get("radius",   0.2))
    _exp_p   = sf(exp_params.get("pressure", 2e9))
    _exp_dur = sf(exp_params.get("duration", 0.015))
    _exp_del = sf(exp_params.get("delay",    0.05))

    # константы Python (но p_mass/p_vol/DT/inv_dx — ещё и как поля для kernel)
    _NP=NP; _NG=NG; _inv_dx=1.0/dx_val; _dx=dx_val
    _p_mass=p_mass_v; _p_vol=p_vol_v; _DT=DT_v

    # поля для доступа из @ti.kernel (Taichi не видит Python-float глобалы)
    global _f_pmass, _f_pvol, _f_dt, _f_inv_dx, _f_NG
    _f_pmass  = sf(p_mass_v)
    _f_pvol   = sf(p_vol_v)
    _f_dt     = sf(DT_v)
    _f_inv_dx = sf(1.0/dx_val)
    _f_NG     = ti.field(ti.i32, ()); _f_NG[None] = NG

    # Обновляем глобальные переменные модуля
    import sys
    _mod = sys.modules[__name__]
    for _name, _val in [
        ('_x',_x),('_v',_v),('_C',_C),('_F',_F),('_Jp',_Jp),
        ('_damage',_damage),('_eps_p',_eps_p),
        ('_gv',_gv),('_gm',_gm),
        ('_mu',_mu),('_lam',_lam),('_Rb',_Rb),('_Rbt',_Rbt),
        ('_eb1',_eb1),('_eb0',_eb0),('_eb2',_eb2),('_Kb',_Kb),('_E',_E),
        ('_fric',_fric),('_coh',_coh),('_hard',_hard),
        ('_Kbulk',_Kbulk),('_gam',_gam),('_mtype',_mtype),
        ('_sim_time',_sim_time),('_grav',_grav),
        ('_exp_c',_exp_c),('_exp_r',_exp_r),('_exp_p',_exp_p),
        ('_exp_dur',_exp_dur),('_exp_del',_exp_del),
        ('_NP',NP),('_NG',NG),('_inv_dx',1.0/dx_val),('_dx',dx_val),
        ('_p_mass',p_mass_v),('_p_vol',p_vol_v),('_DT',DT_v),
        ('_f_pmass',_f_pmass),('_f_pvol',_f_pvol),('_f_dt',_f_dt),
        ('_f_inv_dx',_f_inv_dx),('_f_NG',_f_NG),
    ]:
        setattr(_mod, _name, _val)


# ─── @ti.func (без аргументов с аннотациями) ──────────────────────────────────

@ti.func
def _sigma_c(eps_vol):
    s    = 0.0
    Rb_  = _Rb[None];  Rbt_ = _Rbt[None]
    eb1_ = _eb1[None]; eb0_ = _eb0[None]; eb2_ = _eb2[None]
    Kb_  = _Kb[None];  E_   = _E[None]
    if   eps_vol >= 0.0:  s = ti.min(E_*eps_vol, Rbt_)
    elif eps_vol >= eb1_: s = E_*eps_vol
    elif eps_vol >= eb0_: s = -Rb_ + Kb_*(eps_vol-eb0_)
    elif eps_vol >= eb2_: s = -Rb_
    return s


@ti.func
def _dp(sigma_dev, p_hyd, ep_old):
    fric_=_fric[None]; coh_=_coh[None]; hard_=_hard[None]
    mu_=_mu[None];     lam_=_lam[None]
    dn    = tm.length(sigma_dev)
    c_eff = coh_*(1.0+hard_*ep_old)
    f_y   = dn + fric_*p_hyd - c_eff
    dp    = 0.0
    sd    = sigma_dev
    pn    = p_hyd
    if f_y>0.0 and dn>1e-20:
        K_ = lam_+2.0/3.0*mu_
        D  = mu_ + hard_*coh_*mu_/(dn+1e-20) + fric_*fric_*K_
        dl = f_y/(D+1e-20)
        sd = sigma_dev*(1.0-dl*mu_/(dn+1e-20))
        pn = p_hyd+dl*fric_*K_
        dp = dl
    return sd, pn, dp


# ─── @ti.kernel БЕЗ АРГУМЕНТОВ ────────────────────────────────────────────────

@ti.kernel
def k_init_all():
    for i in _x:
        _v[i]=tm.vec3(0); _F[i]=tm.mat3([[1,0,0],[0,1,0],[0,0,1]])
        _C[i]=tm.mat3(0); _Jp[i]=1.0; _damage[i]=0.0; _eps_p[i]=0.0


@ti.kernel
def k_clear():
    for i,j,k in _gm:
        _gv[i,j,k]=tm.vec3(0); _gm[i,j,k]=0.0


@ti.kernel
def k_p2g():
    dt     = _DT
    inv_dx = _inv_dx
    pmass  = _p_mass
    pvol   = _p_vol
    NG     = _NG
    for p in _x:
        if _damage[p]>=1.0: continue
        xp   = _x[p]
        base = ti.cast(xp*inv_dx-0.5, ti.i32)
        fx   = xp*inv_dx - ti.cast(base, ti.f32)
        w0=0.5*(1.5-fx)**2; w1=0.75-(fx-1.0)**2; w2=0.5*(fx-0.5)**2
        wx=tm.vec3(w0.x,w1.x,w2.x)
        wy=tm.vec3(w0.y,w1.y,w2.y)
        wz=tm.vec3(w0.z,w1.z,w2.z)

        mt_=_mtype[None]; stress=tm.mat3(0.0)
        if mt_==MAT_FLUID:
            J_=_Jp[p]; K_=_Kbulk[None]; g_=_gam[None]
            pr=K_/g_*(J_**(-g_)-1.0)
            stress=-pr*tm.mat3([[1,0,0],[0,1,0],[0,0,1]])
        else:
            Fi=_F[p]; J_=ti.max(tm.determinant(Fi),1e-4)
            FT=tm.inverse(Fi).transpose()
            mu_=_mu[None]; lam_=_lam[None]
            stress=mu_*(Fi-FT)+lam_*ti.log(J_)*FT
            if mt_==MAT_CONCRETE:
                tr_=stress[0,0]+stress[1,1]+stress[2,2]
                ph=-tr_/3.0
                dev=stress+ph*tm.mat3([[1,0,0],[0,1,0],[0,0,1]])
                dn,pn,dp=_dp(dev,ph,_eps_p[p])
                _eps_p[p]+=dp
                stress=dn-pn*tm.mat3([[1,0,0],[0,1,0],[0,0,1]])
                ev=(_F[p][0,0]+_F[p][1,1]+_F[p][2,2])/3.0-1.0
                ss=_sigma_c(ev)
                if ev<_eb2[None] or (ev>0.0 and ti.abs(ss)<1e-3*_Rbt[None]):
                    _damage[p]=ti.min(_damage[p]+0.015,1.0)
                stress=stress*(1.0-_damage[p])

        aff=pmass*_C[p]-dt*pvol*4.0*inv_dx*inv_dx*stress
        for i,j,k in ti.static(ti.ndrange(3,3,3)):
            off=tm.vec3(float(i),float(j),float(k))
            w=wx[i]*wy[j]*wz[k]
            dp2=(off-fx)/_inv_dx
            gi=base+ti.Vector([i,j,k])
            if 0<=gi.x<NG and 0<=gi.y<NG and 0<=gi.z<NG:
                ti.atomic_add(_gm[gi], w*pmass)
                ti.atomic_add(_gv[gi], w*(pmass*_v[p]+aff@dp2))


@ti.kernel
def k_grid():
    dt=_DT; NG=_NG
    for i,j,k in _gm:
        if _gm[i,j,k]>0:
            _gv[i,j,k]/=_gm[i,j,k]
            _gv[i,j,k]+=dt*_grav[None]
            if i<3     and _gv[i,j,k].x<0: _gv[i,j,k].x=0.0
            if i>NG-3  and _gv[i,j,k].x>0: _gv[i,j,k].x=0.0
            if j<3     and _gv[i,j,k].y<0: _gv[i,j,k].y=0.0
            if j>NG-3  and _gv[i,j,k].y>0: _gv[i,j,k].y=0.0
            if k<3     and _gv[i,j,k].z<0: _gv[i,j,k].z=0.0
            if k>NG-3  and _gv[i,j,k].z>0: _gv[i,j,k].z=0.0


@ti.kernel
def k_g2p():
    dt=_DT; inv_dx=_inv_dx; NG=_NG
    for p in _x:
        if _damage[p]>=1.0: continue
        xp=_x[p]
        base=ti.cast(xp*inv_dx-0.5, ti.i32)
        fx=xp*inv_dx-ti.cast(base,ti.f32)
        w0=0.5*(1.5-fx)**2; w1=0.75-(fx-1.0)**2; w2=0.5*(fx-0.5)**2
        wx=tm.vec3(w0.x,w1.x,w2.x)
        wy=tm.vec3(w0.y,w1.y,w2.y)
        wz=tm.vec3(w0.z,w1.z,w2.z)
        nv=tm.vec3(0.0); nC=tm.mat3(0.0)
        for i,j,k in ti.static(ti.ndrange(3,3,3)):
            gi=base+ti.Vector([i,j,k])
            if 0<=gi.x<NG and 0<=gi.y<NG and 0<=gi.z<NG:
                off=tm.vec3(float(i),float(j),float(k))
                w=wx[i]*wy[j]*wz[k]
                dp=(off-fx)/inv_dx
                gvi=_gv[gi]
                nv+=w*gvi
                nC+=4.0*inv_dx*w*gvi.outer_product(dp)
        # ── Взрыв: прямой импульс скорости ──────────────────────────────
        t_   = _sim_time[None]
        del_ = _exp_del[None]
        dur_ = _exp_dur[None]
        if t_ >= del_ and t_ <= del_ + dur_:
            de = tm.length(_x[p] - _exp_c[None])
            re = _exp_r[None]
            if de < re and de > 1e-6:
                # Импульс = давление * площадь * dt / масса
                # Масштаб: p_exp задаётся в Па·м³/кг = м/с² * с → м/с за substep
                sc_e  = (1.0 - de / re) ** 2
                dir_e = (_x[p] - _exp_c[None]) / de
                # pressure (Па) * p_vol (м³) / p_mass (кг) * dt → Δv (м/с)
                dv    = dir_e * (_exp_p[None] * sc_e * _f_pvol[None] * dt / _f_pmass[None])
                nv   += dv

        _v[p]=nv; _C[p]=nC; _x[p]+=dt*nv
        mt_=_mtype[None]
        if mt_==MAT_FLUID:
            _Jp[p]*=(1.0+dt*(nC[0,0]+nC[1,1]+nC[2,2]))
            _Jp[p]=ti.max(_Jp[p],0.01)
        else:
            _F[p]=(tm.mat3([[1,0,0],[0,1,0],[0,0,1]])+dt*nC)@_F[p]


# ─── step / get_state ─────────────────────────────────────────────────────────

def step(n_sub=8):
    for _ in range(n_sub):
        k_clear(); k_p2g(); k_grid(); k_g2p()
        _sim_time[None] += _DT


def get_state():
    return (_x.to_numpy().astype(np.float32),
            _damage.to_numpy().astype(np.float32))


# ─── Точка входа процесса ─────────────────────────────────────────────────────

def run_mpm(particles_bytes, shape, mat_params, exp_params,
            result_queue, cmd_queue):

    ti.init(arch=ti.gpu, device_memory_GB=2,
            offline_cache=True, debug=False)

    pts = np.frombuffer(particles_bytes, dtype=np.float32).reshape(shape)
    mn,mx = pts.min(0), pts.max(0)
    span  = max((mx-mn).max(), 1e-6)
    ctr   = (mn+mx)*0.5
    sc    = 0.70/span
    pts_n = (pts-ctr)*sc + 0.5

    grid_res=64; dx=1.0/grid_res

    ec   = np.array(exp_params.get("center",[0,0,0]),np.float32)
    exp_n= dict(exp_params)
    # Центр взрыва: абсолютные координаты → нормализованные [0,1]
    exp_n["center"] = ((ec-ctr)*sc+0.5).tolist()
    # Радиус задаётся в нормализованном пространстве [0,1] — НЕ масштабируем
    exp_n["radius"] = float(exp_params.get("radius", 0.3))

    init_sim(len(pts_n), grid_res, dx, mat_params, exp_n)
    _x.from_numpy(pts_n.astype(np.float32))
    k_init_all()

    paused=False; frame_dt=1/30; n_sub=20; t0=time.perf_counter()

    while True:
        while not cmd_queue.empty():
            try:
                cmd=cmd_queue.get_nowait()
                if   cmd[0]=="quit":     return
                elif cmd[0]=="pause":    paused=True
                elif cmd[0]=="resume":   paused=False
                elif cmd[0]=="substeps": n_sub=max(1,int(cmd[1]))
            except: pass

        if not paused:
            step(n_sub)
            pos,dmg=get_state()
            try:
                result_queue.put_nowait(
                    ("frame",pos.tobytes(),pos.shape,dmg.tobytes()))
                result_queue.put_nowait(("time",float(_sim_time[None])))
            except: pass

        sl=frame_dt-(time.perf_counter()-t0)
        if sl>0: time.sleep(sl)
        t0=time.perf_counter()
