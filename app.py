"""Streamlit dashboard: quantum proton tunneling in carbonic anhydrase II, sonified with Moth.

Run with::

    streamlit run app.py

Layout
------
* **Header** - active-site provenance from RCSB PDB ``1HED`` (Zn-HOH distance) and the live
  physics readouts for the current parameter set.
* **Sidebar** - barrier height, asymmetry drive, simulated duration and the base audio pitch,
  plus an advanced block (geometry, wavepacket width, QPAM shots, H/D isotope switch).
* **Plots** - the double-well potential with the wavepacket density on top, and the
  donor/acceptor population dynamics over time.
* **Moth audio** - QPAM sonification of the statevector trajectory with an in-browser player,
  and the Moth Atlas platform job document.

The physics lives in :mod:`src.hamiltonian` / :mod:`src.quantum_sim`; every call into Moth
Quantum's ``quantumaudio`` goes through :mod:`src.moth_sonification`. This file is pure UI.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import moth_sonification
from src.hamiltonian import M_H, joules_to_ev
from src.pdb_parser import ActiveSite, load_active_site
from src.quantum_sim import TunnelingResult, simulate_tunneling

PAGE_TITLE = "Quantum Tunneling & Sonification Engine"
DONOR_COLOR = "#4cc9f0"
ACCEPTOR_COLOR = "#f72585"
POTENTIAL_COLOR = "#adb5bd"
ENERGY_COLOR = "#ffd166"

st.set_page_config(page_title=PAGE_TITLE, page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------------------
# Cached data layer
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching 1HED from the RCSB PDB…", ttl=3600)
def get_active_site() -> ActiveSite:
    """Load (and cache for an hour) the CA II active-site geometry from RCSB PDB ``1HED``."""
    return load_active_site()


@st.cache_data(show_spinner="Propagating the 8-qubit statevector…", ttl=900)
def run_simulation(
    barrier_height_ev: float,
    asymmetry_ev: float,
    duration_fs: float,
    num_steps: int,
    well_separation_ang: float,
    width_ang: float,
    mass_factor: float,
) -> TunnelingResult:
    """Run the tunneling simulation for one parameter set, memoised by Streamlit.

    ``mass_factor`` multiplies the proton mass: 1 for hydrogen, 2 for deuterium. Swapping it
    is the in-app kinetic isotope effect experiment — the heavier deuteron has a shorter de
    Broglie wavelength and its tunneling probability collapses.
    """
    return simulate_tunneling(
        barrier_height_ev=barrier_height_ev,
        asymmetry_ev=asymmetry_ev,
        duration_fs=duration_fs,
        num_steps=num_steps,
        well_separation_ang=well_separation_ang,
        width_ang=width_ang,
        mass=mass_factor * M_H,
    )


# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------
def render_sidebar() -> Dict[str, Any]:
    """Draw the control panel and return the current parameter set."""
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.caption(
        "The proton coordinate is discretised onto 256 grid points = one 8-qubit register."
    )

    barrier = st.sidebar.slider(
        "Barrier Height $E_b$ (eV)",
        min_value=0.10,
        max_value=1.00,
        value=0.15,
        step=0.01,
        help="Height of the central barrier separating the donor and acceptor wells. "
        "Tunneling probability falls off exponentially with barrier height and width.",
    )
    asymmetry = st.sidebar.slider(
        "Asymmetry Drive $\\lambda$ (eV)",
        min_value=0.00,
        max_value=0.20,
        value=0.10,
        step=0.01,
        help="Energy bias at the well minima: the thermodynamic driving force of the "
        "transfer. 0 eV gives a symmetric well and coherent back-and-forth oscillation.",
    )
    duration = st.sidebar.slider(
        "Simulation Duration (fs)",
        min_value=5,
        max_value=50,
        value=50,
        step=1,
        help="Femtoseconds of quantum dynamics to propagate. Coherent proton transfer in an "
        "enzyme active site plays out on exactly this timescale.",
    )
    pitch = st.sidebar.slider(
        "Base Audio Pitch (Hz)",
        min_value=55,
        max_value=880,
        value=220,
        step=5,
        help="Fundamental of the QPAM wavetable oscillator. The pitch glides upward as "
        "probability crosses the barrier.",
    )

    with st.sidebar.expander("🔬 Advanced physics & audio"):
        isotope = st.radio(
            "Tunneling particle",
            options=("Proton (¹H)", "Deuteron (²H)"),
            help="Doubling the mass is the classic kinetic-isotope-effect test: if transfer "
            "collapses when H becomes D, the mechanism is tunneling.",
        )
        separation = st.slider(
            "Well half-separation $a$ (Å)",
            min_value=0.35,
            max_value=1.00,
            value=0.60,
            step=0.05,
            help="Minima sit at x = ±a, so the proton hops 2a. Set by the donor-acceptor "
            "hydrogen-bond geometry of the active site.",
        )
        width = st.slider(
            "Initial wavepacket width $\\sigma$ (Å)",
            min_value=0.08,
            max_value=0.30,
            value=0.15,
            step=0.01,
            help="Width of the Gaussian prepared in the donor well, comparable to the "
            "zero-point width of that well.",
        )
        steps = st.select_slider(
            "Propagation steps",
            options=(60, 120, 240, 480),
            value=120,
            help="Number of applications of U(dt). The propagator is exact in dt, so this "
            "only sets the time resolution of the plots.",
        )
        shots = st.select_slider(
            "QPAM shots per frame",
            options=(500, 2000, 8000, 20000),
            value=8000,
            help="Measurement shots used to decode each audio frame. Fewer shots = more "
            "quantum shot noise in the sound.",
        )
        frames = st.select_slider(
            "Wavetable morph frames",
            options=(8, 16, 24, 32),
            value=16,
            help="Statevector snapshots QPAM-encoded and crossfaded across the render.",
        )
        audio_seconds = st.slider(
            "Audio length (s)",
            min_value=1.0,
            max_value=6.0,
            value=2.5,
            step=0.5,
        )

    return {
        "barrier": float(barrier),
        "asymmetry": float(asymmetry),
        "duration": float(duration),
        "pitch": float(pitch),
        "mass_factor": 2.0 if isotope.startswith("Deuteron") else 1.0,
        "isotope": isotope,
        "separation": float(separation),
        "width": float(width),
        "steps": int(steps),
        "shots": int(shots),
        "frames": int(frames),
        "audio_seconds": float(audio_seconds),
    }


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
def render_header(site: ActiveSite, result: TunnelingResult) -> None:
    """Title block plus active-site and physics metrics."""
    st.title("⚡ Quantum Tunneling & Sonification Engine (Moth Hacks 2026)")
    st.markdown(
        "Hydrogen **tunnels** through the proton-transfer barrier of Human Carbonic Anhydrase "
        "II — the step that limits the fastest enzyme in your bloodstream. This app solves the "
        "proton's Schrödinger equation on an 8-qubit register and turns the resulting "
        "statevector into sound with Moth Quantum's `quantumaudio` **QPAM** encoding."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PDB ID", site.pdb_id, help=site.note)
    col2.metric(
        "Zn–HOH Distance",
        f"{site.distance_ang:.2f} Å",
        help="Zn(II) 262 to catalytic water 342 (chain A): the coordination vector that "
        "anchors the proton donor in the active site.",
    )
    col3.metric(
        "Quantum Register",
        f"{result.num_qubits} qubits",
        f"{result.grid.size}-point grid",
        delta_color="off",
        help="One computational basis state |k⟩ per spatial grid point x[k].",
    )
    splitting = result.tunneling_splitting_ev
    col4.metric(
        "Tunneling Splitting",
        f"{splitting * 1e3:.1f} meV" if splitting is not None else "n/a",
        help="E₁ − E₀ of the lowest doublet. The coherent transfer period is h/ΔE, so a "
        "larger splitting means faster tunneling.",
    )
    col5.metric(
        "Peak Transfer",
        f"{result.max_transfer * 100:.1f}%",
        f"{result.acceptor_population[-1] * 100:.1f}% at t_end",
        delta_color="off",
        help="Largest probability found on the acceptor side of the barrier during the run.",
    )

    if site.is_offline_fallback:
        st.warning(
            f"RCSB PDB could not be reached — running on the deposited 1HED reference "
            f"coordinates. ({site.note})",
            icon="📡",
        )


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def build_figure(result: TunnelingResult, snapshot_index: int, barrier_ev: float) -> go.Figure:
    """Two stacked Plotly panels: the potential + density, and the population dynamics."""
    x_ang = result.grid.x_angstrom
    potential_ev = np.asarray(joules_to_ev(result.potential_j))
    density_now = result.density[snapshot_index]
    density_initial = result.density[0]
    time_now = result.times_fs[snapshot_index]

    figure = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.16,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Double-well potential V(x) and proton density |ψ(x,t)|²",
            "Population dynamics: donor well decay vs. acceptor well tunneling",
        ),
    )

    figure.add_trace(
        go.Scatter(
            x=x_ang,
            y=potential_ev,
            name="V(x)",
            line=dict(color=POTENTIAL_COLOR, width=2.5),
            hovertemplate="x = %{x:.2f} Å<br>V = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=x_ang,
            y=[result.energy_ev] * x_ang.size,
            name="⟨H⟩",
            line=dict(color=ENERGY_COLOR, width=1.4, dash="dot"),
            hovertemplate=f"⟨H⟩ = {result.energy_ev:.3f} eV<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=x_ang,
            y=density_initial,
            name="|ψ(x,0)|² (donor)",
            line=dict(color=DONOR_COLOR, width=1.5, dash="dash"),
            hovertemplate="x = %{x:.2f} Å<br>P = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=x_ang,
            y=density_now,
            name=f"|ψ(x,{time_now:.1f} fs)|²",
            line=dict(color=ACCEPTOR_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(247, 37, 133, 0.28)",
            hovertemplate="x = %{x:.2f} Å<br>P = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_vline(
        x=0.0,
        line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=result.times_fs,
            y=result.donor_population,
            name="Donor P(x < 0)",
            line=dict(color=DONOR_COLOR, width=2.5),
            hovertemplate="t = %{x:.1f} fs<br>P_donor = %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=result.times_fs,
            y=result.acceptor_population,
            name="Acceptor P(x > 0)",
            line=dict(color=ACCEPTOR_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(247, 37, 133, 0.18)",
            hovertemplate="t = %{x:.1f} fs<br>P_acceptor = %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_vline(
        x=float(time_now),
        line=dict(color=ENERGY_COLOR, width=1.5, dash="dot"),
        row=2,
        col=1,
    )

    v_floor = float(potential_ev.min())
    figure.update_yaxes(
        title_text="Energy (eV)",
        range=[v_floor - 0.05, barrier_ev + 0.35],
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="|ψ|²",
        range=[0.0, float(max(result.density.max(), 1e-6)) * 1.15],
        showgrid=False,
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Proton coordinate x (Å)", row=1, col=1)
    figure.update_yaxes(title_text="Probability", range=[0.0, 1.05], row=2, col=1)
    figure.update_xaxes(title_text="Time (fs)", row=2, col=1)
    figure.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def waveform_figure(sonification: moth_sonification.SonificationResult) -> go.Figure:
    """Plot the rendered QPAM waveform (decimated for the browser)."""
    samples = sonification.samples
    stride = max(int(samples.size // 4000), 1)
    decimated = samples[::stride]
    times = np.arange(decimated.size) * stride / sonification.sample_rate

    figure = go.Figure(
        go.Scatter(
            x=times,
            y=decimated,
            mode="lines",
            line=dict(color="#90e0ef", width=1),
            name="QPAM waveform",
            hovertemplate="t = %{x:.3f} s<br>amp = %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=50, r=20, t=30, b=40),
        title="Decoded QPAM waveform (16-bit PCM, mono)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis_range=[-1.05, 1.05],
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return figure


# --------------------------------------------------------------------------------------
# Moth audio + Atlas section
# --------------------------------------------------------------------------------------
def render_audio_section(result: TunnelingResult, params: Dict[str, Any]) -> None:
    """QPAM synthesis button, audio player and the Moth Atlas job document."""
    st.subheader("🎛️ Moth Quantum Audio (QPAM) & Atlas Platform")
    st.markdown(
        "Each statevector snapshot is encoded with `quantumaudio.encode(..., scheme=\"qpam\")`, "
        "measured back through `quantumaudio.decode(...)`, and used as a single-cycle "
        "wavetable. Crossfading those wavetables stretches a ~50 fs tunneling event across "
        "seconds, so the timbre morph **is** the proton crossing the barrier."
    )

    left, right = st.columns([1, 2])
    with left:
        synthesize = st.button(
            "🎵 Synthesize Quantum Audio (QPAM)",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            f"{params['frames']} frames · {params['shots']} shots/frame · "
            f"{params['pitch']:.0f} Hz base pitch"
        )

    if synthesize:
        try:
            with st.spinner("Encoding statevectors with Moth Quantum QPAM…"):
                st.session_state["sonification"] = moth_sonification.sonify_tunneling_trajectory(
                    result.statevectors,
                    base_pitch_hz=params["pitch"],
                    duration_s=params["audio_seconds"],
                    shots=params["shots"],
                    num_frames=params["frames"],
                    acceptor_population=result.acceptor_population,
                )
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            st.session_state.pop("sonification", None)
            st.error(f"QPAM synthesis failed: {exc}")

    sonification: moth_sonification.SonificationResult | None = st.session_state.get(
        "sonification"
    )

    if sonification is None:
        with right:
            st.info(
                "No audio rendered yet. Hit **Synthesize Quantum Audio (QPAM)** to hear the "
                "current statevector trajectory.",
                icon="🔇",
            )
        return

    with right:
        sonification.wav_bytes.seek(0)
        st.audio(sonification.wav_bytes, format="audio/wav")
        metric_cols = st.columns(4)
        metric_cols[0].metric("QPAM Fidelity", f"{sonification.encode_decode_fidelity:.4f}")
        metric_cols[1].metric("Register", f"{sonification.num_qubits} qubits")
        metric_cols[2].metric("Circuit Depth", str(sonification.circuit_depth))
        metric_cols[3].metric("Audio", f"{sonification.duration_s:.1f} s")
        st.download_button(
            "⬇️ Download WAV",
            data=sonification.wav_value(),
            file_name="ca2_proton_tunneling_qpam.wav",
            mime="audio/wav",
        )

    if not sonification.used_quantumaudio:
        st.warning(
            "`quantumaudio` was unavailable, so the built-in reference QPAM map produced this "
            "audio. Install it with `pip install quantumaudio` for the real Moth encoder.",
            icon="🧩",
        )

    st.plotly_chart(waveform_figure(sonification), use_container_width=True)

    atlas_payload = moth_sonification.build_atlas_job_payload(
        sonification,
        statevector_fidelity=result.norm_fidelity,
        simulation_metadata={
            "pdb_id": "1HED",
            "system": "Human Carbonic Anhydrase II active site",
            "tunneling_particle": params["isotope"],
            "barrier_height_ev": params["barrier"],
            "asymmetry_drive_ev": params["asymmetry"],
            "well_half_separation_ang": params["separation"],
            "duration_fs": params["duration"],
            "propagation_steps": params["steps"],
            "grid_points": result.grid.size,
            "hamiltonian": "H = T + V (2nd-order finite difference, SI units)",
            "propagator": "U(dt) = expm(-i H dt / hbar) as a Qiskit UnitaryGate",
            "tunneling_splitting_mev": (
                round(result.tunneling_splitting_ev * 1e3, 4)
                if result.tunneling_splitting_ev is not None
                else None
            ),
            "peak_acceptor_population": round(result.max_transfer, 6),
        },
    )
    atlas_status = moth_sonification.submit_atlas_job(atlas_payload)

    with st.expander("🌐 Moth Atlas Platform API Output", expanded=False):
        badge = {
            "atlas-api": "✅ Dispatched to the Atlas API",
            "local-simulation": "🖥️ Executed locally (no Atlas credentials)",
            "atlas-unreachable": "⚠️ Atlas unreachable — local results retained",
        }.get(atlas_status["mode"], atlas_status["mode"])
        st.markdown(f"**{badge}** — {atlas_status['detail']}")
        st.json({**atlas_payload, "atlas_response": atlas_status["response"]})


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    """Assemble the dashboard."""
    params = render_sidebar()
    site = get_active_site()

    try:
        result = run_simulation(
            barrier_height_ev=params["barrier"],
            asymmetry_ev=params["asymmetry"],
            duration_fs=params["duration"],
            num_steps=params["steps"],
            well_separation_ang=params["separation"],
            width_ang=params["width"],
            mass_factor=params["mass_factor"],
        )
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        st.error(f"Simulation failed: {exc}")
        st.stop()
        return

    render_header(site, result)
    st.divider()

    final_time = float(result.times_fs[-1])
    snapshot_time = st.slider(
        "Snapshot time (fs)",
        min_value=0.0,
        max_value=final_time,
        value=final_time,
        step=float(result.times_fs[1] - result.times_fs[0]),
        help="Scrub through the propagation to watch density leak across the barrier.",
    )
    snapshot_index = int(np.argmin(np.abs(result.times_fs - snapshot_time)))
    st.caption(
        f"Showing |ψ(x,t)|² at **t = {result.times_fs[snapshot_index]:.2f} fs** · "
        f"donor {result.donor_population[snapshot_index] * 100:.1f}% / "
        f"acceptor {result.acceptor_population[snapshot_index] * 100:.1f}% · "
        f"norm fidelity {result.norm_fidelity:.10f}"
    )
    st.plotly_chart(
        build_figure(result, snapshot_index, params["barrier"]), use_container_width=True
    )

    st.divider()
    render_audio_section(result, params)

    st.divider()
    with st.expander("📖 How this works (physics + Moth Quantum integration)"):
        st.markdown(
            f"""
**The system.** Carbonic anhydrase II hydrates CO₂ at ~10⁶ turnovers per second, and the
rate-limiting step is handing a proton from the zinc-bound water to an acceptor. The
Zn(II)–HOH342 vector measured from PDB `1HED` ({site.distance_ang:.2f} Å) locates that donor
water; the transfer coordinate modelled here spans the donor→acceptor hydrogen bond, with
minima at x = ±{params['separation']:.2f} Å.

**The quantum problem.** The proton moves in the asymmetric double well
V(x) = E_b(1 − (x/a)²)² + λ(x/a) (in SI joules internally). The kinetic operator is a
2nd-order finite-difference Laplacian, so H = T + V is a sparse 256×256 Hermitian matrix on a
256-point grid spanning ±1.5 Å — one basis state per state of an **8-qubit register**
(2⁸ = 256).

**The dynamics.** A Gaussian wavepacket starts in the donor well (x < 0). We exponentiate the
Hamiltonian into U(dt) = exp(−iH·dt/ħ), wrap it in a Qiskit `UnitaryGate`, and apply it
step by step to a `Statevector`. The propagator is exact in dt and unitary to machine
precision (norm fidelity here: {result.norm_fidelity:.12f}). Probability appearing at x > 0 has
gone *through* a barrier it cannot classically climb — the ⟨H⟩ = {result.energy_ev:.3f} eV line
in the plot sits below the barrier top.

**Why it's tunneling.** Switch the particle to a deuteron in the sidebar: doubling the mass
shortens the de Broglie wavelength and the transfer collapses. That mass sensitivity is the
kinetic isotope effect, the experimental signature of hydrogen tunneling in enzymes.

**The sonification.** `src/moth_sonification.py` projects each complex statevector onto a real
256-sample frame (keeping the quantum phase), encodes it with
`quantumaudio.encode(frame, scheme="qpam")` — QPAM puts sample *k* in the probability of basis
state |k⟩ — then measures it back with `quantumaudio.decode(circuit, shots=...)`. The decoded
frames become single-cycle wavetables that crossfade over a couple of seconds, with pitch
gliding up as probability crosses the barrier. Shot noise from the measurement is left in the
signal on purpose: it is the sound of sampling a quantum state.
            """
        )

    st.caption(
        "Moth Hacks 2026 · quantumaudio "
        f"{moth_sonification.QUANTUMAUDIO_VERSION} · Qiskit statevector simulation · "
        "structure data from RCSB PDB 1HED"
    )


if __name__ == "__main__":
    main()
