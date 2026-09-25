"""Listening bench for isotope-dependent tunneling on an 8-qubit grid.

The exact propagator is ``evolve_exact``. The split-operator circuit is drawn
on the same clock. Audio is either an artistic beat mapped from ΔE and the
well populations, or QPAM measurements of ``|ψ|²`` frames. Atlas is not
connected until ``AtlasClient`` returns a real payload.
"""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from qiskit import transpile

from src.atlas_client import AtlasClient
from src.hamiltonian import ANGSTROM_M, JOULE_PER_EV
from src.moth_sonification import (
    DEFAULT_PITCH_HZ,
    DEFAULT_SAMPLE_RATE,
    measured_state,
    tunneling_beat,
)
from src.pdb_parser import fetch_active_site
from src.quantum_sim import evolve_exact, evolve_split_operator, split_operator_circuit
from src.rates import k_classical, k_tunnel, kie

DEFAULT_BARRIER_EV = 0.15
DEFAULT_HALF_WIDTH_A = 0.35
DEFAULT_ASYMMETRY = 0.0
DEFAULT_ISOTOPE = "H"
DEFAULT_TEMPERATURE_K = 300
N_PERIODS = 1.0
N_STEPS = 200
BASIS_GATES = ["rz", "sx", "x", "cx"]

INK = "#f3eee6"
MUTED = "#a39888"
AMBER = "#e4a23a"
TEAL = "#3cbaa6"
CORAL = "#e07a5f"
LINE = "#3a342c"
PANEL = "#191714"
PAPER = "#12110f"

MODES = {
    "Tunneling beat": "tunneling_beat",
    "Measured state": "measured_state",
}


def main() -> None:
    st.set_page_config(
        page_title="Listening to Quantum Tunneling",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    controls = _sidebar()
    _header()

    try:
        with st.spinner("Evolving the exact doublet and one period of the split-operator circuit…"):
            sim = _simulate(
                controls["barrier_ev"],
                controls["half_width_a"],
                controls["asymmetry"],
                controls["isotope"],
            )
    except Exception as exc:
        st.error(f"The tunneling evolution failed: {exc}")
        _atlas_panel()
        return

    _metrics(sim)
    time_index = _time_slider(sim)
    st.plotly_chart(
        _potential_figure(sim, time_index),
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.plotly_chart(
        _population_figure(sim),
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.plotly_chart(
        _rate_figure(sim, controls["temperature_k"]),
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption(
        "V(x) = E_b (1 − (x/a)²)² + λx. "
        f"Populations cover one tunneling period in {N_STEPS} steps. "
        "The circuit is the symmetric split-operator product on 8 qubits. "
        "Fidelity is |⟨ψ_circuit|ψ_exact⟩|² at the end of that period. "
        "On a taller barrier the dashed circuit curves leave the solid ones, "
        "and that fidelity falls with them. "
        "Depth is one Trotter step after transpiling to rz, sx, x, cx "
        "(optimization level 1)."
    )
    _audio_player(sim, controls)
    _context_caption()
    _atlas_panel()


def _sidebar() -> dict:
    st.sidebar.markdown("### Double well")
    barrier_ev = st.sidebar.slider(
        "Barrier E_b (eV)",
        min_value=0.1,
        max_value=0.5,
        value=DEFAULT_BARRIER_EV,
        step=0.01,
        help="Height of V(x) at x = 0.",
    )
    half_width_a = st.sidebar.slider(
        "Well half-separation a (Å)",
        min_value=0.25,
        max_value=0.50,
        value=DEFAULT_HALF_WIDTH_A,
        step=0.01,
        help="The well minima of the model potential sit at x = ±a.",
    )
    asymmetry = st.sidebar.slider(
        "Asymmetry λ (eV)",
        min_value=0.0,
        max_value=0.02,
        value=DEFAULT_ASYMMETRY,
        step=0.001,
        help=(
            "Coefficient of λx, applied as eV per angstrom. "
            "Larger values detune the two wells and suppress coherent tunneling."
        ),
    )
    isotope = st.sidebar.selectbox(
        "Isotope",
        options=["H", "D", "T"],
        index=["H", "D", "T"].index(DEFAULT_ISOTOPE),
        help="Mass presets for the proton, deuteron, and triton. V(x) is unchanged.",
    )
    temperature_k = st.sidebar.slider(
        "Temperature (K)",
        min_value=100,
        max_value=400,
        value=int(DEFAULT_TEMPERATURE_K),
        step=5,
        help="Marks a point on the classical Arrhenius curve. k_tunnel is flat in this model.",
    )
    st.sidebar.markdown("### Listening")
    mode_label = st.sidebar.selectbox(
        "Sonification mode",
        options=list(MODES),
        help="Tunneling beat pans two tones with the populations. Measured state plays QPAM reconstructions of |ψ|².",
    )
    shots = st.sidebar.select_slider(
        "Shot count",
        options=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        value=1024,
        help="Measurements per probability frame in Measured state. Fewer shots make a noisier sound.",
    )
    pitch_hz = st.sidebar.slider(
        "Base pitch (Hz)",
        min_value=80.0,
        max_value=880.0,
        value=DEFAULT_PITCH_HZ,
        step=1.0,
        help="Carrier of the tunneling beat. 220 Hz is A3.",
    )
    return {
        "barrier_ev": float(barrier_ev),
        "half_width_a": float(half_width_a),
        "asymmetry": float(asymmetry),
        "isotope": str(isotope),
        "temperature_k": float(temperature_k),
        "mode": MODES[mode_label],
        "mode_label": mode_label,
        "shots": int(shots),
        "pitch_hz": float(pitch_hz),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _load_site() -> dict:
    site = fetch_active_site()
    if site is None:
        return {"available": False, "pdb_id": "1HED", "distance_angstrom": None}
    return {
        "available": True,
        "pdb_id": site.pdb_id,
        "distance_angstrom": site.distance_angstrom,
    }


@st.cache_data(show_spinner=False)
def _simulate(barrier_ev: float, half_width_a: float, asymmetry: float, isotope: str) -> dict:
    model, evolution = evolve_exact(
        barrier_ev=barrier_ev,
        asymmetry_ev=asymmetry,
        well_half_width_angstrom=half_width_a,
        mass=isotope,
        n_periods=N_PERIODS,
        n_steps=N_STEPS,
        keep_states=True,
    )
    if evolution.states is None:
        raise RuntimeError("The exact evolution did not keep its states.")
    dt = float(evolution.times_s[1] - evolution.times_s[0])
    circuit_states = evolve_split_operator(
        evolution.initial_statevector,
        model.v_joule,
        model.mass_kg,
        model.spacing_m,
        dt,
        n_steps=N_STEPS,
        sample_at=tuple(range(N_STEPS + 1)),
    )
    donor_mask = model.x_m < 0.0
    circuit_donor = np.empty(N_STEPS + 1, dtype=np.float64)
    circuit_acceptor = np.empty(N_STEPS + 1, dtype=np.float64)
    for frame in range(N_STEPS + 1):
        probability = np.abs(circuit_states[frame]) ** 2
        circuit_donor[frame] = float(probability[donor_mask].sum())
        circuit_acceptor[frame] = float(probability[~donor_mask].sum())
    fidelity = float(np.abs(np.vdot(circuit_states[N_STEPS], evolution.states[N_STEPS])) ** 2)

    step = split_operator_circuit(
        model.v_joule, model.mass_kg, model.spacing_m, dt, n_steps=1
    )
    compiled = transpile(step, basis_gates=BASIS_GATES, optimization_level=1)
    spacing_angstrom = model.spacing_m / ANGSTROM_M
    densities = np.abs(evolution.states) ** 2
    return {
        "x_angstrom": model.x_m / ANGSTROM_M,
        "v_ev": model.v_joule / JOULE_PER_EV,
        "times_in_periods": evolution.times_in_periods,
        "donor_exact": evolution.donor_population,
        "acceptor_exact": evolution.acceptor_population,
        "donor_circuit": circuit_donor,
        "acceptor_circuit": circuit_acceptor,
        "density_per_angstrom": densities / spacing_angstrom,
        "splitting_ev": float(evolution.splitting_ev),
        "splitting_j": float(evolution.splitting_j),
        "tunneling_period_s": float(evolution.tunneling_period_s),
        "fidelity": fidelity,
        "circuit_depth": int(compiled.depth()),
        "circuit_size": int(compiled.size()),
        "kie": float(kie(barrier_ev, half_width_a)),
        "k_tunnel": float(k_tunnel(evolution.splitting_j)),
        "barrier_ev": float(barrier_ev),
        "isotope": isotope,
    }


def _header() -> None:
    st.markdown(
        """
        <p class="eyebrow">Moth Hacks 2026 · H / D / T on one double well</p>
        <h1>Listening to Quantum Tunneling: isotope effects, sonified</h1>
        <p class="lede">The tunneling doublet of a model double well, compared for
        hydrogen, deuterium, and tritium. A matrix exponential is the reference.
        A split-operator circuit runs the same grid. The sound is either a beat
        mapped from the splitting, or a QPAM measurement of the probability.</p>
        """,
        unsafe_allow_html=True,
    )


def _metrics(sim: dict) -> None:
    cards = [
        ("ΔE", _format_mev(sim["splitting_ev"]), "E1 − E0"),
        ("Tunneling period", _format_period(sim["tunneling_period_s"]), "T_tun = h / ΔE"),
        ("KIE (H/D)", f"{sim['kie']:.2f}", "ΔE(H) / ΔE(D) at λ = 0"),
        ("Circuit fidelity", f"{sim['fidelity']:.4f}", "versus exact at T_tun"),
        ("Circuit depth", str(sim["circuit_depth"]), f"{sim['circuit_size']} gates, one Trotter step"),
    ]
    cells = []
    for label, value, detail in cards:
        cells.append(
            "<div class='metric-card'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<em>{html.escape(detail)}</em>"
            "</div>"
        )
    st.markdown(f"<div class='metric-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def _time_slider(sim: dict) -> int:
    period = st.slider(
        "Snapshot time (tunneling periods)",
        min_value=0.0,
        max_value=float(N_PERIODS),
        value=0.5,
        step=0.01,
        help="Selects the |ψ|² frame drawn on V(x). Populations use the full period.",
    )
    index = int(np.argmin(np.abs(sim["times_in_periods"] - period)))
    return index


def _potential_figure(sim: dict, time_index: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sim["x_angstrom"],
            y=sim["v_ev"],
            name="V(x)",
            line={"color": MUTED, "width": 2},
            hovertemplate="x = %{x:.3f} Å<br>V = %{y:.3f} eV<extra></extra>",
        )
    )
    density = sim["density_per_angstrom"][time_index]
    clock = float(sim["times_in_periods"][time_index])
    figure.add_trace(
        go.Scatter(
            x=sim["x_angstrom"],
            y=density,
            name=f"|ψ|² at {clock:.2f} T_tun",
            line={"color": TEAL, "width": 2.4},
            fill="tozeroy",
            fillcolor="rgba(60, 186, 166, 0.18)",
            yaxis="y2",
            hovertemplate="x = %{x:.3f} Å<br>|ψ|² = %{y:.3f} Å⁻¹<extra></extra>",
        )
    )
    figure.update_layout(
        title="Potential and probability",
        height=420,
        yaxis={"title": "V(x) (eV)", "range": _potential_window(sim["x_angstrom"], sim["v_ev"])},
        yaxis2={"title": "|ψ|² (Å⁻¹)", "overlaying": "y", "side": "right"},
        xaxis={"title": "x (Å)"},
        **_layout(top=60),
    )
    _style_axes(figure)
    return figure


def _population_figure(sim: dict) -> go.Figure:
    figure = go.Figure()
    clock = sim["times_in_periods"]
    series = (
        ("Donor, exact", sim["donor_exact"], AMBER, "solid"),
        ("Donor, circuit", sim["donor_circuit"], AMBER, "dash"),
        ("Acceptor, exact", sim["acceptor_exact"], TEAL, "solid"),
        ("Acceptor, circuit", sim["acceptor_circuit"], TEAL, "dash"),
    )
    for name, values, color, dash in series:
        figure.add_trace(
            go.Scatter(
                x=clock,
                y=values,
                name=name,
                line={"color": color, "width": 2.2, "dash": dash},
                hovertemplate="t = %{x:.3f} T_tun<br>P = %{y:.3f}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"Donor and acceptor populations · {sim['isotope']}",
        height=420,
        yaxis={"title": "Population", "range": [0, 1.02]},
        xaxis={"title": "t / T_tun"},
        **_layout(top=60),
    )
    _style_axes(figure)
    return figure


def _rate_figure(sim: dict, temperature_k: float) -> go.Figure:
    temperatures = np.linspace(100.0, 400.0, 80)
    classical = np.array(
        [k_classical(float(temperature), sim["barrier_ev"]) for temperature in temperatures]
    )
    tunnel = np.full_like(classical, sim["k_tunnel"])
    marked = k_classical(temperature_k, sim["barrier_ev"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=temperatures,
            y=classical,
            name="k_classical",
            line={"color": AMBER, "width": 2.4},
            hovertemplate="T = %{x:.0f} K<br>k = %{y:.3e} s⁻¹<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=temperatures,
            y=tunnel,
            name="k_tunnel",
            line={"color": TEAL, "width": 2.4, "dash": "dash"},
            hovertemplate="T = %{x:.0f} K<br>k_tunnel = %{y:.3e} s⁻¹<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[temperature_k],
            y=[marked],
            name=f"{temperature_k:.0f} K",
            mode="markers",
            marker={"color": CORAL, "size": 11},
            hovertemplate="T = %{x:.0f} K<br>k_classical = %{y:.3e} s⁻¹<extra></extra>",
        )
    )
    figure.update_layout(
        title="Qualitative illustration",
        height=420,
        yaxis={"title": "Rate (s⁻¹)", "type": "log"},
        xaxis={"title": "Temperature (K)"},
        **_layout(top=60),
    )
    _style_axes(figure)
    return figure


def _audio_player(sim: dict, controls: dict) -> None:
    st.markdown(f"### {controls['mode_label']}")
    try:
        if controls["mode"] == "tunneling_beat":
            clip = tunneling_beat(
                sim["donor_exact"],
                sim["acceptor_exact"],
                sim["splitting_ev"],
                base_pitch_hz=controls["pitch_hz"],
                sample_rate=DEFAULT_SAMPLE_RATE,
            )
        else:
            with st.spinner("Measuring probability frames with QPAM…"):
                wav = _measured_wav(sim["density_per_angstrom"], controls["shots"])
            clip = None
    except Exception as exc:
        st.error(f"Sonification failed: {exc}")
        return

    if clip is not None:
        st.audio(clip.wav_bytes, format="audio/wav")
        st.caption(clip.note)
        st.plotly_chart(
            _waveform_figure(clip.samples, clip.sample_rate),
            width="stretch",
            config={"displaylogo": False, "displayModeBar": False},
        )
    else:
        st.audio(wav, format="audio/wav")
        st.caption(
            f"QPAM encode then decode of probability frames, {controls['shots']} shots each. "
            "Fewer shots leave a noisier sound."
        )


@st.cache_data(show_spinner=False)
def _measured_wav(density_per_angstrom: np.ndarray, shots: int) -> bytes:
    clip = measured_state(density_per_angstrom, shots=shots, sample_rate=DEFAULT_SAMPLE_RATE)
    return clip.wav_bytes


def _waveform_figure(samples: np.ndarray, sample_rate: int) -> go.Figure:
    window_n = int(0.05 * sample_rate)
    times_ms = np.arange(window_n) / sample_rate * 1000.0
    figure = go.Figure()
    audio = np.asarray(samples)
    if audio.ndim == 1:
        figure.add_trace(
            go.Scatter(x=times_ms, y=audio[:window_n], name="Mono", line={"color": AMBER, "width": 1.5})
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=times_ms, y=audio[:window_n, 0], name="Left (donor)", line={"color": AMBER, "width": 1.5}
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times_ms,
                y=audio[:window_n, 1],
                name="Right (acceptor)",
                line={"color": TEAL, "width": 1.5},
            )
        )
    figure.update_layout(
        title="Opening 50 ms",
        height=240,
        xaxis={"title": "t (ms)"},
        yaxis={"title": "Amplitude", "range": [-1.05, 1.05]},
        **_layout(top=48),
    )
    _style_axes(figure)
    return figure


def _context_caption() -> None:
    site = _load_site()
    if site["available"] and site["distance_angstrom"] is not None:
        st.caption(
            f"Illustrative active-site context from PDB {site['pdb_id']}: "
            f"Zn–HOH distance {site['distance_angstrom']:.2f} Å. "
            "That distance is not an input to the Hamiltonian."
        )
    else:
        st.caption("PDB unavailable. The Hamiltonian does not use a crystal distance.")


def _atlas_panel() -> None:
    st.markdown("### Atlas")
    payload = _atlas_payload()
    if payload is None:
        st.info("Atlas: not connected")
        return
    st.json(payload)


def _atlas_payload() -> dict | None:
    client = AtlasClient()
    try:
        data = client.get_job("connection")
    except NotImplementedError:
        return None
    if not data:
        return None
    return data


def _potential_window(x_angstrom: np.ndarray, v_ev: np.ndarray) -> list[float]:
    x_angstrom = np.asarray(x_angstrom)
    v_ev = np.asarray(v_ev)
    crest = v_ev[np.abs(x_angstrom) <= 0.15]
    top = float(crest.max()) if crest.size else float(np.median(v_ev))
    floor = float(v_ev.min())
    return [floor - 0.05, max(top * 1.35, floor + 0.2)]


def _format_mev(splitting_ev: float) -> str:
    mev = splitting_ev * 1.0e3
    if mev >= 1.0:
        return f"{mev:.2f} meV"
    if mev >= 0.01:
        return f"{mev:.3f} meV"
    return f"{mev:.2e} meV"


def _format_period(seconds: float) -> str:
    scales = (
        (1e-15, 1e18, "as"),
        (1e-12, 1e15, "fs"),
        (1e-9, 1e12, "ps"),
        (1e-6, 1e9, "ns"),
        (1e-3, 1e6, "µs"),
        (1.0, 1e3, "ms"),
    )
    for limit, factor, unit in scales:
        if seconds < limit:
            return f"{seconds * factor:.2f} {unit}"
    return f"{seconds:.3f} s"


def _layout(top: int) -> dict:
    return {
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PANEL,
        "font": {"color": INK, "family": "Outfit, Helvetica, sans-serif", "size": 13},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        "margin": {"l": 60, "r": 60, "t": top, "b": 48},
        "hovermode": "x unified",
    }


def _style_axes(figure: go.Figure) -> None:
    figure.update_xaxes(
        gridcolor="rgba(243, 238, 230, 0.08)",
        zerolinecolor="rgba(243, 238, 230, 0.18)",
        linecolor=LINE,
    )
    figure.update_yaxes(
        gridcolor="rgba(243, 238, 230, 0.08)",
        zerolinecolor="rgba(243, 238, 230, 0.18)",
        linecolor=LINE,
    )


def _inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,640&family=Outfit:wght@380;520;600&display=swap" rel="stylesheet">
        <style>
          html, body, [class*="css"] { font-family: "Outfit", Helvetica, sans-serif; }
          .block-container { padding-top: 1.3rem; max-width: 1180px; }
          h1 {
            font-family: "Fraunces", Georgia, serif !important;
            font-weight: 560 !important;
            letter-spacing: -0.03em;
            font-size: 2.15rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.35rem;
          }
          .eyebrow {
            color: #e4a23a;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.74rem;
            margin: 0 0 0.35rem 0;
          }
          .lede { color: #c9c0b2; font-size: 1.05rem; max-width: 46rem; margin-bottom: 0.4rem; }
          .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.8rem 0 1rem;
          }
          .metric-card {
            background: #1c1a16;
            border: 1px solid #3a342c;
            border-radius: 14px;
            padding: 0.8rem 0.95rem 0.85rem;
          }
          .metric-card span {
            display: block;
            color: #a39888;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          .metric-card strong {
            display: block;
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.35rem;
            font-weight: 560;
            margin-top: 0.18rem;
            color: #f3eee6;
          }
          .metric-card em {
            display: block;
            color: #a39888;
            font-style: normal;
            font-size: 0.78rem;
            margin-top: 0.15rem;
          }
          div[data-testid="stAlert"] {
            border-radius: 14px;
          }
          @media (max-width: 900px) {
            .metric-grid { grid-template-columns: 1fr 1fr; }
            h1 { font-size: 1.65rem !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
