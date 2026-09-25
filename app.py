"""Listening bench for isotope-dependent tunneling on an 8-qubit grid.

Experience plays a precomputed exact movie for H, D, and T. Lab keeps the
split-operator comparison, the rate sketch, and the QPAM player. When
``src/atlas_bank.json`` is present, tap-to-observe and the Bloch sphere use
that Aer tomography run. Aer is a simulator.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from qiskit import transpile

from src.atlas_tomography import bloch_length_shot_noise
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
        initial_sidebar_state="auto",
    )
    _inject_css()
    view = st.segmented_control(
        "View",
        options=["Experience", "Lab"],
        default="Experience",
        key="app_view",
        label_visibility="collapsed",
    )
    if view == "Lab":
        _render_lab()
    else:
        _inject_experience_chrome()
        _render_experience()


def _render_experience() -> None:
    root = Path(__file__).resolve().parent
    template = root / "src" / "experience.html"
    data = root / "src" / "experience_data.json"
    bank = root / "src" / "bank.json"
    atlas = root / "src" / "atlas_bank.json"
    intro = root / "atlas_outputs"
    intro_stamp = 0
    if intro.exists():
        for frame in intro.glob("frame_*.png"):
            intro_stamp = max(intro_stamp, frame.stat().st_mtime_ns)
    stamp = (
        template.stat().st_mtime_ns,
        data.stat().st_mtime_ns if data.exists() else 0,
        bank.stat().st_mtime_ns if bank.exists() else 0,
        atlas.stat().st_mtime_ns if atlas.exists() else 0,
        intro_stamp,
    )
    components.html(_experience_document(stamp), height=1100, scrolling=False)


@st.cache_data(show_spinner=False)
def _experience_document(stamp: tuple) -> str:
    del stamp
    root = Path(__file__).resolve().parent
    template = (root / "src" / "experience.html").read_text(encoding="utf-8")
    data_path = root / "src" / "experience_data.json"
    if not data_path.exists():
        from src.export_frames import write_experience_data

        write_experience_data(data_path)
    data = data_path.read_text(encoding="utf-8")
    bank_path = root / "src" / "bank.json"
    if not bank_path.exists():
        from src.measurement_bank import write_bank

        write_bank(bank_path)
    bank = bank_path.read_text(encoding="utf-8")
    atlas_path = root / "src" / "atlas_bank.json"
    atlas = atlas_path.read_text(encoding="utf-8") if atlas_path.exists() else "null"
    intro = _intro_frames_json(root / "atlas_outputs")
    replacements = (
        ("__EXPERIENCE_DATA__", data),
        ("__BANK_DATA__", bank),
        ("__ATLAS_BANK__", atlas),
        ("__INTRO_FRAMES__", intro),
    )
    document = template
    for token, value in replacements:
        if token not in document:
            raise RuntimeError(f"experience.html is missing {token}.")
        document = document.replace(token, value, 1)
    return document


def _intro_frames_json(folder: Path) -> str:
    """Base64 data URLs for Teleblur frames, or JSON null when none exist."""

    if not folder.exists():
        return "null"
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        return "null"
    urls = []
    for path in frames:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        urls.append(f"data:image/png;base64,{encoded}")
    return json.dumps(urls)


def _render_lab() -> None:
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
    bank_figure, bank_note = _bank_figure()
    if bank_figure is not None:
        st.plotly_chart(
            bank_figure,
            width="stretch",
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(bank_note)
    atlas_figure, atlas_note, atlas_jobs = _atlas_figure()
    if atlas_figure is not None:
        st.plotly_chart(
            atlas_figure,
            width="stretch",
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(atlas_note)
        st.code("\n".join(atlas_jobs), language="text")
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


def _bank_figure() -> tuple[go.Figure | None, str]:
    path = Path(__file__).resolve().parent / "src" / "bank.json"
    if not path.exists():
        return None, ""
    bank = json.loads(path.read_text(encoding="utf-8"))
    source = str(bank.get("source", ""))
    points = bank.get("points", [])
    budgets = [
        int(point.get("counts", {}).get("0", 0)) + int(point.get("counts", {}).get("1", 0))
        for point in points
    ]
    if not budgets or min(budgets) < 1:
        return None, ""
    used_shots = budgets[0]
    times = [float(point["t_over_period"]) for point in points]
    ideal = [float(point["ideal_acceptor"]) for point in points]
    measured = [
        float(point["counts"].get("1", 0)) / budget
        for point, budget in zip(points, budgets)
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=ideal,
            name="Exact acceptor",
            line={"color": TEAL, "width": 2.4},
            hovertemplate="t = %{x:.3f} T_tun<br>exact = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=measured,
            name=source,
            mode="lines+markers",
            line={"color": AMBER, "width": 1.6},
            marker={"size": 7, "color": AMBER},
            hovertemplate="t = %{x:.3f} T_tun<br>measured = %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Measurement bank · acceptor population · {source} · {used_shots} shots",
        height=420,
        yaxis={"title": "Acceptor population", "range": [0, 1.02]},
        xaxis={"title": "t / T_tun"},
        **_layout(top=60),
    )
    _style_axes(figure)
    note = (
        f"Hydrogen at the default well (E_b = {bank.get('barrier_ev', 0.15)} eV, "
        f"a = {bank.get('half_width_a', 0.35)} Å, λ = {bank.get('asymmetry', 0)}). "
        f"{len(points)} times across one tunneling period, {used_shots} shots, source: {source}."
    )
    return figure, note


def _atlas_is_simulator(bank: dict) -> bool:
    provider = str(bank.get("provider", "")).lower()
    execution = str(bank.get("execution", "")).lower()
    return provider == "aer" or execution == "simulator"


def _atlas_figure() -> tuple[go.Figure | None, str, list[str]]:
    path = Path(__file__).resolve().parent / "src" / "atlas_bank.json"
    if not path.exists():
        return None, "", []
    bank = json.loads(path.read_text(encoding="utf-8"))
    points = bank.get("points") or []
    if not points:
        return None, "", []
    times = [float(point["t_over_period"]) for point in points]
    ideal = [float(point["ideal_acceptor"]) for point in points]
    measured = [float(point["acceptor_population"]) for point in points]
    length = [float(point["bloch_length"]) for point in points]
    provider = str(bank.get("provider", "aer"))
    backend = str(bank.get("backend", "automatic"))
    shots = bank.get("shots")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=ideal,
            name="Exact acceptor",
            line={"color": TEAL, "width": 2.4},
            hovertemplate="t = %{x:.3f} T_tun<br>exact = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=measured,
            name="Atlas acceptor",
            mode="lines+markers",
            line={"color": AMBER, "width": 1.6},
            marker={"size": 7, "color": AMBER},
            hovertemplate="t = %{x:.3f} T_tun<br>Atlas P = %{y:.3f}<extra></extra>",
        )
    )
    simulator = _atlas_is_simulator(bank)
    if simulator:
        grid = np.linspace(0.0, 1.0, 361)
        half = 2.0 * np.array([bloch_length_shot_noise(float(t), int(shots)) for t in grid])
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([1.0 + half, (1.0 - half)[::-1]]),
                fill="toself",
                fillcolor="rgba(224, 122, 95, 0.28)",
                line={"width": 0},
                name="±2σ shot noise",
                hovertemplate="t = %{x:.3f} T_tun<br>±2σ around |r| = 1<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=length,
            name="Bloch length |r|",
            mode="lines+markers",
            line={"color": CORAL, "width": 1.6, "dash": "dot"},
            marker={"size": 6, "color": CORAL},
            hovertemplate="t = %{x:.3f} T_tun<br>|r| = %{y:.3f}<extra></extra>",
        )
    )
    kind = "simulator" if simulator else provider
    figure.update_layout(
        title=f"Atlas tomography · {provider}/{backend} · {kind} · {shots} shots",
        height=420,
        yaxis={"title": "Acceptor population and |r|", "range": [0, 1.06]},
        xaxis={"title": "t / T_tun"},
        **_layout(top=60),
    )
    _style_axes(figure)
    if simulator:
        note = (
            f"{provider}/{backend} is the Aer simulator, not a hardware QPU. "
            f"{shots} shots on each of X, Y, and Z. "
            "On a noiseless simulator, deviations from |r| = 1 are sampling noise, not decoherence. "
            "Acceptor population is (1 − z) / 2 from observables X, Y, Z. Job ids:"
        )
    else:
        note = (
            f"{provider}/{backend} · {shots} shots at each of {len(points)} times. "
            "Acceptor population is (1 − z) / 2 from observables X, Y, Z. Job ids:"
        )
    jobs = [str(job_id) for job_id in bank.get("job_ids", [])]
    return figure, note, jobs


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
    st.markdown("### Atlas tomography")
    path = Path(__file__).resolve().parent / "src" / "atlas_bank.json"
    if not path.exists():
        st.info(
            "src/atlas_bank.json is not in this checkout. "
            "Experience tap-to-observe falls back to the local simulator bank."
        )
        return
    bank = json.loads(path.read_text(encoding="utf-8"))
    provider = bank.get("provider", "aer")
    backend = bank.get("backend", "automatic")
    st.markdown(
        f"`{provider}` / `{backend}` · {bank.get('execution', 'simulator')} · "
        f"{bank.get('shots')} shots · `{bank.get('engine')}` · {bank.get('date')}"
    )
    st.caption(
        "aer is a simulator. The tomography-api-v2 schema does not list an IBM hardware provider, "
        "so this page does not call the run a QPU."
    )


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


def _inject_experience_chrome() -> None:
    st.markdown(
        """
        <style>
          [data-testid="stHeader"],
          [data-testid="stToolbar"],
          [data-testid="stDecoration"],
          [data-testid="stSidebar"],
          [data-testid="stSidebarCollapsedControl"],
          #MainMenu,
          footer,
          .stDeployButton { display: none !important; }
          .stApp { background: #0e0e10; }
          [data-testid="stMain"] { overflow: hidden; }
          [data-testid="stMainBlockContainer"] {
            height: 100dvh !important;
            max-height: 100dvh !important;
            max-width: 100% !important;
            padding: 0.4rem 0.5rem 0.2rem !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
          }
          [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {
            height: 100% !important;
            min-height: 0 !important;
            gap: 0.3rem !important;
          }
          [data-testid="stElementContainer"]:has(iframe) {
            flex: 1 1 auto !important;
            height: auto !important;
            min-height: 0 !important;
          }
          iframe[data-testid="stIFrame"] {
            height: 100% !important;
            min-height: 0 !important;
          }
          div[data-testid="stSegmentedControl"] { margin: 0 0 0.1rem 0.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
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
