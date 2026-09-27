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

INK = "#eceae4"
MUTED = "#8d8b84"
ACCENT = "#2b35c8"
PLOT = "#eceae4"
PLOT_ALT = "#9aa2ff"
BAND = "rgba(43, 53, 200, 0.42)"
LINE = "#2a2a32"
PANEL = "#121216"
PAPER = "#0b0b0d"
FONT = '"JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace'

MODES = {
    "Tunneling beat": "tunneling_beat",
    "Measured state": "measured_state",
}


def main() -> None:
    st.set_page_config(
        page_title="Listening to a proton tunnel",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
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
    components.html(_experience_document(_experience_stamp(root)), height=1100, scrolling=False)


def _experience_stamp(root: Path) -> tuple:
    watched = [
        root / "src" / "experience.html",
        root / "src" / "experience_data.json",
        root / "src" / "bank.json",
        root / "src" / "atlas_bank.json",
        root / "atlas_inputs" / "donor.png",
        root / "atlas_inputs" / "acceptor.png",
        root / "atlas_outputs" / "teleblur.png",
        root / "notebooks" / "media" / "through_the_wall.wav",
        root / "atlas_tessa" / "carpet_aer.png",
        root / "atlas_tessa" / "carpet.png",
    ]
    stamp = [path.stat().st_mtime_ns if path.exists() else 0 for path in watched]
    intro = root / "atlas_outputs"
    if intro.exists():
        for frame in intro.glob("frame_*.png"):
            stamp.append(frame.stat().st_mtime_ns)
    return tuple(stamp)


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
    from src.atlas_tomography import hardware_pending, load_tomography_banks

    saved_banks = []
    for saved in load_tomography_banks():
        saved_banks.append({key: value for key, value in saved.items() if key != "_path"})
    atlas_banks = json.dumps(
        {"banks": saved_banks, "hardwarePending": hardware_pending(saved_banks)}
    )
    intro = _intro_frames_json(root / "atlas_outputs")
    piece = root / "notebooks" / "media" / "through_the_wall.wav"
    wall_audio = _data_url(piece, "audio/wav") if piece.exists() else ""
    cover = root / "atlas_tessa" / "carpet_aer.png"
    if not cover.exists():
        cover = root / "atlas_tessa" / "carpet.png"
    wall_cover = _data_url(cover, "image/png") if cover.exists() else ""
    replacements = (
        ("__EXPERIENCE_DATA__", data),
        ("__BANK_DATA__", bank),
        ("__ATLAS_BANK__", atlas),
        ("__ATLAS_BANKS__", atlas_banks),
        ("__INTRO_FRAMES__", intro),
        ("__WALL_AUDIO__", wall_audio),
        ("__WALL_COVER__", wall_cover),
    )
    document = template
    for token, value in replacements:
        if token not in document:
            raise RuntimeError(f"experience.html is missing {token}.")
        document = document.replace(token, value, 1)
    return document


def _data_url(path: Path, mime: str = "image/png") -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _intro_frames_json(folder: Path) -> str:
    """Three-panel intro when the contrast Teleblur run exists, else a frame list."""

    root = folder.parent
    donor = root / "atlas_inputs" / "donor.png"
    acceptor = root / "atlas_inputs" / "acceptor.png"
    teleblur = folder / "teleblur.png"
    if donor.exists() and acceptor.exists() and teleblur.exists():
        return json.dumps(
            {
                "donor": _data_url(donor),
                "teleblur": _data_url(teleblur),
                "acceptor": _data_url(acceptor),
            }
        )
    if not folder.exists():
        return "null"
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        return "null"
    return json.dumps([_data_url(path) for path in frames])


def _chart(figure: go.Figure) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def _section(number: str, title: str) -> None:
    st.markdown(
        f"<h2 class='story'><span>{html.escape(number)}</span> {html.escape(title)}</h2>",
        unsafe_allow_html=True,
    )


def _pipeline() -> None:
    st.markdown(
        """
        <div class="pipeline">
          <div class="pipe">
            <em>measure</em>
            <span>exact ψ(x,t)</span><i>→</i>
            <span>1-qubit circuit</span><i>→</i>
            <span class="engine">Atlas Tomography</span><i>→</i>
            <span>Bloch vectors</span><i>→</i>
            <span>tap-to-observe + Bloch sphere</span>
          </div>
          <div class="pipe">
            <em>see</em>
            <span>donor/acceptor renders</span><i>→</i>
            <span class="engine">Teleblur</span><i>→</i>
            <span>intro</span>
          </div>
          <div class="pipe">
            <em></em>
            <span>quantum carpet</span><i>→</i>
            <span class="engine">Tessa (Aer, emulated IBM Fez)</span><i>→</i>
            <span>cover art</span>
          </div>
          <div class="pipe">
            <em>hear</em>
            <span>tunneling clips</span><i>→</i>
            <span class="engine">Retrocausal Echo</span>
          </div>
          <div class="pipe">
            <em></em>
            <span>tunneling rhythm</span><i>→</i>
            <span class="engine">QRC MIDI</span><i>→</i>
            <span class="engine">Blur Jazz</span><i>→</i>
            <span>soft synth</span><i>→</i>
            <span class="engine">Retrocausal Echo</span><i>→</i>
            <span>the piece</span>
          </div>
          <div class="pipe">
            <em>develop</em>
            <span>|ψ|² at T/4</span><i>→</i>
            <span>one grain per measurement</span><i>→</i>
            <span>contact sheet and film</span>
          </div>
        </div>
        <p class="pipe-caption">Each engine does one job, and each job follows the physics.</p>
        """,
        unsafe_allow_html=True,
    )


def _render_lab() -> None:
    st.markdown("<p class='piece-title'>Listening to a proton tunnel</p>", unsafe_allow_html=True)
    controls = _sidebar()
    root = Path(__file__).resolve().parent
    components.html(_experience_document(_experience_stamp(root)), height=400, scrolling=False)

    try:
        sim = _simulate(
            controls["barrier_ev"],
            controls["half_width_a"],
            controls["asymmetry"],
            controls["isotope"],
        )
    except Exception as exc:
        st.error(f"The tunneling evolution failed: {exc}")
        return

    _metrics(sim)
    st.markdown(
        """
## What's new here

Many data sonifications map numbers to notes one at a time. Here the sound and the pictures come from the physics itself. The rhythm you hear is the tunneling period, so swapping hydrogen for deuterium or tritium slows it down by the real ratios, about 5.3 and 19.3 times. That slowing is the kinetic isotope effect, and in this notebook you can hear it.

The proton is also treated as what it really is at this scale: a qubit. Atlas measures its state at twenty-four moments along the crossing, and those measurements decide where the proton lands when you tap to observe it. This follows the same tomography-then-expectation-values pattern Moth uses in Quantum Graph. And because Tessa treats every colour as a point on the qubit sphere, the proton's measured states can be shown as their own true colours.
        """
    )
    _section("01", "The wall")
    time_index = _time_slider(sim)
    _chart(_potential_figure(sim, time_index))

    _section("02", "The isotopes")
    _chart(_population_figure(sim))
    bank_figure, bank_note = _bank_figure()
    if bank_figure is not None:
        _chart(bank_figure)
        st.caption(bank_note)

    _section("03", "Atlas measures it")
    _pipeline()
    atlas_figure, atlas_note, atlas_jobs = _atlas_figure()
    if atlas_figure is not None:
        _chart(atlas_figure)
        st.caption(atlas_note)
        _tomography_rows()
        with st.expander("Job ids", expanded=False):
            st.code("\n".join(atlas_jobs), language="text")
    st.caption(
        "Teleblur strength is the teleportation-effect parameter, from 0 to 1: "
        "a selector-qubit rotation interpolates the two images at the amplitude level. "
        "Output alpha is alpha1·(1−strength)+alpha2·strength. "
        "Strength is not a linear RGB mix. The optional mask slot was not sent."
    )

    _section("04", "Listen")
    _audio_player(sim, controls)

    _section("05", "Under the hood")
    _chart(_rate_figure(sim, controls["temperature_k"]))
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
    _context_caption()
    _atlas_panel()

    _section("06", "Through the Wall")
    _ep_players(root)
    st.markdown(
        """
## Through the Wall
### a short piece in three movements and a coda

I. Hydrogen · II. Deuterium · III. Tritium · Coda

Each movement is the same crossing, played by a heavier particle, so each is slower than the last. The rhythm follows the exact tunneling curve, which the Atlas tomography measurements reproduce. In the coda, Moth's quantum reservoir engine (QRC MIDI) continues that rhythm as a melody, and Blur Jazz gently scatters it. The sense of space throughout comes from Retrocausal Echo: a measurement that runs time forwards and then backwards through a chain of qubits decides when each echo returns.

Everything here ran on simulators, and the echo's physics belongs to its qubit chain, not to the proton. What you're hearing is the proton's journey, given a room to sound in.
        """
    )
    st.markdown(
        "Through the Wall, cover and pipeline by Lily S. Engines by Moth; Retrocausal Echo by Spencer Topel."
    )

    _section("07", "Exposures")
    _exposures(root)


def _tomography_rows() -> None:
    from src.atlas_tomography import HARDWARE_PENDING, hardware_pending, length_label, load_tomography_banks

    banks = load_tomography_banks()
    lines = []
    for bank in banks:
        label = f"{bank.get('provider')}/{bank.get('backend')} · {length_label(bank)}"
        lines.append(f"<p class='bank-row'>{html.escape(label)}</p>")
    if hardware_pending(banks):
        lines.append(f"<p class='bank-row pending'>{html.escape(HARDWARE_PENDING)}</p>")
    if lines:
        st.markdown("".join(lines), unsafe_allow_html=True)


def _exposures(root: Path) -> None:
    from src.exposures import WARM_LINE, exposure_caption

    sheet = root / "exposures" / "contact_sheet.png"
    if sheet.exists():
        st.image(str(sheet), caption=exposure_caption())
    video = root / "exposures" / "developing_H.mp4"
    if video.exists():
        st.video(str(video))
    elif not sheet.exists():
        st.info("The contact sheet is written by python src/exposures.py.")
    st.markdown(WARM_LINE)


def _ep_players(root: Path) -> None:
    cover = root / "atlas_tessa" / "carpet_aer.png"
    if not cover.exists():
        cover = root / "atlas_tessa" / "carpet.png"
    if cover.exists():
        st.image(
            str(cover),
            width=420,
            caption="Cover: the proton's crossing, position across and time down, coloured by phase",
        )
    from src.atlas_music import EP_DIR, ep_manifest

    folder = EP_DIR if (EP_DIR / "liner_notes.md").exists() else root / "notebooks" / "media"
    for track in ep_manifest():
        path = folder / track["file"]
        if track["file"] == "through_the_wall.wav" and not path.exists():
            path = root / "notebooks" / "media" / "through_the_wall.wav"
        st.markdown(f"**{track['title']}**")
        if path.exists():
            st.audio(path.read_bytes(), format="audio/wav")
        st.markdown(track["note"])
        st.markdown(
            f"<p class='ep-jobs'>{html.escape(track['jobs'])}</p>",
            unsafe_allow_html=True,
        )


def _sidebar() -> dict:
    barrier_ev = st.sidebar.slider(
        "Barrier (eV)",
        min_value=0.1,
        max_value=0.5,
        value=DEFAULT_BARRIER_EV,
        step=0.01,
        help="Height of V(x) at x = 0.",
    )
    half_width_a = st.sidebar.slider(
        "Well separation a (Å)",
        min_value=0.25,
        max_value=0.50,
        value=DEFAULT_HALF_WIDTH_A,
        step=0.01,
        help="The well minima of the model potential sit at x = ±a.",
    )
    isotope = st.sidebar.selectbox(
        "Isotope",
        options=["H", "D", "T"],
        index=["H", "D", "T"].index(DEFAULT_ISOTOPE),
        help="Mass presets for the proton, deuteron, and triton. V(x) is unchanged.",
    )
    pitch_hz = st.sidebar.slider(
        "Base pitch (Hz)",
        min_value=80.0,
        max_value=880.0,
        value=DEFAULT_PITCH_HZ,
        step=1.0,
        help="Carrier of the tunneling beat. 220 Hz is A3.",
    )
    with st.sidebar.expander("Advanced", expanded=False):
        asymmetry = st.slider(
            "Asymmetry λ (eV/Å)",
            min_value=0.0,
            max_value=0.02,
            value=DEFAULT_ASYMMETRY,
            step=0.001,
            help=(
                "Coefficient of λx. "
                "Larger values detune the two wells and suppress coherent tunneling."
            ),
        )
        temperature_k = st.slider(
            "Temperature (K)",
            min_value=100,
            max_value=400,
            value=int(DEFAULT_TEMPERATURE_K),
            step=5,
            help="Marks a point on the classical Arrhenius curve. k_tunnel is flat in this model.",
        )
        mode_label = st.selectbox(
            "Sonification mode",
            options=list(MODES),
            help="Tunneling beat pans two tones with the populations. Measured state plays QPAM reconstructions of |ψ|².",
        )
        shots = st.select_slider(
            "Shot count",
            options=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
            value=1024,
            help="Measurements per probability frame in Measured state. Fewer shots make a noisier sound.",
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
            line={"color": PLOT, "width": 2},
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
            line={"color": PLOT_ALT, "width": 2.4},
            fill="tozeroy",
            fillcolor="rgba(43, 53, 200, 0.28)",
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
        ("Donor, exact", sim["donor_exact"], PLOT, "solid"),
        ("Donor, circuit", sim["donor_circuit"], PLOT, "dash"),
        ("Acceptor, exact", sim["acceptor_exact"], PLOT_ALT, "solid"),
        ("Acceptor, circuit", sim["acceptor_circuit"], PLOT_ALT, "dash"),
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
            line={"color": PLOT, "width": 2.4},
            hovertemplate="t = %{x:.3f} T_tun<br>exact = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=measured,
            name=source,
            mode="lines+markers",
            line={"color": PLOT_ALT, "width": 1.6},
            marker={"size": 7, "color": PLOT_ALT},
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
            line={"color": PLOT, "width": 2.4},
            hovertemplate="t = %{x:.3f} T_tun<br>exact = %{y:.3f}<extra></extra>",
        )
    )
    from src.atlas_tomography import length_label, load_tomography_banks

    banks = load_tomography_banks() or [bank]
    palette = [PLOT_ALT, "#e4a23a", "#3cbaa6", "#e07a5f"]
    for index, saved in enumerate(banks):
        saved_points = saved.get("points") or []
        if not saved_points:
            continue
        figure.add_trace(
            go.Scatter(
                x=[float(point["t_over_period"]) for point in saved_points],
                y=[float(point["acceptor_population"]) for point in saved_points],
                name=f"{saved.get('provider')}/{saved.get('backend')} · {length_label(saved)}",
                mode="lines+markers",
                line={"color": palette[index % len(palette)], "width": 1.6},
                marker={"size": 7, "color": palette[index % len(palette)]},
                hovertemplate="t = %{x:.3f} T_tun<br>measured = %{y:.3f}<extra></extra>",
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
                fillcolor=BAND,
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
            line={"color": PLOT, "width": 1.6, "dash": "dot"},
            marker={"size": 6, "color": PLOT},
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
            "Acceptor population is (1 − z) / 2 from observables X, Y, Z."
        )
    else:
        note = (
            f"{provider}/{backend} · {shots} shots at each of {len(points)} times. "
            "Acceptor population is (1 − z) / 2 from observables X, Y, Z."
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
            line={"color": PLOT, "width": 2.4},
            hovertemplate="T = %{x:.0f} K<br>k = %{y:.3e} s⁻¹<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=temperatures,
            y=tunnel,
            name="k_tunnel",
            line={"color": PLOT_ALT, "width": 2.4, "dash": "dash"},
            hovertemplate="T = %{x:.0f} K<br>k_tunnel = %{y:.3e} s⁻¹<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[temperature_k],
            y=[marked],
            name=f"{temperature_k:.0f} K",
            mode="markers",
            marker={"color": PLOT_ALT, "size": 11},
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
    st.markdown(f"<p class='listen-mode'>{html.escape(controls['mode_label'])}</p>", unsafe_allow_html=True)
    try:
        if controls["mode"] == "tunneling_beat":
            wav, samples, sample_rate, note = _beat_payload(
                sim["donor_exact"],
                sim["acceptor_exact"],
                float(sim["splitting_ev"]),
                float(controls["pitch_hz"]),
            )
        else:
            wav = _measured_wav(sim["density_per_angstrom"], controls["shots"])
            samples = None
            sample_rate = 0
            note = (
                f"QPAM encode then decode of probability frames, {controls['shots']} shots each. "
                "Fewer shots leave a noisier sound."
            )
    except Exception as exc:
        st.error(f"Sonification failed: {exc}")
        return

    st.audio(wav, format="audio/wav")
    st.caption(note)
    if samples is not None:
        st.plotly_chart(
            _waveform_figure(samples, sample_rate),
            width="stretch",
            config={"displaylogo": False, "displayModeBar": False},
        )


@st.cache_data(show_spinner=False)
def _beat_payload(
    donor: np.ndarray,
    acceptor: np.ndarray,
    splitting_ev: float,
    pitch_hz: float,
) -> tuple[bytes, np.ndarray, int, str]:
    clip = tunneling_beat(
        donor,
        acceptor,
        splitting_ev,
        base_pitch_hz=pitch_hz,
        sample_rate=DEFAULT_SAMPLE_RATE,
    )
    return clip.wav_bytes, clip.samples, clip.sample_rate, clip.note


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
            go.Scatter(x=times_ms, y=audio[:window_n], name="Mono", line={"color": PLOT, "width": 1.5})
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=times_ms, y=audio[:window_n, 0], name="Left (donor)", line={"color": PLOT, "width": 1.5}
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times_ms,
                y=audio[:window_n, 1],
                name="Right (acceptor)",
                line={"color": PLOT_ALT, "width": 1.5},
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
        "font": {"color": INK, "family": FONT, "size": 12},
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
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
          [data-testid="stDecoration"],
          #MainMenu,
          [data-testid="stMainMenu"],
          [data-testid="stToolbarActions"],
          footer,
          .stDeployButton,
          [data-testid="stAppDeployButton"] { display: none !important; }
          [data-testid="stHeader"],
          [data-testid="stToolbar"] {
            background: transparent !important;
            border: none !important;
            height: 0 !important;
            min-height: 0 !important;
            overflow: visible !important;
          }
          [data-testid="stExpandSidebarButton"] {
            position: fixed !important;
            top: 0.55rem !important;
            left: 0.55rem !important;
            z-index: 1000 !important;
          }
          html, body, [class*="css"], .stApp, .stMarkdown, p, label, input, textarea,
          span:not([class*="material"]):not([data-testid="stIconMaterial"]),
          button:not(:has([class*="material"])):not(:has([data-testid="stIconMaterial"])) {
            font-family: "JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace !important;
          }
          span[class*="material"],
          [data-testid="stIconMaterial"],
          [data-testid="stSidebarCollapsedControl"],
          [data-testid="stSidebarCollapsedControl"] *,
          [data-testid="stExpandSidebarButton"],
          [data-testid="stExpandSidebarButton"] *,
          [data-testid="stSidebarNav"] span[class*="material"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
          }
          .stApp { background: #0b0b0d; color: #eceae4; }
          .block-container { padding-top: 1.1rem; max-width: 1180px; }
          h2, h3, p, li, label { color: #eceae4; }
          .story {
            font-weight: 500;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
            margin: 3.4rem 0 1.1rem;
            color: #eceae4;
          }
          .story span { color: #9aa2ff; margin-right: 0.55rem; }
          .listen-mode { margin: 0 0 0.6rem; color: #9aa2ff; font-size: 0.85rem; }
          .pipeline { display: flex; flex-direction: column; gap: 0.7rem; margin: 0.2rem 0 1.3rem; }
          .pipe { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; }
          .pipe em {
            color: #9aa2ff;
            font-style: normal;
            min-width: 4.8rem;
            font-size: 0.72rem;
          }
          .pipe span {
            border: 1px solid #2b35c8;
            background: #121216;
            color: #eceae4;
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            font-size: 0.78rem;
          }
          .pipe span.engine { background: #2b35c8; }
          .pipe i { color: #9aa2ff; font-style: normal; }
          .pipe-caption { color: #8d8b84; font-size: 0.85rem; margin: 0.15rem 0 0.2rem; }
          .ep-jobs { color: #8d8b84; font-size: 0.75rem; margin: 0.15rem 0 1.1rem; line-height: 1.45; }
          .piece-title {
            margin: 0.2rem 0 0.6rem;
            font-size: 1.05rem;
            letter-spacing: -0.02em;
            color: #eceae4;
          }
          .bank-row { margin: 0.12rem 0; font-size: 0.82rem; color: #eceae4; }
          .bank-row.pending { color: #6e6a78; }
          .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.9rem 0 0.4rem;
          }
          .metric-card {
            background: #121216;
            border: 1px solid #2b35c8;
            border-radius: 14px;
            padding: 0.8rem 0.95rem 0.85rem;
          }
          .metric-card span {
            display: block;
            color: #9aa2ff;
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
          }
          .metric-card strong {
            display: block;
            font-size: 1.15rem;
            font-weight: 500;
            margin-top: 0.28rem;
            color: #eceae4;
          }
          .metric-card em {
            display: block;
            color: #8d8b84;
            font-style: normal;
            font-size: 0.72rem;
            margin-top: 0.2rem;
          }
          [data-testid="stSidebar"] { background: #0b0b0d; }
          div[data-testid="stAlert"] { border-radius: 14px; }
          iframe[title="streamlit_components_v1.html"] { border-radius: 12px; }
          @media (max-width: 900px) {
            .metric-grid { grid-template-columns: 1fr 1fr; }
            .story { margin-top: 2.4rem; font-size: 1.05rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
